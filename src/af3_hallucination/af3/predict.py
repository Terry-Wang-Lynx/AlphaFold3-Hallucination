"""AF3DesignModel.predict() glue: gradient-free official ModelRunner path.

predict() is the BindCraft/ColabDesign `predict(backprop=False)` analogue (note
017): a GRADIENT-FREE wrapper around the OFFICIAL AF3 inference path
(trunk + diffusion + ConfidenceHead + ranking) that scores a single hard
candidate. It is COMPLETELY DECOUPLED from the differentiable trunk-only design
loop -- it does NOT call runner.forward() / _loss_from_logits() / optimize() /
the DesignEvoformer soft trunk. The two paths share only masks (012) + config +
af3.bin weights, never a forward function (017 section 2).

Two hard constraints (017 section 0):
  - `ranking_score` is REPORT/FILTER ONLY, never a scoring_loss term.
  - `iptm` is a FORWARD scalar for semigreedy scoring only, never a gradient
    mixed-loss term (P2-F/P2-F2 unchanged; this path has no gradients at all).

Policy:
  - "refeaturise" (exact gate/scoring policy): rewrite the binder chain sequence
    letters and featurise the mutated input (exact atom layout). Requires
    binder_chain_id.
  - "token_only" (cheap diagnostic policy): featurise the original ONCE at the
    predict bucket, then inject_hard_sequence (predict_injection.py) on the design
    positions. This is only layout-exact for token-level fields.

AF3/jax imports are deferred to call time so this module is import-light; scoring
/ losses / masks (jnp) and predict_injection (numpy) are imported at module load.
"""

from __future__ import annotations

import contextlib
import copy
import json
import pathlib
import tempfile
import time

import numpy as np

from . import losses as lossmod
from . import masks as maskmod
from . import predict_injection as pinj
from . import scoring as scoremod

# AF3 protein restype order (residue_names.PROTEIN_TYPES_ONE_LETTER /
# POLYMER_TYPES_ORDER_WITH_UNKNOWN_AND_GAP: ids 0-19 are protein, 20=UNK, 21=gap,
# 22-30 nucleic). Hard candidate aa ids must be designable protein residues.
PROTEIN_ONE_LETTER = ("A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
                      "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V")


def validate_protein_seq_indices(seq_indices, *, expected_len=None, name="seq_indices") -> np.ndarray:
    """Validate hard design candidates as protein residue ids [0, 20)."""
    seq = np.asarray(seq_indices)
    if seq.ndim != 1:
        raise ValueError(f"{name}: expected 1-D hard candidate, got shape {seq.shape}.")
    if not np.issubdtype(seq.dtype, np.integer):
        raise TypeError(f"{name}: expected integer aa ids, got dtype {seq.dtype}.")
    seq = seq.astype(np.int32, copy=False)
    if expected_len is not None and int(seq.shape[0]) != int(expected_len):
        raise ValueError(
            f"{name}: length {seq.shape[0]} != expected design length {expected_len}.")
    bad = seq[(seq < 0) | (seq >= len(PROTEIN_ONE_LETTER))]
    if bad.size:
        raise ValueError(
            f"{name}: non-protein aa ids {bad[:5].tolist()} are not designable; "
            f"expected ids in [0, {len(PROTEIN_ONE_LETTER)}).")
    return seq


def seq_indices_to_letters(seq_indices) -> list:
    """Map AF3 protein aatype ids -> one-letter codes (for the refeaturise path).

    Raises on any id outside the protein range [0, 20): UNK/gap/nucleic tokens are
    not designable protein residues and must never reach the binder sequence.
    """
    letters = []
    for x in np.asarray(seq_indices).tolist():
        i = int(x)
        if not (0 <= i < len(PROTEIN_ONE_LETTER)):
            raise ValueError(
                f"seq_indices_to_letters: aa id {i} is not a designable protein "
                f"residue in [0, {len(PROTEIN_ONE_LETTER)}).")
        letters.append(PROTEIN_ONE_LETTER[i])
    return letters


def _wrap_sequence(sequence: str, width: int = 80) -> list[str]:
    return [sequence[i:i + width] for i in range(0, len(sequence), width)] or [""]


def replace_first_fasta_sequence(msa: str, query_sequence: str) -> str:
    """Replace the first inline-MSA query record with the mutated sequence."""
    if not isinstance(msa, str):
        raise TypeError(f"MSA must be a string, got {type(msa).__name__}.")
    if not msa.strip():
        return msa

    lines = msa.splitlines()
    trailing_newline = "\n" if msa.endswith("\n") else ""
    if not lines:
        return msa

    if lines[0].startswith(">"):
        out = [lines[0]]
        i = 1
        while i < len(lines) and not lines[i].startswith(">"):
            i += 1
        out.extend(_wrap_sequence(query_sequence))
        out.extend(lines[i:])
        return "\n".join(out) + trailing_newline

    # Defensive fallback for non-FASTA inline MSA text: make the first line match
    # the query and keep any remaining lines untouched.
    return "\n".join([query_sequence] + lines[1:]) + trailing_newline


def _protein_id_matches(protein_id, binder_chain_id: str) -> bool:
    if protein_id == binder_chain_id:
        return True
    if isinstance(protein_id, list):
        return binder_chain_id in protein_id
    return False


def prepare_predict_batch(policy, *, orig_batch=None, seq_indices, design_indices,
                          is_ligand=None, fold_input_path=None, binder_chain_id=None,
                          binder_token_start=0, bucket=None, width=31,
                          refeaturise_fn=None):
    """Dispatch the predict batch by policy (token_only inject vs refeaturise).

    token_only: inject_hard_sequence on a copy of orig_batch (cheap, numpy).
    refeaturise: requires binder_chain_id; rewrites the binder sequence and
      re-featurises (exact). refeaturise_fn is injectable for testing; defaults to
      the real _refeaturise_candidate. Raises a clear error (never silently
      mis-uses token indices) when binder_chain_id is missing.
    """
    seq_indices = validate_protein_seq_indices(
        seq_indices, expected_len=len(np.asarray(design_indices)), name="seq_indices")
    if policy == "token_only":
        if orig_batch is None:
            raise ValueError("prepare_predict_batch(token_only): orig_batch required.")
        return pinj.inject_hard_sequence(orig_batch, seq_indices, design_indices,
                                         is_ligand=is_ligand, width=width)
    if policy == "refeaturise":
        if binder_chain_id is None:
            raise ValueError(
                "predict(policy='refeaturise') requires binder_chain_id to rewrite "
                "the binder chain sequence; refusing to guess from token indices.")
        fn = refeaturise_fn or _refeaturise_candidate
        return fn(fold_input_path=fold_input_path, binder_chain_id=binder_chain_id,
                  binder_token_start=binder_token_start, design_indices=design_indices,
                  seq_indices=seq_indices, bucket=bucket)
    raise ValueError(f"predict: unknown policy {policy!r} "
                     "(expected 'token_only' or 'refeaturise').")


def _featurise(fold_input_path, bucket):
    """Featurise a fold input at a given bucket (deferred AF3 imports)."""
    from alphafold3.common import folding_input
    from alphafold3.constants import chemical_components
    from alphafold3.data import featurisation
    fi = list(folding_input.load_fold_inputs_from_path(pathlib.Path(fold_input_path)))[0]
    return featurisation.featurise_input(
        fold_input=fi, ccd=chemical_components.Ccd(), buckets=[bucket])[0]


def _refeaturise_candidate(*, fold_input_path, binder_chain_id, binder_token_start,
                           design_indices, seq_indices, bucket):
    """Rewrite binder chain letters, sync inline MSA query rows, and re-featurise."""
    seq_indices = validate_protein_seq_indices(
        seq_indices, expected_len=len(np.asarray(design_indices)), name="seq_indices")
    raw = json.loads(pathlib.Path(fold_input_path).read_text())
    raw2 = copy.deepcopy(raw)
    letters = seq_indices_to_letters(seq_indices)
    found = False
    mutated = None
    for s in raw2["sequences"]:
        protein = s.get("protein")
        if not protein or not _protein_id_matches(protein.get("id"), binder_chain_id):
            continue
        found = True
        bseq = list(protein["sequence"])
        for di, letter in zip(np.asarray(design_indices).tolist(), letters, strict=True):
            local = int(di) - int(binder_token_start)
            if not (0 <= local < len(bseq)):
                raise ValueError(
                    f"_refeaturise_candidate: design token {di} maps to "
                    f"binder-local {local}, outside chain length {len(bseq)}.")
            bseq[local] = letter
        mutated = "".join(bseq)
        protein["sequence"] = mutated

        for msa_key in ("unpairedMsa", "pairedMsa"):
            if msa_key in protein and protein[msa_key] is not None:
                protein[msa_key] = replace_first_fasta_sequence(protein[msa_key], mutated)

        for msa_path_key in ("unpairedMsaPath", "pairedMsaPath"):
            if protein.get(msa_path_key):
                raise ValueError(
                    "_refeaturise_candidate cannot safely mutate path-backed MSA "
                    f"field {msa_path_key!r}; inline MSA is required.")

    if not found:
        raise ValueError(f"_refeaturise_candidate: binder chain '{binder_chain_id}' "
                         "not found among protein sequences.")
    if mutated is None:
        raise ValueError("_refeaturise_candidate: mutated binder sequence was not created.")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix=f"af3predict_refeat_{binder_chain_id}_",
        delete=False) as fh:
        tmp = pathlib.Path(fh.name)
        json.dump(raw2, fh)
        fh.write("\n")
    try:
        return _featurise(str(tmp), bucket)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()


def build_scoring_dict(*, con, i_con, helix, binder_plddt, mean_plddt, pae, i_pae,
                       iptm, per_token_plddt, ranking_score, ptm, has_clash,
                       fraction_disordered, chain_pair_pae_min, policy, bucket,
                       diffusion_steps, num_samples, runtime_sec, peak_gpu_gb,
                       weights=None) -> dict:
    """Assemble the predict() return dict (017 section 1c).

    scoring_loss = scoring.scoring_loss (BindCraft weights; iptm enters as a
    forward scalar via 1-iptm). ranking_score is report-only and is NEVER passed
    into scoring_loss.
    """
    weights = weights if weights is not None else scoremod.ScoringWeights()
    scoring_loss, comps = scoremod.scoring_loss(
        con, i_con, pae, i_pae, binder_plddt, iptm, helix=helix, weights=weights)
    return {
        # ---- design-loss inputs (semigreedy scoring) ----
        "con": float(con), "i_con": float(i_con),
        "helix": (None if helix is None else float(helix)),
        "binder_plddt": float(binder_plddt), "mean_plddt": float(mean_plddt),
        "pae": float(pae), "i_pae": float(i_pae), "iptm": float(iptm),
        "scoring_loss": float(scoring_loss),
        "scoring_components": comps,
        # ---- semigreedy position prior (per-token, not just scalar) ----
        "per_token_plddt": per_token_plddt,
        # ---- report-only (filter/gate/ranking; NEVER design loss) ----
        "ranking_score": float(ranking_score), "ptm": float(ptm),
        "has_clash": bool(has_clash), "fraction_disordered": float(fraction_disordered),
        "chain_pair_pae_min": chain_pair_pae_min,
        # ---- engineering ----
        "policy": policy, "bucket": int(bucket), "diffusion_steps": int(diffusion_steps),
        "num_samples": int(num_samples),
        "runtime_sec": runtime_sec, "peak_gpu_gb": peak_gpu_gb,
    }


def build_context(*, fold_input_path, prep_spec, design_cfg, model_dir, bucket,
                  diffusion_steps, num_samples, num_recycles, flash_attention="xla"):
    """Featurise the original at the predict bucket and build the ModelRunner.

    Independent of the design-loop bucket: predict needs bucket>=256 + xla for the
    diffusion/confidence path (P2-A GLU blocker at 64). Returns a context reused
    across candidates so run_inference compiles once.

    `num_recycles` is an EXPLICIT argument (no longer hardcoded to 0): the cheap
    semigreedy scorer uses 0 while the higher-fidelity final validation uses 3
    (BindCraft validation parity, note: recycle/dropout/model-sampling spec). It is
    written into the context so callers can record/diff it.
    """
    import jax
    import jax.numpy as jnp
    import run_alphafold as ra

    orig_batch = _featurise(fold_input_path, bucket)
    seq_mask = jnp.asarray(orig_batch["seq_mask"])
    asym_id = jnp.asarray(orig_batch["asym_id"]) if "asym_id" in orig_batch else None
    is_ligand = jnp.asarray(orig_batch["is_ligand"]) if "is_ligand" in orig_batch else None
    masks = maskmod.build_masks(
        seq_mask=seq_mask,
        binder_token_range=prep_spec.get("binder_token_range"),
        hotspot_token_indices=prep_spec.get("hotspot_token_indices", ()),
        binder_mask=prep_spec.get("binder_mask"),
        binder_asym_id=prep_spec.get("binder_asym_id"),
        asym_id=asym_id, is_ligand=is_ligand,
        design_token_indices=prep_spec.get("design_token_indices"),
        target_token_indices=prep_spec.get("target_token_indices"))

    mcfg = ra.make_model_config(num_diffusion_samples=num_samples,
                                num_recycles=num_recycles,
                                return_distogram=True,
                                flash_attention_implementation=flash_attention)
    mcfg.heads.diffusion.eval.steps = diffusion_steps
    runner = ra.ModelRunner(config=mcfg, device=jax.devices()[0],
                            model_dir=pathlib.Path(model_dir))

    binder_idx = np.flatnonzero(np.asarray(masks.binder_mask).astype(bool))
    return {
        "orig_batch": orig_batch,
        "masks": masks,
        "design_indices": np.asarray(masks.design_indices),
        "binder_indices": binder_idx,
        "binder_token_start": int(binder_idx.min()) if binder_idx.size else 0,
        "residue_index": jnp.asarray(orig_batch["residue_index"]),
        "atom_mask": jnp.asarray(np.asarray(orig_batch["pred_dense_atom_mask"]).astype(np.float32)),
        "is_ligand": (np.asarray(orig_batch["is_ligand"]) if "is_ligand" in orig_batch else None),
        "runner": runner,
        "design_cfg": design_cfg,
        "fold_input_path": fold_input_path,
        "bucket": bucket, "diffusion_steps": diffusion_steps, "num_samples": num_samples,
        "num_recycles": num_recycles,
    }


def predict_candidate(ctx, *, seq_indices=None, policy="token_only",
                      binder_chain_id=None, seed=0, key=None, weights=None,
                      return_raw=False, return_arrays=False) -> dict:
    """Run ONE gradient-free official predict for a single hard candidate."""
    import jax
    import jax.numpy as jnp

    design_indices = ctx["design_indices"]
    orig_batch = ctx["orig_batch"]
    masks = ctx["masks"]
    if seq_indices is None:
        seq_indices = np.asarray(orig_batch["aatype"])[design_indices].astype(np.int32)
    seq_indices = validate_protein_seq_indices(
        seq_indices, expected_len=len(design_indices), name="seq_indices")

    batch = prepare_predict_batch(
        policy, orig_batch=orig_batch, seq_indices=seq_indices,
        design_indices=design_indices, is_ligand=ctx["is_ligand"],
        fold_input_path=ctx["fold_input_path"], binder_chain_id=binder_chain_id,
        binder_token_start=ctx["binder_token_start"], bucket=ctx["bucket"])

    runner = ctx["runner"]
    if key is None:
        key = jax.random.PRNGKey(seed)
    t0 = time.time()
    result = runner.run_inference(batch, key)
    jax.block_until_ready(result.get("predicted_lddt"))
    runtime_sec = round(time.time() - t0, 2)
    md = runner.extract_inference_results(batch, result, target_name="predict")[0].metadata

    binder = masks.binder_mask
    target = masks.target_mask
    valid = (jnp.asarray(orig_batch["seq_mask"]) > 0).astype(jnp.float32)
    atom_mask = ctx["atom_mask"]

    distogram_out = result.get("distogram", {})
    distogram_logits_available = (
        isinstance(distogram_out, dict)
        and "distogram" in distogram_out
        and "bin_edges" in distogram_out
    )
    if distogram_logits_available:
        _, aux = lossmod.total_loss(distogram_out, ctx["residue_index"], masks,
                                    ctx["design_cfg"])
        con = float(aux["con"])
        i_con = float(aux["i_con"])
        helix = float(aux["helix"]) if "helix" in aux else None
    else:
        # Some AF3 runtime builds expose confidence/distogram summaries but not
        # the full 64-bin distogram logits. Gate decisions use confidence
        # metrics, so missing logits should not abort the full-diffusion gate.
        con = 0.0
        i_con = 0.0
        helix = None
    pl = scoremod.plddt_metrics(result["predicted_lddt"], atom_mask, binder, valid)
    pm = scoremod.pae_metrics(result["full_pae"], binder, target, valid)
    tok = np.asarray(scoremod.token_plddt(result["predicted_lddt"], atom_mask))
    per_token_plddt = {
        "binder": [float(x) for x in tok[ctx["binder_indices"]]],
        "design": [float(x) for x in tok[design_indices]],
    }

    def _f(k):
        return float(np.asarray(md[k]))

    cpm = md.get("chain_pair_pae_min") if hasattr(md, "get") else md["chain_pair_pae_min"]
    out = build_scoring_dict(
        con=con, i_con=i_con, helix=helix,
        binder_plddt=pl["binder_plddt"], mean_plddt=pl["mean_plddt"],
        pae=pm["pae_loss"], i_pae=pm["i_pae_loss"], iptm=_f("iptm"),
        per_token_plddt=per_token_plddt,
        ranking_score=_f("ranking_score"), ptm=_f("ptm"),
        has_clash=bool(np.asarray(md["has_clash"])),
        fraction_disordered=_f("fraction_disordered"),
        chain_pair_pae_min=np.asarray(cpm).astype(float).tolist(),
        policy=policy, bucket=ctx["bucket"], diffusion_steps=ctx["diffusion_steps"],
        num_samples=ctx["num_samples"], runtime_sec=runtime_sec,
        peak_gpu_gb=_peak_gpu_gb(), weights=weights)
    out["num_recycles"] = ctx["num_recycles"]
    out["seq_indices"] = [int(x) for x in seq_indices]
    out["distogram_logits_available"] = bool(distogram_logits_available)
    if return_raw:
        out["raw"] = {
            "predicted_lddt_shape": list(np.asarray(result["predicted_lddt"]).shape),
            "full_pae_shape": list(np.asarray(result["full_pae"]).shape),
            "pae_mean_raw": pm["pae_mean"], "interface_pae_mean_raw": pm["interface_pae_mean"],
        }
    if return_arrays:
        from alphafold3.model.scoring import scoring as af_scoring

        dense_positions = np.asarray(result["diffusion_samples"]["atom_positions"])
        pos0 = dense_positions[0] if dense_positions.ndim == 4 else dense_positions
        dense_mask = np.asarray(batch["pred_dense_atom_mask"]).astype(np.float32)
        is_ligand = np.asarray(batch["is_ligand"]) if "is_ligand" in batch else None
        pseudo_beta, pseudo_beta_mask = af_scoring.pseudo_beta_fn(
            np.asarray(batch["aatype"]).astype(np.int32),
            pos0,
            dense_mask,
            is_ligand=is_ligand,
            use_jax=False,
        )
        out["_arrays"] = {
            "distogram_contact_probs": np.asarray(result["distogram"]["contact_probs"]),
            "distogram_bin_edges": np.asarray(result["distogram"]["bin_edges"]),
            "distogram_logits": np.asarray(result["distogram"].get("distogram")),
            "pseudo_beta": np.asarray(pseudo_beta),
            "pseudo_beta_mask": np.asarray(pseudo_beta_mask),
            "seq_mask": np.asarray(batch["seq_mask"]),
            "asym_id": np.asarray(batch["asym_id"]),
            "residue_index": np.asarray(batch["residue_index"]),
        }
    return out


def _peak_gpu_gb():
    try:
        import jax
        return round(jax.devices()[0].memory_stats().get("peak_bytes_in_use", 0) / 1e9, 2)
    except Exception:  # noqa: BLE001
        return None

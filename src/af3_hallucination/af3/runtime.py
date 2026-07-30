"""First-party AF3 diffusion, Consistency, and final-evaluation runtimes.

All heavy imports are local to public functions. The core package therefore
remains usable on macOS without an AlphaFold 3 installation.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import HallucinationSpec
from ..contracts import normalize_candidate_pool
from ..hashing import sha256_file
from .engine import AA_INDEX, AF3_AA_ORDER


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def _platform() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "physical_gpu": os.environ.get("AF3_PHYSICAL_GPU", "unknown"),
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _protein_sequence(path: Path, chain_id: str) -> str:
    document = _load_json(path)
    matches = []
    for entry in document.get("sequences", []):
        protein = entry.get("protein")
        identifier = None if protein is None else protein.get("id")
        if protein and (identifier == chain_id or (isinstance(identifier, list) and chain_id in identifier)):
            matches.append(str(protein["sequence"]))
    if len(matches) != 1:
        raise ValueError(f"expected one protein chain {chain_id!r} in {path}")
    return matches[0]


def _raw_and_clean_batch(input_json: Path, bucket: int):
    import jax
    import jax.numpy as jnp
    from alphafold3.common import folding_input
    from alphafold3.constants import chemical_components
    from alphafold3.data import featurisation
    from alphafold3.model.components import utils

    fold_input = list(folding_input.load_fold_inputs_from_path(input_json))[0]
    raw = featurisation.featurise_input(
        fold_input=fold_input,
        ccd=chemical_components.Ccd(),
        buckets=[bucket],
    )[0]
    clean = utils.remove_invalidly_typed_feats(raw)
    return raw, jax.tree_util.tree_map(jnp.asarray, clean)


def _token_layout(batch, *, binder_asym_id: int):
    from .hybrid import binder_mask_from_batch

    valid = np.asarray(batch["seq_mask"]) > 0
    binder = binder_mask_from_batch(batch, binder_asym_id)
    return valid, binder, np.flatnonzero(binder)


def _resolve_indices(batch, spec: HallucinationSpec):
    backend = spec.backend_config
    binder_asym_id = int(backend.get("binder_asym_id", 2))
    target_asym_id = int(backend.get("target_asym_id", 1))
    valid, binder, binder_indices = _token_layout(batch, binder_asym_id=binder_asym_id)
    asym = np.asarray(batch["asym_id"], np.int32)
    if "design_token_indices" in backend and "design_local_indices" in backend:
        raise ValueError("provide only one of design_token_indices/design_local_indices")
    if "design_token_indices" in backend:
        design = np.asarray(backend["design_token_indices"], np.int32)
        local_lookup = {int(token): index for index, token in enumerate(binder_indices)}
        try:
            design_local = np.asarray([local_lookup[int(token)] for token in design], np.int32)
        except KeyError as exc:
            raise ValueError("a design token is outside the binder chain") from exc
    elif "design_local_indices" in backend:
        design_local = np.asarray(backend["design_local_indices"], np.int32)
        if np.any(design_local < 0) or np.any(design_local >= binder_indices.size):
            raise ValueError("a design-local index is outside the binder chain")
        design = binder_indices[design_local]
    else:
        design_local = np.arange(binder_indices.size, dtype=np.int32)
        design = binder_indices
    if "hotspot_token_indices" in backend and "hotspot_local_indices" in backend:
        raise ValueError("provide only one of hotspot_token_indices/hotspot_local_indices")
    if "hotspot_token_indices" in backend:
        hotspot = np.asarray(backend["hotspot_token_indices"], np.int32)
    elif "hotspot_local_indices" in backend:
        target = np.flatnonzero(valid & (asym == target_asym_id))
        local = np.asarray(backend["hotspot_local_indices"], np.int32)
        if np.any(local < 0) or np.any(local >= target.size):
            raise ValueError("a hotspot-local index is outside the target chain")
        hotspot = target[local]
    else:
        hotspot = np.flatnonzero(valid & ~binder)
    return binder_asym_id, binder_indices, design, design_local, hotspot


def _hybrid_features(batch, design_indices, pseudo20, query20, width: int):
    import jax
    import jax.numpy as jnp

    pseudo31 = jnp.pad(jnp.asarray(pseudo20), ((0, 0), (0, width - 20)))
    query31 = jnp.pad(jnp.asarray(query20), ((0, 0), (0, width - 20)))
    indices = jnp.asarray(design_indices, jnp.int32)
    hard_ids = jnp.asarray(batch["aatype"])[indices]
    hard_onehot = jax.nn.one_hot(hard_ids, width)
    hard_profile = jnp.asarray(batch["profile"])[indices]
    query_weight = jnp.sum(hard_profile * hard_onehot, axis=-1, keepdims=True)
    expected_profile = hard_profile + query_weight * (query31 - hard_onehot)
    n_tokens = int(np.asarray(batch["seq_mask"]).size)
    pseudo_full = jnp.zeros((n_tokens, width), pseudo31.dtype).at[indices].set(pseudo31)
    profile_full = jnp.zeros((n_tokens, width), expected_profile.dtype).at[indices].set(
        expected_profile
    )
    return pseudo_full, profile_full


def _pseudo_beta_from_dense(batch_obj, dense_positions, batch) -> tuple[np.ndarray, np.ndarray]:
    info = batch_obj.pseudo_beta_info.token_atoms_to_pseudo_beta
    gather = np.asarray(info.gather_idxs, np.int64)
    layout = np.asarray(info.gather_mask, bool)
    positions = np.asarray(dense_positions, np.float32).reshape((-1, 3))[gather]
    atom_mask = np.asarray(batch["pred_dense_atom_mask"], bool).reshape(-1)
    valid = layout & atom_mask[gather] & (np.asarray(batch["seq_mask"]) > 0)
    return positions, valid


def decode_checkpoint(
    *,
    public_spec: HallucinationSpec,
    input_json: str | Path,
    checkpoints_json: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    checkpoint_forward: int | None,
    trunk_seed: int,
    diffusion_seed: int,
    diffusion_steps: int,
    num_recycles: int,
) -> dict[str, Any]:
    """Decode one exact pre-update Hallucination checkpoint with AF3 diffusion."""

    import haiku as hk
    import jax
    import jax.numpy as jnp
    import run_alphafold as run_af3
    from alphafold3.model import feat_batch, post_processing
    from alphafold3.model import model as af_model
    from alphafold3.model import params as af_params

    from .config import RematConfig
    from .hybrid import HybridInferenceModel, candidate_input
    from .provenance import parameter_identifier, runtime_source_fingerprint
    from .sequence import SeqOpt, aa_bias, soft_seq
    from .trunk import WIDTH

    input_path = Path(input_json).resolve()
    checkpoint_path = Path(checkpoints_json).resolve()
    model_path = Path(model_dir).resolve()
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = _load_json(checkpoint_path)
    records = list(manifest.get("checkpoints", []))
    if not records:
        raise ValueError("Hallucination produced no checkpoint records")
    if checkpoint_forward is None:
        checkpoint = max(records, key=lambda row: int(row["evaluation_forward"]))
    else:
        matches = [
            row for row in records if int(row["evaluation_forward"]) == checkpoint_forward
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one checkpoint at forward {checkpoint_forward}")
        checkpoint = matches[0]
    logits_path = checkpoint_path.parent / checkpoint["path"]
    if sha256_file(logits_path) != checkpoint["sha256"]:
        raise ValueError("checkpoint logits hash mismatch")
    logits = np.asarray(np.load(logits_path, allow_pickle=False), np.float32)
    if not np.isfinite(logits).all():
        raise ValueError("checkpoint logits contain non-finite values")

    binder_chain = str(public_spec.backend_config.get("binder_chain_id", "B"))
    binder_sequence = _protein_sequence(input_path, binder_chain)
    _, initial_batch = _raw_and_clean_batch(input_path, public_spec.bucket)
    binder_asym_id, _, design, design_local, hotspot = _resolve_indices(
        initial_batch, public_spec
    )
    if logits.shape != (len(design), 20):
        raise ValueError(
            f"checkpoint logits shape {logits.shape} does not match {(len(design), 20)}"
        )
    options = checkpoint["sequence_options"]
    bias = aa_bias(20, tuple(AA_INDEX[aa] for aa in public_spec.omit_amino_acids))
    state = soft_seq(
        jnp.asarray(logits),
        SeqOpt(
            soft=float(options["soft"]),
            temp=float(options.get("temperature", options.get("temp"))),
            hard=float(options["hard"]),
            alpha=float(options.get("alpha", public_spec.alpha)),
        ),
        bias,
    )
    pseudo20 = np.asarray(state["pseudo"], np.float32)
    query20 = np.asarray(
        float(options["hard"]) * state["hard"]
        + (1.0 - float(options["hard"])) * state["soft"],
        np.float32,
    )
    hard_ids = np.asarray([AA_INDEX[aa] for aa in binder_sequence], np.int32)
    hard_ids[design_local] = np.argmax(pseudo20, axis=-1)
    hard_sequence = "".join(AF3_AA_ORDER[int(value)] for value in hard_ids)
    candidate_path = candidate_input(
        input_path, hard_ids, root / "anchor_input.json", binder_chain=binder_chain
    )
    raw_batch, batch = _raw_and_clean_batch(candidate_path, public_spec.bucket)
    _, binder, _ = _token_layout(batch, binder_asym_id=binder_asym_id)
    if not np.array_equal(np.asarray(batch["aatype"])[binder], hard_ids):
        raise ValueError("anchor hard-carrier atom chemistry is stale")
    pseudo_full, profile_full = _hybrid_features(
        batch, design, pseudo20, query20, WIDTH
    )
    design_mask = np.zeros(np.asarray(batch["seq_mask"]).size, bool)
    design_mask[design] = True
    model_config = run_af3.make_model_config(
        num_diffusion_samples=1,
        num_recycles=num_recycles,
        return_distogram=True,
        flash_attention_implementation=str(
            public_spec.backend_config.get("flash_attention", "xla")
        ),
    )
    model_config.heads.diffusion.eval.steps = diffusion_steps

    @hk.transform
    def forward_model(batch_value, pseudo_value, profile_value, key_value, diffusion_key):
        return HybridInferenceModel(
            model_config,
            RematConfig(enabled=False),
            jnp.asarray(design_mask),
            num_design_recycles=public_spec.design_recycles,
            detach_recycles=False,
        )(batch_value, pseudo_value, profile_value, key_value, diffusion_key)

    parameters = af_params.get_model_haiku_params(model_dir=model_path)
    started = time.monotonic()
    result = jax.jit(forward_model.apply)(
        parameters,
        jax.random.PRNGKey(trunk_seed),
        batch,
        pseudo_full,
        profile_full,
        jax.random.PRNGKey(trunk_seed),
        jax.random.PRNGKey(diffusion_seed),
    )
    jax.block_until_ready(result["predicted_lddt"])
    result = jax.tree_util.tree_map(np.asarray, result)
    result = jax.tree_util.tree_map(
        lambda value: value.astype(np.float32)
        if getattr(value, "dtype", None) == jnp.bfloat16
        else value,
        result,
    )
    result = dict(result)
    result["__identifier__"] = parameters["__meta__"]["__identifier__"].tobytes()
    extracted = list(
        af_model.Model.get_inference_result(
            batch=raw_batch,
            result=result,
            target_name=f"af3h_anchor_f{int(checkpoint['evaluation_forward'])}",
        )
    )
    if len(extracted) != 1:
        raise ValueError(f"expected one AF3 anchor output, got {len(extracted)}")
    post_processing.write_output(extracted[0], root, name="anchor")
    structures = sorted(root.glob("anchor*.cif"))
    if len(structures) != 1:
        raise ValueError(f"expected one anchor CIF, got {len(structures)}")
    dense_positions = np.asarray(result["diffusion_samples"]["atom_positions"])[0]
    pseudo_beta, geometry_valid = _pseudo_beta_from_dense(
        feat_batch.Batch.from_data_dict(batch), dense_positions, batch
    )
    geometry_path = root / "anchor_geometry.npz"
    np.savez_compressed(
        geometry_path,
        pseudo_beta=np.asarray(pseudo_beta, np.float32),
        geometry_valid_mask=np.asarray(geometry_valid, bool),
    )
    metadata = extracted[0].metadata
    model_parameter_identifier = parameter_identifier(parameters)
    elapsed = time.monotonic() - started
    manifest_path = root / "anchor_manifest.json"
    manifest_payload = {
        "schema_version": "af3h_anchor_v1",
        "status": "completed",
        "input_json": str(input_path),
        "input_sha256": sha256_file(input_path),
        "candidate_input_path": str(candidate_path),
        "candidate_input_sha256": sha256_file(candidate_path),
        "structure_path": str(structures[0]),
        "structure_sha256": sha256_file(structures[0]),
        "geometry_path": str(geometry_path),
        "geometry_sha256": sha256_file(geometry_path),
        "hard_sequence": hard_sequence,
        "binder_chain_id": binder_chain,
        "binder_asym_id": binder_asym_id,
        "design_token_indices": [int(value) for value in design],
        "design_local_indices": [int(value) for value in design_local],
        "hotspot_token_indices": [int(value) for value in hotspot],
        "bucket": public_spec.bucket,
        "checkpoint_evaluation_forward": int(checkpoint["evaluation_forward"]),
        "checkpoint_logits_sha256": checkpoint["sha256"],
        "sequence_options": options,
        "trunk_seed": trunk_seed,
        "diffusion_seed": diffusion_seed,
        "diffusion_steps": diffusion_steps,
        "num_recycles": num_recycles,
        "af3_parameter_identifier": model_parameter_identifier,
        "af3_source_commit_audited": "b2f3d45fbfcacc5183bd5345d15df93571b8437f",
        "af3_runtime": runtime_source_fingerprint(),
        "ptm": float(np.asarray(metadata["ptm"])),
        "iptm": float(np.asarray(metadata["iptm"])),
        "ranking_score": float(np.asarray(metadata["ranking_score"])),
        "has_clash": bool(np.asarray(metadata["has_clash"])),
        "runtime_seconds": elapsed,
        "platform": _platform(),
    }
    _write_json(manifest_path, manifest_payload)
    return {
        "manifest_path": manifest_path,
        "structure_path": structures[0],
        "geometry_path": geometry_path,
        "candidate_input_path": candidate_path,
        "metrics": {
            "runtime_seconds": elapsed,
            "ptm": manifest_payload["ptm"],
            "iptm": manifest_payload["iptm"],
            "has_clash": manifest_payload["has_clash"],
            "full_af3_calls": 1,
        },
    }


def _threshold_eligibility(metrics: dict[str, float], thresholds: dict[str, Any]):
    rules = {
        "consistency_loss_max": ("consistency_loss", lambda value, threshold: value <= threshold),
        "all_cdr_plddt_min": ("all_cdr_plddt", lambda value, threshold: value >= threshold),
        "cdr3_plddt_min": ("cdr3_plddt", lambda value, threshold: value >= threshold),
        "cdr_patch_pae_max": ("cdr_patch_pae_symmetric", lambda value, threshold: value <= threshold),
        "cdr_patch_pde_max": ("cdr_patch_pde_symmetric", lambda value, threshold: value <= threshold),
    }
    unknown = set(thresholds) - set(rules)
    if unknown:
        raise ValueError(f"unknown Consistency threshold keys: {sorted(unknown)}")
    checks = {
        name: bool(compare(metrics[metric], float(thresholds[name])))
        for name, (metric, compare) in rules.items()
        if name in thresholds
    }
    return (all(checks.values()) if checks else True), checks


def score_consistency_pool(
    *,
    anchor_manifest: str | Path,
    candidate_pool: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    scorer_seed: int,
    design_recycles: int,
    cdr3_local_indices: tuple[int, ...],
    thresholds: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    """Rank CDR sequences with fixed anchor pseudo-beta geometry and no diffusion."""

    import haiku as hk
    import jax
    import jax.numpy as jnp
    import run_alphafold as run_af3
    from alphafold3.model import feat_batch
    from alphafold3.model import params as af_params

    from .config import RematConfig
    from .consistency import (
        conditioned_confidence_model_class,
        four_term_consistency_loss,
        scatter_pseudo_beta,
    )
    from .hybrid import binder_mask_from_batch, candidate_input
    from .provenance import parameter_identifier, runtime_source_fingerprint
    from .trunk import WIDTH

    anchor_path = Path(anchor_manifest).resolve()
    anchor = _load_json(anchor_path)
    pool_path = Path(candidate_pool).resolve()
    pool = _load_json(pool_path)
    candidates = normalize_candidate_pool(
        pool,
        reference_sequence=anchor["hard_sequence"],
        design_local_indices=anchor["design_local_indices"],
    )
    if top_k < 1:
        raise ValueError("Consistency top_k must be positive")
    geometry_path = Path(anchor["geometry_path"])
    if sha256_file(geometry_path) != anchor["geometry_sha256"]:
        raise ValueError("anchor geometry hash mismatch")
    geometry = np.load(geometry_path, allow_pickle=False)
    pseudo_beta = np.asarray(geometry["pseudo_beta"], np.float32)
    geometry_valid = np.asarray(geometry["geometry_valid_mask"], bool)
    base_input = Path(anchor["candidate_input_path"])
    binder_chain = str(anchor["binder_chain_id"])
    binder_asym_id = int(anchor["binder_asym_id"])
    design = np.asarray(anchor["design_token_indices"], np.int32)
    design_local = np.asarray(anchor["design_local_indices"], np.int32)
    _, initial_batch = _raw_and_clean_batch(base_input, int(anchor["bucket"]))
    valid, binder, binder_indices = _token_layout(
        initial_batch, binder_asym_id=binder_asym_id
    )
    if pseudo_beta.shape != (valid.size, 3) or geometry_valid.shape != valid.shape:
        raise ValueError("anchor geometry layout differs from the AF3 token batch")
    cdr = np.zeros(valid.size, bool)
    cdr[design] = True
    cdr3 = np.zeros(valid.size, bool)
    if cdr3_local_indices:
        local = np.asarray(cdr3_local_indices, np.int32)
        if np.any(local < 0) or np.any(local >= binder_indices.size):
            raise ValueError("cdr3_local_indices contain an out-of-range value")
        cdr3[binder_indices[local]] = True
        if np.any(cdr3 & ~cdr):
            raise ValueError("cdr3_local_indices must be a subset of design positions")
    else:
        cdr3 = cdr.copy()
    patch = np.zeros(valid.size, bool)
    patch[np.asarray(anchor["hotspot_token_indices"], np.int32)] = True
    if not patch.any() or np.any(patch & binder):
        raise ValueError("hotspot patch is empty or overlaps the antibody")
    context = {"cdr_mask": cdr, "cdr3_mask": cdr3, "patch_mask": patch}
    model_config = run_af3.make_model_config(
        num_diffusion_samples=1,
        num_recycles=0,
        return_distogram=True,
        flash_attention_implementation="xla",
    )
    model_class = conditioned_confidence_model_class()

    @hk.transform
    def forward_model(batch, pseudo, profile, dense, geometry_mask, key):
        return model_class(
            model_config,
            RematConfig(enabled=True),
            jnp.asarray(cdr),
            num_design_recycles=design_recycles,
            detach_recycles=True,
        )(batch, pseudo, profile, dense, geometry_mask, key)

    parameters = af_params.get_model_haiku_params(model_dir=Path(model_dir).resolve())

    def objective(batch, pseudo, profile, dense, geometry_mask, key):
        result = forward_model.apply(
            parameters, key, batch, pseudo, profile, dense, geometry_mask, key
        )
        return four_term_consistency_loss(result, batch, context)

    scorer = jax.jit(objective)
    key = jax.random.PRNGKey(scorer_seed)
    scratch = Path(output_dir).resolve() / "candidates"
    scratch.mkdir(parents=True, exist_ok=True)
    ranked = []
    started = time.monotonic()
    for index, candidate in enumerate(candidates):
        sequence = candidate["sequence"]
        candidate_ids = np.asarray([AA_INDEX[aa] for aa in sequence], np.int32)
        candidate_path = candidate_input(
            base_input,
            candidate_ids,
            scratch / f"candidate_{index:04d}.json",
            binder_chain=binder_chain,
        )
        _, batch = _raw_and_clean_batch(candidate_path, int(anchor["bucket"]))
        current_binder = binder_mask_from_batch(batch, binder_asym_id)
        if not np.array_equal(current_binder, binder):
            raise ValueError("candidate changed the AF3 token layout")
        if not np.array_equal(np.asarray(batch["aatype"])[current_binder], candidate_ids):
            raise ValueError("candidate atom chemistry is stale")
        pseudo = np.zeros((valid.size, WIDTH), np.float32)
        pseudo[design, :20] = np.eye(20, dtype=np.float32)[candidate_ids[design_local]]
        dense = scatter_pseudo_beta(feat_batch.Batch.from_data_dict(batch), pseudo_beta)
        total, auxiliary = scorer(
            batch,
            jnp.asarray(pseudo),
            jnp.asarray(batch["profile"], np.float32),
            jnp.asarray(dense),
            jnp.asarray(geometry_valid),
            key,
        )
        jax.block_until_ready(total)
        metrics = {name: float(np.asarray(value)) for name, value in auxiliary.items()}
        if not all(np.isfinite(value) for value in metrics.values()):
            raise ValueError("non-finite Consistency Gate output")
        eligible, checks = _threshold_eligibility(metrics, thresholds)
        ranked.append(
            {
                **candidate,
                "metrics": metrics,
                "eligible": eligible,
                "threshold_checks": checks,
            }
        )
    ranked.sort(key=lambda row: (row["metrics"]["consistency_loss"], row["id"]))
    eligible = [row for row in ranked if row["eligible"]]
    selected = eligible[:top_k]
    result_path = Path(output_dir).resolve() / "consistency_results.json"
    payload = {
        "schema_version": "af3h_consistency_v1",
        "status": "completed" if selected else "rejected",
        "anchor_manifest_sha256": sha256_file(anchor_path),
        "candidate_pool_sha256": sha256_file(pool_path),
        "scorer_seed": scorer_seed,
        "design_recycles": design_recycles,
        "thresholds": thresholds,
        "candidate_count": len(ranked),
        "eligible_count": len(eligible),
        "selected": selected,
        "ranked": ranked,
        "diffusion_calls": 0,
        "actual_model_forwards": len(ranked),
        "af3_parameter_identifier": parameter_identifier(parameters),
        "af3_runtime": runtime_source_fingerprint(),
        "runtime_seconds": time.monotonic() - started,
        "platform": _platform(),
    }
    _write_json(result_path, payload)
    return {
        "result_path": result_path,
        "candidate_count": len(ranked),
        "eligible_count": len(eligible),
    }


def _region_mean(values: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(values, np.float32)[np.asarray(mask, bool)]
    return float(np.mean(selected)) if selected.size else float("nan")


def _directed_mean(matrix: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> float:
    block = np.asarray(matrix, np.float32)[
        np.ix_(np.asarray(rows, bool), np.asarray(columns, bool))
    ]
    return float(np.mean(block)) if block.size else float("nan")


def evaluate_selected_candidates(
    *,
    anchor_manifest: str | Path,
    consistency_results: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    seeds: tuple[int, ...],
    diffusion_steps: int,
    num_recycles: int,
) -> dict[str, Any]:
    """Run independent full AF3 prediction for selected sequences and seeds."""

    import jax
    import run_alphafold as run_af3
    from alphafold3.model import post_processing

    from .hybrid import candidate_input
    from .provenance import result_parameter_identifier, runtime_source_fingerprint

    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("final evaluator seeds must be non-empty and unique")
    anchor_path = Path(anchor_manifest).resolve()
    anchor = _load_json(anchor_path)
    consistency_path = Path(consistency_results).resolve()
    selection = _load_json(consistency_path)
    selected = list(selection.get("selected", []))
    if not selected:
        raise ValueError("Consistency results contain no selected candidates")
    base_input = Path(anchor["candidate_input_path"])
    binder_chain = str(anchor["binder_chain_id"])
    binder_asym_id = int(anchor["binder_asym_id"])
    design = np.asarray(anchor["design_token_indices"], np.int32)
    hotspot = np.asarray(anchor["hotspot_token_indices"], np.int32)
    model_config = run_af3.make_model_config(
        num_diffusion_samples=1,
        num_recycles=num_recycles,
        return_distogram=True,
        flash_attention_implementation="xla",
    )
    model_config.heads.diffusion.eval.steps = diffusion_steps
    runner = run_af3.ModelRunner(
        config=model_config,
        device=jax.devices()[0],
        model_dir=Path(model_dir).resolve(),
    )
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    structure_paths: list[Path] = []
    parameter_ids: set[str] = set()
    started = time.monotonic()
    for candidate_index, candidate in enumerate(selected):
        sequence = str(candidate["sequence"])
        candidate_ids = np.asarray([AA_INDEX[aa] for aa in sequence], np.int32)
        for seed in seeds:
            item_dir = root / f"candidate_{candidate_index:03d}" / f"seed_{seed}"
            item_dir.mkdir(parents=True, exist_ok=True)
            candidate_path = candidate_input(
                base_input,
                candidate_ids,
                item_dir / "candidate_input.json",
                binder_chain=binder_chain,
            )
            raw_batch, batch = _raw_and_clean_batch(candidate_path, int(anchor["bucket"]))
            valid, binder, _ = _token_layout(batch, binder_asym_id=binder_asym_id)
            if not np.array_equal(np.asarray(batch["aatype"])[binder], candidate_ids):
                raise ValueError("full AF3 candidate hard-sequence contract failed")
            cdr = np.zeros(valid.size, bool)
            cdr[design] = True
            patch = np.zeros(valid.size, bool)
            patch[hotspot] = True
            target = valid & ~binder
            item_started = time.monotonic()
            result = runner.run_inference(batch, jax.random.PRNGKey(seed))
            parameter_ids.add(result_parameter_identifier(result))
            jax.block_until_ready(result["predicted_lddt"])
            extracted = list(
                runner.extract_inference_results(
                    batch=raw_batch,
                    result=result,
                    target_name=f"{candidate['id']}_seed{seed}",
                )
            )
            if len(extracted) != 1:
                raise ValueError(f"expected one full AF3 output, got {len(extracted)}")
            post_processing.write_output(extracted[0], item_dir, name="model")
            structures = sorted(item_dir.glob("model*.cif"))
            if len(structures) != 1:
                raise ValueError(f"expected one final CIF, got {len(structures)}")
            structure_paths.append(structures[0])
            predicted = np.asarray(result["predicted_lddt"], np.float32)
            if predicted.ndim == 3:
                predicted = predicted[0]
            atom_mask = np.asarray(batch["pred_dense_atom_mask"], np.float32)
            token_plddt = np.sum(predicted * atom_mask, axis=-1) / (
                np.sum(atom_mask, axis=-1) + 1e-8
            )
            pae = np.asarray(result["full_pae"], np.float32)
            if pae.ndim == 3:
                pae = pae[0]
            metadata = extracted[0].metadata
            rows.append(
                {
                    "candidate_id": candidate["id"],
                    "sequence": sequence,
                    "seed": seed,
                    "binder_plddt": _region_mean(token_plddt, binder),
                    "cdr_plddt": _region_mean(token_plddt, cdr),
                    "binder_target_pae_symmetric": 0.5
                    * (
                        _directed_mean(pae, binder, target)
                        + _directed_mean(pae, target, binder)
                    ),
                    "cdr_patch_pae_symmetric": 0.5
                    * (
                        _directed_mean(pae, cdr, patch)
                        + _directed_mean(pae, patch, cdr)
                    ),
                    "ptm": float(np.asarray(metadata["ptm"])),
                    "iptm": float(np.asarray(metadata["iptm"])),
                    "ranking_score": float(np.asarray(metadata["ranking_score"])),
                    "has_clash": bool(np.asarray(metadata["has_clash"])),
                    "structure_path": str(structures[0]),
                    "structure_sha256": sha256_file(structures[0]),
                    "runtime_seconds": time.monotonic() - item_started,
                }
            )
    result_path = root / "final_evaluation.json"
    payload = {
        "schema_version": "af3h_final_evaluation_v1",
        "status": "completed",
        "anchor_manifest_sha256": sha256_file(anchor_path),
        "consistency_results_sha256": sha256_file(consistency_path),
        "candidate_count": len(selected),
        "seeds": list(seeds),
        "full_af3_calls": len(rows),
        "diffusion_steps": diffusion_steps,
        "num_recycles": num_recycles,
        "af3_parameter_identifiers": sorted(parameter_ids),
        "af3_runtime": runtime_source_fingerprint(),
        "evaluations": rows,
        "runtime_seconds": time.monotonic() - started,
        "platform": _platform(),
    }
    _write_json(result_path, payload)
    return {
        "result_path": result_path,
        "structure_paths": structure_paths,
        "candidate_count": len(selected),
        "full_af3_calls": len(rows),
    }

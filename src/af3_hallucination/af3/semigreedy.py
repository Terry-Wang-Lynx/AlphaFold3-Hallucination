"""Gradient-free semigreedy (discrete hill-climbing) design loop (note 018).

Semantic reproduction of ColabDesign design_semigreedy (af/design.py:427-474) with
the _mutate math (af/design.py:393-425) DIRECTLY PORTED to numpy. semigreedy is a
gradient-free hill-climb: start from the current hard sequence (logits argmax),
each step propose `num_tries` single-point mutations at low-pLDDT design positions,
score each with one predict(seq_indices=..., backprop=False) call, keep the
best-of-tries and the global best. This is exactly why it side-steps the P2-F/P2-F2
confidence-gradient failure (013 section 8): there are NO gradients here at all.

Hard constraints (018 section 0), enforced by construction:
  - `ranking_score` is NEVER a loss term; scoring uses predict()["scoring_loss"]
    (scoring.scoring_loss / BindCraft weights) only.
  - `iptm` enters only the forward scoring_loss scalar, never a gradient term.
  - No confidence gradient: predict() is gradient-free; this loop never differentiates.
  - Ligand tokens are never designable: the loop only mutates the L-length
    design-position array (design_indices already excludes is_ligand, build_masks
    guard 012/015). Framework / non-design / ligand tokens are not in this array.
  - predict() is single-candidate and stateless; the loop / best-of-tries /
    save_best / trajectory live HERE, not in predict().

Pure numpy: NO jax, NO AF3, NO predict execution inside this module. The caller
passes a `predict_fn(seq_indices) -> dict` (runner.design_semigreedy wraps
self.predict); that dict must carry "scoring_loss" and per_token_plddt["design"].
"""

from __future__ import annotations

import numpy as np

# AF3 protein designable alphabet size (residue_names PROTEIN order, ids 0-19);
# semigreedy only proposes protein residues so predict()'s refeaturise / inject
# stay valid (note 015 / 018 section 1).
PROTEIN_ALPHABET_SIZE = 20


def _softmax(x: np.ndarray) -> np.ndarray:
    """ColabDesign shared/utils.softmax (numpy)."""
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _categorical(p: np.ndarray, rng: np.random.RandomState) -> int:
    """ColabDesign shared/utils.categorical (numpy): inverse-CDF sample."""
    return int((p.cumsum(-1) >= rng.uniform()).argmax(-1))


def _validate_seq(seq: np.ndarray, alphabet_size: int, name: str) -> np.ndarray:
    seq = np.asarray(seq, dtype=int)
    if seq.ndim != 1:
        raise ValueError(f"{name}: expected 1-D design aa ids, got shape {seq.shape}.")
    bad = seq[(seq < 0) | (seq >= alphabet_size)]
    if bad.size:
        raise ValueError(
            f"{name}: aa ids {bad[:5].tolist()} outside protein alphabet "
            f"[0, {alphabet_size}).")
    return seq


def mutate(seq, plddt, logits, *, alphabet_size: int = PROTEIN_ALPHABET_SIZE,
           mutation_rate: int = 1, rng: np.random.RandomState | None = None) -> np.ndarray:
    """ColabDesign _mutate (af/design.py:393-425), ported to the design array.

    seq: int[L] aa ids on the design positions (NOT the full token array; the
      L-length array is exactly ColabDesign's non-fixed-position set, so no fix_pos
      subtraction is needed). plddt: float[L] in [0,1] (low -> mutate more) or None
      -> uniform. logits: per-position aa logits (= seq_logits[design] + aa_bias);
      scalar/0, [alphabet] (same for all positions) or [L, alphabet]. Returns a NEW
      seq with `mutation_rate` single-point mutations; each mutated position gets a
      DIFFERENT aa (current aa is masked out by -1e8, ColabDesign :419).
    """
    rng = rng if rng is not None else np.random.RandomState(0)
    seq = _validate_seq(seq, alphabet_size, "mutate(seq)").copy()
    L = seq.shape[-1]
    i_prob = np.ones(L) if plddt is None else np.maximum(1.0 - np.asarray(plddt, float), 0.0)
    i_prob = np.where(np.isnan(i_prob), 0.0, i_prob)
    if i_prob.sum() <= 0:
        i_prob = np.ones(L)
    lg = np.array(0.0 if logits is None else logits, dtype=float)
    for _ in range(mutation_rate):
        i = int(rng.choice(np.arange(L), p=i_prob / i_prob.sum()))
        if lg.ndim == 2:
            row = lg[i]
        elif lg.ndim == 1:
            row = lg
        else:  # scalar
            row = np.zeros(alphabet_size)
        a_logits = row - np.eye(alphabet_size)[int(seq[i])] * 1e8
        seq[i] = _categorical(_softmax(a_logits), rng)
    return seq


def _traj_entry(step, aux, num_tries, best_loss, prior_plddt_design, accepted, seq):
    """One trajectory record (014 scoring fields + bookkeeping)."""
    return {
        "step": step, "num_tries": num_tries, "accepted": accepted,
        "scoring_loss": float(aux["scoring_loss"]),
        "best_scoring_loss": float(best_loss),
        "seq_design": [int(value) for value in np.asarray(seq)],
        "con": float(aux["con"]), "i_con": float(aux["i_con"]),
        "pae": float(aux["pae"]), "i_pae": float(aux["i_pae"]),
        "binder_plddt": float(aux["binder_plddt"]), "iptm": float(aux["iptm"]),
        "prior_plddt_design_mean": (None if prior_plddt_design is None
                                    else float(np.mean(prior_plddt_design))),
    }


def run_semigreedy(predict_fn, *, design_indices, seq_design_start,
                   iters: int = 15, tries: int = 1, e_tries: int | None = None,
                   seq_logits=None, aa_bias=None,
                   alphabet_size: int = PROTEIN_ALPHABET_SIZE,
                   mutation_rate: int = 1,
                   rng: np.random.RandomState | None = None) -> dict:
    """ColabDesign design_semigreedy loop (af/design.py:427-474), semantic reproduction.

    predict_fn(seq_indices) -> dict with "scoring_loss" + per_token_plddt["design"]
    (the runner wraps self.predict; the loop NEVER calls forward()/optimize()).
    Position prior uses the BEST candidate's per_token_plddt["design"]/100 (018
    section 2), not a scalar binder_plddt. Global best (save_best, min scoring_loss)
    is maintained here; the accepted seq each step is the round's best-of-tries.
    """
    rng = rng if rng is not None else np.random.RandomState(0)
    if e_tries is None:
        e_tries = tries
    L = int(np.asarray(design_indices).shape[0])
    seq = _validate_seq(seq_design_start, alphabet_size, "seq_design_start")
    if seq.shape[-1] != L:
        raise ValueError(
            f"run_semigreedy: seq_design_start length {seq.shape[-1]} != "
            f"len(design_indices) {L}.")

    bias = (np.zeros(alphabet_size) if aa_bias is None
            else np.asarray(aa_bias, float)[:alphabet_size])
    base = 0.0 if seq_logits is None else np.asarray(seq_logits, float)

    def eff_logits():
        if np.ndim(base) == 0:
            return bias                      # [alphabet], same for all positions
        if np.ndim(base) == 1:
            return base[:alphabet_size] + bias
        return base[:, :alphabet_size] + bias[None, :]   # [L, alphabet]

    # initial score + starting position prior
    p0 = predict_fn(seq)
    plddt = np.asarray(p0["per_token_plddt"]["design"], float) / 100.0
    best = {"seq": seq.copy(), "scoring_loss": float(p0["scoring_loss"]), "aux": p0}
    traj = [
        _traj_entry(
            -1, p0, 0, best["scoring_loss"], None, accepted=True, seq=seq
        )
    ]

    for i in range(iters):
        # ColabDesign casts the ramped try count with int(), i.e. floor toward
        # zero for the positive values used here (af/design.py:457-458).
        num_tries = int(tries + (e_tries - tries) * ((i + 1) / iters))
        num_tries = max(num_tries, 1)
        prior_plddt = plddt
        cands = []
        for _ in range(num_tries):
            cand = mutate(seq, prior_plddt, eff_logits(),
                          alphabet_size=alphabet_size, mutation_rate=mutation_rate, rng=rng)
            pc = predict_fn(cand)
            cands.append((cand, pc))
        # best-of-tries = argmin scoring_loss (ColabDesign :465-466)
        cand_b, pc_b = min(cands, key=lambda c: float(c[1]["scoring_loss"]))
        seq = np.array(cand_b, dtype=int)
        plddt = np.asarray(pc_b["per_token_plddt"]["design"], float) / 100.0
        # global best (save_best): only update when strictly better
        if float(pc_b["scoring_loss"]) < best["scoring_loss"]:
            best = {"seq": seq.copy(), "scoring_loss": float(pc_b["scoring_loss"]), "aux": pc_b}
        traj.append(
            _traj_entry(
                i,
                pc_b,
                num_tries,
                best["scoring_loss"],
                prior_plddt,
                accepted=True,
                seq=seq,
            )
        )

    return {
        "best_seq_design": [int(x) for x in best["seq"]],
        "best_scoring_loss": best["scoring_loss"],
        "best_aux": best["aux"],
        "trajectory": traj,
        "iters": iters, "tries": tries, "e_tries": e_tries,
        "design_indices": [int(x) for x in np.asarray(design_indices)],
    }

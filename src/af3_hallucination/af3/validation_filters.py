"""BindCraft final-predict confidence gate / filter (non-coordinate subset).

Semantic reproduction of BindCraft's confidence gate/filter (note 021) on the AF3
`predict()` output dict (notes 014/017/019). PURE FUNCTIONS: no AF3 runner, no
jax, no GPU. This is the validation/filter ORCHESTRATION; it never runs a model.

Three BindCraft layers (021 section 0); V0 implements the NON-COORDINATE subset:
  - stage gate (per gradient stage end): binder mean pLDDT > 0.65
    (colabdesign_utils.py:103/142/154). V0 DEFERS the per-stage gate (it needs a
    diffusion+confidence predict per stage); a helper is provided for parity/tests.
  - trajectory final gate: has_clash -> "Clashing"; binder pLDDT/100 < 0.7 ->
    "LowConfidence" (colabdesign_utils.py:197 -- note 0.7, NOT 0.65). interface
    contacts < 3 is COORDINATE -> deferred.
  - validation filter (default_filters.json): pLDDT >= 0.8, pTM >= 0.55,
    i_pTM >= 0.5, i_pAE <= 0.35 (NORMALIZED /31), pAE default None (skip).

UNIT TRAPS (021 P1):
  - binder_plddt is 0-100; divide by 100 before comparing the 0-1 thresholds.
  - i_pAE threshold 0.35 is the get_pae_loss /31 NORMALIZED value (af/loss.py:252),
    NOT raw Angstrom. AF3 predict's `i_pae` field is already that normalized loss;
    compare it to 0.35 directly. NEVER compare a raw interface PAE (Å) to 0.35.
  - pLDDT/pTM/iPTM are 0-1 actual (not losses).

ranking_score is REPORT/FILTER-ORDERING ONLY (AF3-native 0.8*iptm+0.2*ptm+...);
it NEVER enters a pass/fail filter or a design loss (014/017/019 consistent).

Coordinate / Rosetta / MPNN / Relax / secondary-structure / multi-model-average
items are NOT implemented and are returned explicitly under `deferred` so they are
never silently treated as pass (021 P2).

Thresholds come ONLY from external/bindcraft default_filters.json +
colabdesign_utils.py (021 section 1); no thresholds are invented here.
"""

from __future__ import annotations

import dataclasses

# Final trajectory gate thresholds (colabdesign_utils.py:197; 021 section 1b).
FINAL_GATE_PLDDT = 0.7        # binder_plddt/100 < 0.7 -> LowConfidence
# Per-stage gate threshold (colabdesign_utils.py:103/142/154; 021 1a). DEFERRED in
# V0 (needs a predict per stage); helper kept for parity/tests.
STAGE_GATE_PLDDT = 0.65       # binder_plddt/100 > 0.65 to continue the trajectory

# Coordinate / Rosetta / MPNN / structure items NOT implemented in V0 (021 P2/§3).
DEFERRED_ITEMS = (
    "interface_contacts_lt_3",   # colabdesign_utils.py:204-208 (pdb hotspot_residues)
    "n_InterfaceResidues",       # bindcraft.py score_interface (Rosetta/biopython)
    "dG", "ShapeComplementarity", "Surface_Hydrophobicity", "Binder_Energy",
    "dSASA", "Rosetta_Clashes", "ss_pLDDT",  # bindcraft.py score_interface
    "mpnn_redesign",             # mpnn_gen_sequence
    "multi_model_average",       # AF2 2-model Average_*; AF3 single weight/sample
    "ca_clashes_2.5A",           # BindCraft ca_clashes; AF3 uses native has_clash
)


@dataclasses.dataclass(frozen=True)
class FilterCondition:
    """One confidence filter condition (BindCraft filter_conditions row).

    metric: key into the predict() output dict.
    op: ">=" or "<=" (BindCraft `higher` True -> ">=").
    threshold: comparison value, or None to SKIP this filter (BindCraft default
      None -> not applied).
    scale: divide the metric by this before comparing (100 for binder_plddt 0-100;
      1 for already-normalized i_pae /31).
    note: source / unit annotation.
    """

    metric: str
    op: str
    threshold: float | None
    scale: float = 1.0
    note: str = ""


def bindcraft_default_confidence_filters() -> dict:
    """BindCraft default_filters.json confidence subset (021 section 1c / 2).

    Values cited from external/bindcraft/settings_filters/default_filters.json:
    1_pLDDT 0.8 (>=), 1_pTM 0.55 (>=), 1_i_pTM 0.5 (>=), 1_pAE None (skip),
    1_i_pAE 0.35 (<=, NORMALIZED /31). binder_plddt is scaled /100.
    """
    return {
        "binder_plddt": FilterCondition(
            "binder_plddt", ">=", 0.8, scale=100.0,
            note="default_filters 1_pLDDT=0.8; binder_plddt is 0-100 -> /100"),
        "ptm": FilterCondition(
            "ptm", ">=", 0.55, note="default_filters 1_pTM=0.55 (0-1)"),
        "iptm": FilterCondition(
            "iptm", ">=", 0.5, note="default_filters 1_i_pTM=0.5 (0-1)"),
        "pae": FilterCondition(
            "pae", "<=", None, note="default_filters 1_pAE=None (skipped); normalized /31"),
        "i_pae": FilterCondition(
            "i_pae", "<=", 0.35,
            note="default_filters 1_i_pAE=0.35; NORMALIZED /31 (NOT raw Angstrom)"),
    }


def _passes(value: float, op: str, threshold: float) -> bool:
    if op == ">=":
        return value >= threshold
    if op == "<=":
        return value <= threshold
    raise ValueError(f"apply_confidence_filters: unknown op {op!r} (expected '>=' or '<=').")


def apply_confidence_filters(predict_out: dict, filters: dict | None = None, *,
                             mode: str = "accumulate") -> dict:
    """Apply the confidence filter conditions to a predict() output dict.

    Reproduces BindCraft filter_conditions (colabdesign_utils.py:272-292): each
    condition has a metric key, op, threshold; None threshold -> skip; a metric
    value failing its op -> a failure. BindCraft BREAKS on the first fail; this
    defaults to `mode="accumulate"` (record ALL failures, more informative for a
    prototype) and supports `mode="break"` for byte-for-byte BindCraft semantics.

    `ranking_score` is NEVER read here (report/ordering only). Coordinate/Rosetta
    items are returned under `deferred`, never silently passed.

    Returns {pass, failures, metrics, skipped, deferred, mode}.
    """
    if mode not in ("accumulate", "break"):
        raise ValueError("apply_confidence_filters: mode must be 'accumulate' or 'break'.")
    filters = filters if filters is not None else bindcraft_default_confidence_filters()

    failures: dict = {}
    metrics: dict = {}
    skipped: list = []
    passed = True
    for name, cond in filters.items():
        if cond.threshold is None:
            skipped.append(name)
            continue
        if cond.metric not in predict_out or predict_out[cond.metric] is None:
            # conservative: a required metric that is missing FAILS (never silent pass).
            failures[name] = {"reason": "missing_metric", "metric": cond.metric,
                              "threshold": cond.threshold, "op": cond.op}
            passed = False
            if mode == "break":
                break
            continue
        value = float(predict_out[cond.metric]) / cond.scale
        metrics[name] = value
        if not _passes(value, cond.op, cond.threshold):
            failures[name] = {"reason": "threshold", "value": value,
                              "threshold": cond.threshold, "op": cond.op}
            passed = False
            if mode == "break":
                break
    return {
        "pass": passed,
        "failures": failures,
        "metrics": metrics,
        "skipped": skipped,
        "deferred": list(DEFERRED_ITEMS),
        "mode": mode,
    }


def final_trajectory_gate(predict_out: dict) -> dict:
    """BindCraft trajectory final gate, non-coordinate subset (021 1b / §4).

    has_clash == True            -> terminate "Clashing"  (AF3-native has_clash; an
                                    approximation of BindCraft ca_clashes 2.5A).
    binder_plddt/100 < 0.7       -> terminate "LowConfidence" (note 0.7, NOT 0.65).
    interface_contacts < 3       -> DEFERRED (coordinate; never silently passes).

    Returns {terminate, reason, deferred}. terminate is None when the gate passes.
    """
    if bool(predict_out.get("has_clash", False)):
        return {"terminate": True, "reason": "Clashing",
                "deferred": ["interface_contacts_lt_3"]}
    bp = predict_out.get("binder_plddt")
    if bp is not None and (float(bp) / 100.0) < FINAL_GATE_PLDDT:
        return {"terminate": True, "reason": "LowConfidence",
                "deferred": ["interface_contacts_lt_3"]}
    return {"terminate": False, "reason": None,
            "deferred": ["interface_contacts_lt_3"]}


def stage_plddt_gate(binder_plddt, threshold: float = STAGE_GATE_PLDDT) -> bool:
    """Per-stage gate (DEFERRED in V0): binder pLDDT/100 > 0.65 to continue.

    BindCraft applies this at each gradient-stage transition (colabdesign_utils.py:
    103/142/154) using free trunk pLDDT; AF3 needs a diffusion+confidence predict
    per stage, so V0 does NOT wire it into the loop. Helper kept for parity/tests.
    """
    return (float(binder_plddt) / 100.0) > threshold


def report_only_fields(predict_out: dict) -> dict:
    """Fields useful for CSV/reporting; ranking_score remains report-only.

    pTM/iPTM can also be used by `apply_confidence_filters`; this helper is only
    a reporting extractor and does not decide pass/fail.
    """
    return {k: predict_out.get(k) for k in
            ("ranking_score", "ptm", "iptm", "fraction_disordered", "has_clash")}

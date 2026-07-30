"""Gradient-free full-diffusion gates for AF3Design internal preview.

Internal preview v2 keeps the differentiable objective restricted to AF3 trunk
distogram/contact terms. Confidence quantities (pLDDT, iPTM, PAE, clash) are
queried only through explicit full-diffusion predict gates on a hardened
sequence.

This module is import-light: no JAX or AlphaFold imports at module load.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class GateConfig:
    """Configuration for one gradient-free AF3 confidence gate."""

    bucket: int = 256
    diffusion_steps: int = 200
    num_recycles: int = 10
    num_samples: int = 1
    policy: str = "refeaturise"
    binder_chain_id: str | None = None
    stage_plddt_min: float = 65.0
    force_pass: bool = False


@dataclasses.dataclass(frozen=True)
class GateResult:
    """Structured result of one gate evaluation."""

    name: str
    kind: str
    passed: bool
    unforced_pass: bool
    forced_pass: bool
    reason: str | None
    binder_plddt: float | None
    iptm: float | None
    i_pae: float | None
    ptm: float | None
    ranking_score: float | None
    has_clash: bool | None
    runtime_sec: float
    predict: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        # Full predict dicts can contain large raw arrays when callers request
        # return_raw. Gate logs keep scalar fields only.
        out["predict"] = {
            k: v
            for k, v in self.predict.items()
            if isinstance(v, (str, int, float, bool, type(None)))
        }
        return out


def hard_sequence_indices_from_logits(logits, bias=None) -> np.ndarray:
    """Return argmax(logits + bias) as int32 hard residue ids.

    The bias term is load-bearing: it enforces omitted amino acids and prevents
    non-protein AF3 token ids from being selected when the design width is 31.
    """

    arr = np.asarray(logits)
    if bias is not None:
        arr = arr + np.asarray(bias)
    return np.asarray(arr.argmax(axis=-1), dtype=np.int32)


def current_hard_sequence(model) -> np.ndarray:
    """Hard sequence for the current model state, respecting omit-AA bias."""

    if getattr(model.state, "logits", None) is None:
        raise RuntimeError("current_hard_sequence: model.state.logits is empty")
    return hard_sequence_indices_from_logits(
        model.state.logits, bias=getattr(model.state, "bias", None)
    )


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return None


def _predict_scalars(predict_out: dict[str, Any]) -> dict[str, Any]:
    return {
        "binder_plddt": _float_or_none(predict_out.get("binder_plddt")),
        "iptm": _float_or_none(predict_out.get("iptm")),
        "i_pae": _float_or_none(predict_out.get("i_pae")),
        "ptm": _float_or_none(predict_out.get("ptm")),
        "ranking_score": _float_or_none(predict_out.get("ranking_score")),
        "has_clash": (
            None if predict_out.get("has_clash") is None
            else bool(predict_out.get("has_clash"))
        ),
    }


def _stage_decision(metrics: dict[str, Any], cfg: GateConfig) -> tuple[bool, str | None]:
    """BindCraft-parity intermediate gate: pLDDT only.

    BindCraft's intermediate 4-stage checks use best-iterate pLDDT > 0.65.
    Clash/contact checks belong to the final trajectory filter, not the stage
    transition gate.
    """

    plddt = metrics["binder_plddt"]
    if plddt is None:
        return False, "missing_binder_plddt"
    if plddt <= cfg.stage_plddt_min:
        return False, "binder_plddt_below_stage_min"
    return True, None


def _final_decision(predict_out: dict[str, Any]) -> tuple[bool, str | None]:
    """BindCraft-parity final trajectory gate.

    This gate intentionally does NOT apply BindCraft default confidence filters
    (pLDDT 0.8 / pTM / iPTM / i_pAE). Those are validation/reporting filters,
    not the hallucination trajectory pass/fail criterion.
    """

    from . import validation_filters as vf

    trajectory_gate = vf.final_trajectory_gate(predict_out)
    if trajectory_gate["terminate"]:
        return False, trajectory_gate["reason"]
    return True, None


def run_full_diffusion_gate(
    model,
    *,
    name: str,
    kind: str = "stage",
    seed: int | None = None,
    config: GateConfig | None = None,
    seq_indices=None,
    force_pass: bool | None = None,
) -> GateResult:
    """Run one gradient-free full-diffusion/confidence gate.

    Args:
      model: prepared AF3DesignModel with params loaded.
      name: human-readable gate name, e.g. ``after_50_logits``.
      kind: ``stage`` uses the BindCraft-style intermediate pLDDT gate;
        ``final`` applies the final trajectory gate. BindCraft default
        confidence filters are intentionally separate reporting/validation.
      seed: predict seed. ``None`` uses the model default.
      config: gate runtime config.
      seq_indices: optional hard sequence; when omitted, computed from
        ``model.state.logits + model.state.bias``.
      force_pass: override only the returned ``passed`` field while preserving
        ``unforced_pass`` and ``reason`` for timing-only experiments.
    """

    cfg = config or GateConfig()
    if kind not in ("stage", "final"):
        raise ValueError("kind must be 'stage' or 'final'")
    if seq_indices is None:
        seq_indices = current_hard_sequence(model)
    force = cfg.force_pass if force_pass is None else bool(force_pass)

    t0 = time.time()
    predict_out = model.predict(
        seq_indices=np.asarray(seq_indices, dtype=np.int32),
        policy=cfg.policy,
        binder_chain_id=cfg.binder_chain_id,
        diffusion_steps=cfg.diffusion_steps,
        num_recycles=cfg.num_recycles,
        num_samples=cfg.num_samples,
        bucket=cfg.bucket,
        seed=seed,
    )
    runtime = time.time() - t0
    metrics = _predict_scalars(predict_out)
    if kind == "stage":
        unforced, reason = _stage_decision(metrics, cfg)
    else:
        unforced, reason = _final_decision(predict_out)
    passed = bool(unforced or force)
    return GateResult(
        name=name,
        kind=kind,
        passed=passed,
        unforced_pass=bool(unforced),
        forced_pass=bool(force and not unforced),
        reason=None if passed and unforced else reason,
        runtime_sec=round(runtime, 3),
        predict=predict_out,
        **metrics,
    )

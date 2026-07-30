"""Pure-Python Hallucination schedule expansion."""

from __future__ import annotations

from typing import Any

from .config import HallucinationSpec, StageSpec

SCHEDULE_PROVENANCE = {
    "bindcraft_commit": "b971db42ba6e091afab63ccb30ae02215150a990",
    "colabdesign_commit": "e31a56fe1d9b4de25c8697f3a28b75892941cc72",
    "ramp_semantics": "ColabDesign af/design.py design(): linear soft/hard/step, quadratic temperature",
}


def ramp(start: float, end: float, index: int, steps: int, *, quadratic: bool = False) -> float:
    if steps <= 0:
        return float(start)
    fraction = (index + 1) / steps
    if quadratic:
        return float(end + (start - end) * (1.0 - fraction) ** 2)
    return float(start + (end - start) * fraction)


def expand_stage(stage: StageSpec, *, stage_index: int, global_offset: int, default_lr: float) -> list[dict[str, Any]]:
    rows = []
    assert stage.soft_start is not None and stage.soft_end is not None
    assert stage.temperature_start is not None and stage.temperature_end is not None
    assert stage.hard_start is not None and stage.hard_end is not None
    step_end = stage.step_start if stage.step_end is None else stage.step_end
    for index in range(stage.steps):
        soft = ramp(stage.soft_start, stage.soft_end, index, stage.steps)
        temperature = ramp(
            stage.temperature_start,
            stage.temperature_end,
            index,
            stage.steps,
            quadratic=True,
        )
        hard = ramp(stage.hard_start, stage.hard_end, index, stage.steps)
        step_scale = ramp(stage.step_start, step_end, index, stage.steps)
        gradient_enabled = stage.type != "semigreedy"
        rows.append(
            {
                "stage": stage.name or stage.type,
                "stage_type": stage.type,
                "stage_index": stage_index,
                "stage_step": index,
                "global_step": global_offset + index,
                "soft": soft,
                "temperature": temperature,
                "hard": hard,
                "step_scale": step_scale,
                "learning_rate": stage.learning_rate or default_lr,
                "lr_scale": 0.0 if not gradient_enabled else step_scale * ((1.0 - soft) + soft * temperature),
                "gradient_enabled": gradient_enabled,
                "dropout": stage.dropout,
                "tries": stage.tries,
            }
        )
    return rows


def expand_schedule(spec: HallucinationSpec) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    offset = 0
    for stage_index, stage in enumerate(spec.stages):
        rows = expand_stage(
            stage,
            stage_index=stage_index,
            global_offset=offset,
            default_lr=spec.learning_rate,
        )
        plan.extend(rows)
        offset += stage.steps
    return plan


def bindcraft_four_stage() -> tuple[StageSpec, ...]:
    """Return the audited BindCraft 75/45/5/15 schedule as explicit stages."""

    return (
        StageSpec("logits", 50, soft_start=0.0, soft_end=0.9, temperature_start=1.0, temperature_end=1.0, hard_start=0.0, hard_end=0.0, name="g1_logits"),
        StageSpec("logits", 25, soft_start=0.0, soft_end=1.0, temperature_start=1.0, temperature_end=1.0, hard_start=0.0, hard_end=0.0, name="additional_logits"),
        StageSpec("soft", 45, soft_start=1.0, soft_end=1.0, temperature_start=1.0, temperature_end=0.01, hard_start=0.0, hard_end=0.0, name="g2_soft"),
        StageSpec("hard", 5, soft_start=1.0, soft_end=1.0, temperature_start=0.01, temperature_end=0.01, hard_start=1.0, hard_end=1.0, name="g3_hard"),
        StageSpec("semigreedy", 15, soft_start=1.0, soft_end=1.0, temperature_start=0.01, temperature_end=0.01, hard_start=1.0, hard_end=1.0, name="g4_semigreedy"),
    )

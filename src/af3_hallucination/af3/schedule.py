"""BindCraft/ColabDesign-style design schedule planner (internal preview, no jax).

Pure-python schedule planner so the no-GPU schema/unit tests and the CLI dry-run
can build/inspect the schedule without importing jax/alphafold3. Only depends on
`af3design.config` (import-safe, no jax). The GPU loop in tools/run_af3_design.py
consumes the same plan to drive the differentiable stages.

Source of truth (semantic reproduction, cited per stage):
  external/bindcraft/functions/colabdesign_utils.py:94-160   (4stage flow + clear_best/save_best)
  external/bindcraft/settings_advanced/default_4stage_multimer.json
      soft_iterations=75 / temporary_iterations=45 / hard_iterations=5 / greedy_iterations=15
  external/colabdesign/colabdesign/af/design.py:313-362
      design() ramp: soft/hard/step linear, temp quadratic-decay (set_opt loop);
      lr_scale = step * ((1 - soft) + soft * temp); design_logits/soft/hard wrappers
  external/colabdesign/colabdesign/shared/model.py
      norm_seq_grad, soft_seq (applied in the GPU loop, not here)

Strict scope: this builds the schedule SHAPE only. It does not run AF3 and does
not decide yield/effect. Internal preview v2 exposes only the no-confidence
distogram/contact objective; confidence is evaluated only by explicit gates.
"""
from . import config

SCHEDULE_SOURCE = (
    "bindcraft/settings_advanced/default_4stage_multimer.json(75/45/5/15) + "
    "bindcraft/functions/colabdesign_utils.py:94-160 + "
    "colabdesign/af/design.py:313-362(design ramp, lr_scale)"
)

# BindCraft default counts (default_4stage_multimer.json). logits maps to the
# BindCraft soft_iterations total (design_logits iters=50 then soft_iterations-50).
FULL_COUNTS = {"logits": 75, "soft": 45, "hard": 5, "semigreedy": 15}
# Reduced defaults: small enough to smoke a real stage of every kind.
REDUCED_COUNTS = {"logits": 1, "soft": 1, "hard": 1, "semigreedy": 1}

GRADIENT_STAGE_NAMES = ("logits", "soft", "hard")
STAGE_ORDER = ("logits", "soft", "hard", "semigreedy")
SUPPORTED_ARMS = ("trunk_only",)


def stage_endpoints(name):
    """Ramp endpoints for a stage (ColabDesign design_logits/soft/hard + BindCraft).

    Mirrors af3design.config.BINDCRAFT_4STAGE / colabdesign af/design.py wrappers:
      logits: design_logits, soft ramps 0 -> 1 (BindCraft 1b e_soft=1), temp=1, hard=0
      soft  : design_soft(e_temp=1e-2), soft=1, temp ramps 1 -> 1e-2 (quadratic)
      hard  : design_hard(temp=1e-2), soft=1, hard=1, temp=1e-2, dropout off
      semigreedy: gradient-free; endpoints kept for display only
    """
    if name == "logits":
        return dict(soft=0.0, e_soft=1.0, temp=1.0, e_temp=None, hard=0.0, e_hard=None, dropout=False)
    if name == "soft":
        return dict(soft=1.0, e_soft=None, temp=1.0, e_temp=1e-2, hard=0.0, e_hard=None, dropout=False)
    if name == "hard":
        return dict(soft=1.0, e_soft=None, temp=1e-2, e_temp=None, hard=1.0, e_hard=None, dropout=False)
    if name == "semigreedy":
        return dict(soft=1.0, e_soft=None, temp=1e-2, e_temp=None, hard=1.0, e_hard=None, dropout=False)
    if name == "reduced":  # Round2 single-stage soft representation (SeqOpt soft=1/temp=1/hard=0)
        return dict(soft=1.0, e_soft=None, temp=1.0, e_temp=None, hard=0.0, e_hard=None, dropout=False)
    raise ValueError(f"unknown stage name: {name}")


def resolve_counts(overrides=None, full=False):
    """Per-stage iteration counts. overrides: dict subset of stage->int (CLI flags).

    full=True returns BindCraft full defaults (recorded / dry-run only this round).
    """
    base = dict(FULL_COUNTS if full else REDUCED_COUNTS)
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                base[k] = int(v)
    return base


def build_stage_plan(schedule_mode, arm, counts=None, num_updates=3):
    """Return the ordered list of stage dicts for a schedule_mode.

    Fields per stage (task contract):
      name, stage_index, iterations, soft, temp, hard, e_soft, e_temp, dropout,
      save_best, clear_best_before_stage, gradient_enabled, semigreedy_enabled,
      confidence_gradient_enabled
    `confidence_gradient_enabled` is always False in internal preview v2. AF3
    confidence is used only by explicit gradient-free gates, not by the design
    gradient.
    """
    if arm not in SUPPORTED_ARMS:
        raise ValueError(
            f"unsupported arm {arm!r}; this backend supports only {SUPPORTED_ARMS} "
            "(confidence-gradient arms are intentionally disabled)"
        )
    stages = []

    if schedule_mode == "reduced":
        # Round2 parity: one soft-like gradient stage of num_updates steps, no ramp.
        ep = stage_endpoints("reduced")
        stages.append(dict(
            name="reduced", stage_index=0, iterations=int(max(1, num_updates)),
            soft=ep["soft"], e_soft=ep["e_soft"], temp=ep["temp"], e_temp=ep["e_temp"],
            hard=ep["hard"], e_hard=ep["e_hard"], dropout=ep["dropout"],
            save_best=True, clear_best_before_stage=False,
            gradient_enabled=True, semigreedy_enabled=False,
            confidence_gradient_enabled=False))
        return stages

    if schedule_mode != "bindcraft_reduced":
        raise ValueError(f"unknown schedule_mode: {schedule_mode}")

    counts = counts or dict(REDUCED_COUNTS)
    for idx, nm in enumerate(STAGE_ORDER):
        ep = stage_endpoints(nm)
        iters = int(counts.get(nm, 0))
        is_grad = nm in GRADIENT_STAGE_NAMES
        # BindCraft clear_best() before every stage after the first gradient stage.
        clear_before = (idx > 0)
        stages.append(dict(
            name=nm, stage_index=idx, iterations=iters,
            soft=ep["soft"], e_soft=ep["e_soft"], temp=ep["temp"], e_temp=ep["e_temp"],
            hard=ep["hard"], e_hard=ep["e_hard"], dropout=ep["dropout"],
            save_best=True, clear_best_before_stage=clear_before,
            gradient_enabled=is_grad, semigreedy_enabled=(nm == "semigreedy"),
            confidence_gradient_enabled=False))
    return stages


def expand_plan(stages, learning_rate=0.1):
    """Expand stages to a flat per-step plan with ColabDesign lr_scale.

    Each step row (task trajectory contract, GPU loop fills loss/grad/confidence):
      stage, stage_index, stage_step, global_step, soft, temp, hard, lr_scale,
      gradient_enabled, semigreedy_enabled, save_best_enabled,
      confidence_gradient_enabled, semigreedy_candidate_count
    lr_scale = step(=1.0) * ((1 - soft) + soft * temp), with soft/hard linear and
    temp quadratic-decay ramps (config.ramp_value), per af/design.py:336-346.
    """
    plan = []
    gstep = 0
    for st in stages:
        iters = int(st["iterations"])
        for i in range(iters):
            if st["gradient_enabled"]:
                soft_i = config.ramp_value(st["soft"], st["soft"] if st["e_soft"] is None else st["e_soft"], i, iters)
                temp_i = config.ramp_value(st["temp"], st["temp"] if st["e_temp"] is None else st["e_temp"], i, iters, quadratic=True)
                hard_i = config.ramp_value(st["hard"], st["hard"] if st["e_hard"] is None else st["e_hard"], i, iters)
                lr_scale = 1.0 * ((1.0 - soft_i) + soft_i * temp_i)
                conf_step = bool(st["confidence_gradient_enabled"] and i == iters - 1)
                plan.append(dict(
                    stage=st["name"], stage_index=st["stage_index"], stage_step=i,
                    global_step=gstep, soft=soft_i, temp=temp_i, hard=hard_i,
                    lr_scale=lr_scale, learning_rate=learning_rate,
                    gradient_enabled=True, semigreedy_enabled=False,
                    save_best_enabled=bool(st["save_best"]),
                    confidence_gradient_enabled=conf_step,
                    semigreedy_candidate_count=0))
            else:  # semigreedy: gradient-free, wired but not GPU-smoked this round
                plan.append(dict(
                    stage=st["name"], stage_index=st["stage_index"], stage_step=i,
                    global_step=gstep, soft=st["soft"], temp=st["temp"], hard=st["hard"],
                    lr_scale=0.0, learning_rate=learning_rate,
                    gradient_enabled=False, semigreedy_enabled=True,
                    save_best_enabled=bool(st["save_best"]),
                    confidence_gradient_enabled=False,
                    semigreedy_candidate_count=0))
            gstep += 1
    return plan

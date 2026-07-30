"""Configuration dataclasses for the AF3 binder design model.

These are stable, model-agnostic config objects. They intentionally mirror the
ColabDesign/BindCraft option names (`cutoff`, `num`, `seqsep`, `binary`, the
4-stage iteration counts, the loss weights) so the BindCraft schedule can be
ported with minimal renaming. See writing/research-notes/006 sections 3.5 / 4.3
for the source-of-truth mapping.

No AF3 / JAX imports here on purpose: config must be importable anywhere.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class ContactSpec:
    """Semantics of one contact objective (con or i_con).

    Faithful to ColabDesign `get_con_loss` (af/loss.py:261). NOT a block mean.

    - cutoff: contact distance threshold in angstrom. ColabDesign bins with lower
      edge < cutoff count as "in contact".
    - num: per row, average the `num` closest partners (top-k over the masked
      axis via min_k). 1 = single best contact, BindCraft intra uses 2.
    - num_pos: number of rows (residues) to average over after the per-row
      reduction. None -> all rows.
    - binary: True uses the binary contact cross-entropy
      (-log sum_bins<cutoff p); False uses the categorical cross-entropy form.
    - seqsep: minimum |i-j| sequence separation for a pair to count. Excludes
      trivial local contacts (BindCraft intra default 9, helix uses offset==3).
    """

    cutoff: float = 8.0
    num: int = 1
    num_pos: int | None = None
    binary: bool = True
    seqsep: int = 0


@dataclasses.dataclass(frozen=True)
class LossWeights:
    """Weights for the differentiable V0 objective.

    Only trunk/distogram-derived terms are differentiable in V0. plddt / pae /
    iptm / rg / NC are validation/gating terms (note 006 section 3.5) and are
    deliberately absent here; they belong to the (later) validation path.
    """

    # BindCraft default_4stage_multimer parity for the DIFFERENTIABLE trunk
    # skeleton (weights_con_intra=1.0 / weights_con_inter=1.0 / weights_helicity=
    # -0.3). These are the only trunk-distogram (gradient) terms; plddt/pae/i_pae/
    # iptm/rg/NC are confidence/coordinate terms handled elsewhere (forward
    # scoring/validation, or experimental detached loss) -- see af3design.full_loss.
    con: float = 1.0       # intra-binder contact (BindCraft weights_con_intra)
    i_con: float = 1.0      # interface contact (BindCraft weights_con_inter)
    helix: float = -0.3     # helix bias (BindCraft weights_helicity; negative = encourage)
    seq_ent: float = 0.0    # sequence entropy regulariser
    # ESMFold2-style coordinate-free distogram Rg/globularity loss:
    # ELU(sqrt(sum_{i>j} E[d_ij^2] / N^2) - 2.38*N^0.365), adapted to AF3's
    # 64-bin distogram with the final open bin capped at 27 A. DEFAULT 0.0 =
    # OFF: it must not silently change the trunk_only / BindCraft-parity
    # objective. ESMFold2 reference lambda_glob=0.2 is opt-in only. Reads the
    # binder x binder distogram (NOT diffusion coordinates).
    glob: float = 0.0       # ESMFold2-style distogram Rg/globularity; OFF by default
    # Germinal-style paratope localization (note 191). When framework_contact_loss is True,
    # the i_con (hotspot x design-core) term is modulated by dividing by
    # max(off_design_i_con - framework_contact_offset, 1e-8), where off_design_i_con
    # is the hotspot x off_design_binder contact loss. This penalizes framework/non-design
    # residues from being pulled into the target interface. Default False/1.0 = OFF,
    # backward-compatible with all existing objective arms.
    framework_contact_loss: bool = False
    framework_contact_offset: float = 1.0


@dataclasses.dataclass(frozen=True)
class RematConfig:
    """Local block rematerialisation for the Evoformer/Pairformer trunk.

    AF3 ships an unwired `Evoformer.Config.block_remat` field; the trunk stacks
    its layers with hk.experimental.layer_stack and never remats. We wire remat
    *locally* (see remat.py) instead of globally monkeypatching layer_stack.

    - enabled: wrap each stacked block fn in hk.remat.
    - block_size: optional grouping; None -> per-block remat (matches the
      experiment/runs/2026-06-26_af3-v0-diff-loop result: bucket 256 = 3.64 GB).
    """

    enabled: bool = False
    block_size: int | None = None


@dataclasses.dataclass(frozen=True)
class StageConfig:
    """One BindCraft-style design stage.

    Stage names follow ColabDesign design_logits/soft/hard + BindCraft 4-stage
    (note 006 section 4.3). V0 implements only `soft`; the rest are declared so
    the schedule shape is expressible and stageable later.
    """

    name: str                  # "logits" | "soft" | "hard" | "semigreedy"
    iterations: int
    # None means use DesignConfig.learning_rate, matching ColabDesign's
    # model-level opt["learning_rate"]; set a value only for explicit overrides.
    learning_rate: float | None = None
    # ramp endpoints, ColabDesign af/design.py:design(). soft/temp/hard/step are
    # the START values; e_* are the END values (None -> constant = start). The
    # per-step ramp is linear for soft/hard/step and quadratic-decay for temp.
    soft: float = 1.0          # softmax mixing (0 = raw logits, 1 = full soft)
    e_soft: float | None = None
    temp: float = 1.0          # softmax temperature
    e_temp: float | None = None
    hard: float = 0.0          # straight-through one-hot mixing (1 = hard)
    e_hard: float | None = None
    step: float = 1.0          # lr_scale multiplier (ColabDesign design() `step`)
    e_step: float | None = None
    dropout: bool = False
    num_recycles: int = 0


@dataclasses.dataclass(frozen=True)
class DesignConfig:
    """Top-level config for an AF3DesignModel run."""

    # BindCraft default_4stage_multimer parity (semantic_reproduction). Values
    # taken directly from external/bindcraft/settings_advanced/
    # default_4stage_multimer.json + external/bindcraft/functions/
    # colabdesign_utils.py:48-49, which set con/i_con with binary=False:
    #   con   -> {num: intra_contact_number=2, cutoff: intra_contact_distance=14,
    #             binary: False, seqsep: 9}
    #   i_con -> {num: inter_contact_number=2, cutoff: inter_contact_distance=20,
    #             binary: False}  (no seqsep key -> ones; seqsep=0 reproduces it).
    # ColabDesign raw defaults (af/model.py:50-51) match except i_con num=1 /
    # cutoff=21.6875, which BindCraft overrides to the values used here.
    con_spec: ContactSpec = dataclasses.field(
        default_factory=lambda: ContactSpec(cutoff=14.0, num=2, seqsep=9, binary=False)
    )
    i_con_spec: ContactSpec = dataclasses.field(
        default_factory=lambda: ContactSpec(cutoff=20.0, num=2, seqsep=0, binary=False)
    )
    helix_cutoff: float = 6.0
    weights: LossWeights = dataclasses.field(default_factory=LossWeights)
    remat: RematConfig = dataclasses.field(default_factory=RematConfig)
    # Conservative design default: bucket 64 is the proven no-remat single-4090
    # backprop setting. Larger buckets should be requested explicitly with remat.
    bucket: int = 64           # token bucket; pair activations ~ bucket^2
    seed: int = 0
    # ColabDesign optimizer parity (af/model.py:38,47-49). Default optimizer is
    # SGD with model-level learning_rate=0.1 applied manually as
    # `params -= (learning_rate * lr_scale) * grad`; norm_seq_grad rescales the
    # design-logit gradient each step; alpha=2.0 is the soft_seq logit scale.
    # "adam" is allowed but is NOT BindCraft parity (kept only as an option).
    optimizer: str = "sgd"
    learning_rate: float = 0.1
    norm_seq_grad: bool = True
    alpha: float = 2.0
    # BindCraft/ColabDesign update_seq writes seq["pseudo"] into the MSA
    # one-hot/query channel and seq_pssm into the profile channel. AF3 stores
    # msa rows as int ids, so this flag enables the AF3 adapter that intercepts
    # create_msa_feat() and replaces the query-row one-hot with soft pseudo.
    soft_msa_query: bool = True
    # Differentiable trunk/distogram recycle count for design optimization.
    # This is separate from official predict/validation recycles below. It
    # follows ColabDesign's default recycle_mode="last": run num_recycles
    # non-gradient recycle updates to prev, then backprop through the final pass.
    # 0 preserves the original prototype behavior (one trunk pass).
    design_num_recycles: int = 0
    # predict() prototype (gradient-free OFFICIAL ModelRunner path; note 017).
    # Independent of the design-loop bucket: diffusion+confidence need bucket>=256
    # + xla (P2-A GLU/Triton blocker at bucket 64). Full-diffusion gates should use
    # "refeaturise" so AF3 atom layout matches the hardened sequence; "token_only"
    # remains available as a cheap diagnostic policy. These never touch the
    # differentiable trunk path (forward()/optimize()).
    predict_policy: str = "refeaturise"         # "refeaturise" | "token_only"
    predict_bucket: int = 256
    predict_diffusion_steps: int = 50
    predict_num_samples: int = 1
    predict_num_recycles: int = 0               # cheap semigreedy scorer (accepted default)
    predict_flash_attention: str = "xla"
    # Final-validation predict config (BindCraft design vs validation split, note
    # recycle/dropout/model-sampling spec): the cheap scorer above only ranks
    # candidates (steps50/recycle0), but the final confidence FILTER thresholds must
    # be applied to a higher-fidelity predict (BindCraft validation = recycle 3).
    # validate_* must NOT change runner.predict()'s accepted scorer defaults; they
    # are used only by runner.validate(). bucket/policy reuse the predict_* values.
    validate_diffusion_steps: int = 100
    validate_num_recycles: int = 3              # BindCraft validation parity
    validate_num_samples: int = 1


def ramp_value(start: float, end: float, i: int, iters: int,
               quadratic: bool = False) -> float:
    """Per-step ramp from ColabDesign af/design.py:design() (i is 0-indexed).

    - linear (soft/hard/step): `start + (end-start) * (i+1)/iters`
    - quadratic decay (temp):  `end + (start-end) * (1 - (i+1)/iters)**2`

    At i = iters-1 both reach `end`; at i = 0 they sit one step in from `start`.
    """
    if iters <= 0:
        return start
    frac = (i + 1) / iters
    if quadratic:
        return end + (start - end) * (1.0 - frac) ** 2
    return start + (end - start) * frac


# BindCraft default 4-stage schedule (semantic reproduction of
# external/bindcraft/functions/colabdesign_utils.py:94-160 +
# default_4stage_multimer.json: soft=75 / temporary=45 / hard=5 / greedy=15).
# Each StageConfig carries the ColabDesign design()/design_logits/soft/hard ramp
# endpoints. V0 actually executes only the differentiable logits/soft/hard
# stages; semigreedy stays a NotImplementedError placeholder but is kept here so
# the full schedule shape is explicit.
BINDCRAFT_4STAGE: tuple[StageConfig, ...] = (
    # Stage 1  - design_logits(iters=50, e_soft=0.9): soft ramps 0 -> 0.9
    StageConfig(name="logits", iterations=50, soft=0.0, e_soft=0.9, temp=1.0, hard=0.0),
    # Stage 1b - design_logits(iters=soft_iterations-50=25, e_soft=1.0): soft 0 -> 1
    StageConfig(name="logits", iterations=25, soft=0.0, e_soft=1.0, temp=1.0, hard=0.0),
    # Stage 2  - design_soft(45, e_temp=1e-2): soft=1, temp ramps 1 -> 1e-2 (quad)
    StageConfig(name="soft", iterations=45, soft=1.0, temp=1.0, e_temp=1e-2, hard=0.0),
    # Stage 3  - design_hard(5, temp=1e-2): soft=1, hard=1, temp=1e-2, dropout off
    StageConfig(name="hard", iterations=5, soft=1.0, temp=1e-2, hard=1.0, dropout=False),
    # Stage 4  - design_pssm_semigreedy(hard_iters=15): gradient-free (placeholder)
    StageConfig(name="semigreedy", iterations=15),
)

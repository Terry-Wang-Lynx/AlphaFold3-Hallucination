"""BindCraft full-loss weight + differentiability classification table.

A single source of truth that pins each BindCraft default_4stage_multimer.json
loss term to (weight, golden source, AF3 status). PURE PYTHON: no jax, no AF3.
This does NOT implement any loss; it documents/contracts what is differentiable
today vs forward-only vs deferred, so a full-loss prototype cannot silently
mis-classify a term (note 025 gap inventory).

Classification (`kind`):
  - "trunk_grad"  : differentiable design loss from the trunk distogram
                    (losses.total_loss). con / i_con / helix.
  - "forward"     : confidence forward scoring / validation (scoring.py), NOT a
                    differentiable design loss by default (P2-F/P2-F2 rejected the
                    single-sample detached-confidence gradient). plddt/pae/i_pae/iptm.
  - "experimental": may enter an EXPERIMENTAL detached-confidence loss (coords
                    stop_gradient, grad via trunk/confidence embeddings) -- never
                    the optimize() default. plddt/pae/i_pae here overlap "forward".
  - "coordinate"  : needs differentiable coordinates; AF3 coords come from
                    diffusion and are stop_gradient'd -> NO gradient. rg / NC.
  - "deferred"    : not implemented in the current prototype default path.
                    exp_res / i_ptm_grad.

Weights are quoted verbatim from external/bindcraft/settings_advanced/
default_4stage_multimer.json (no invented thresholds).
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class LossTerm:
    name: str
    weight: float
    kind: str          # trunk_grad | forward | experimental | coordinate | deferred
    source: str        # golden-source citation
    differentiable_now: bool


# BindCraft default_4stage_multimer.json (weights_*). The con/i_con/helix subset
# is the differentiable trunk skeleton (LossWeights defaults); the rest is
# forward/experimental/coordinate/deferred per note 025.
BINDCRAFT_FULL_LOSS: tuple[LossTerm, ...] = (
    LossTerm("con", 1.0, "trunk_grad",
             "default_4stage weights_con_intra; af/loss.py get_con_loss", True),
    LossTerm("i_con", 1.0, "trunk_grad",
             "default_4stage weights_con_inter; af/loss.py _loss_binder", True),
    LossTerm("helix", -0.3, "trunk_grad",
             "default_4stage weights_helicity; add_helix_loss/get_helix_loss", True),
    LossTerm("plddt", 0.1, "forward",
             "default_4stage weights_plddt; 1-get_plddt (af/model.py:202)", False),
    LossTerm("pae", 0.4, "forward",
             "default_4stage weights_pae_intra; get_pae_loss /31 (af/loss.py:252)", False),
    LossTerm("i_pae", 0.1, "forward",
             "default_4stage weights_pae_inter; get_pae_loss interface /31", False),
    LossTerm("iptm", 0.05, "forward",
             "default_4stage weights_iptm; metadata iptm (numpy, non-diff)", False),
    LossTerm("rg", 0.3, "coordinate",
             "default_4stage weights_rg; add_rg_loss CA structure_module", False),
    LossTerm("NC", 0.1, "coordinate",
             "default_4stage weights_termini_loss; add_termini_distance_loss CA", False),
    LossTerm("exp_res", 0.0, "deferred",
             "ColabDesign default weight 0 (af/model.py:53); BindCraft not enabled", False),
    LossTerm("i_ptm_grad", 0.05, "deferred",
             "differentiable i_ptm needs pae_logits; Route B source patch/thin head copy, Route A AF3-native approx (030)", False),
)

# Convenience views.
BINDCRAFT_FULL_LOSS_WEIGHTS: dict = {t.name: t.weight for t in BINDCRAFT_FULL_LOSS}
TRUNK_GRAD_TERMS: tuple = tuple(t.name for t in BINDCRAFT_FULL_LOSS if t.kind == "trunk_grad")
FORWARD_TERMS: tuple = tuple(t.name for t in BINDCRAFT_FULL_LOSS if t.kind == "forward")
COORDINATE_TERMS: tuple = tuple(t.name for t in BINDCRAFT_FULL_LOSS if t.kind == "coordinate")
DEFERRED_TERMS: tuple = tuple(t.name for t in BINDCRAFT_FULL_LOSS if t.kind == "deferred")

# Experimental detached-confidence loss (BoltzDesign-style): con/i_con/helix
# (trunk grad) + plddt/pae/i_pae (detached confidence, coords stop_gradient).
# NEVER the optimize() default; iptm/rg/NC/exp_res are NOT in it.
EXPERIMENTAL_DETACHED_TERMS: tuple = ("con", "i_con", "helix", "plddt", "pae", "i_pae")

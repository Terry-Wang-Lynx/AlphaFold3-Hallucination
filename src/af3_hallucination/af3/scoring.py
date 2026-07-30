"""BindCraft semigreedy candidate-scoring utilities (PROTOTYPE, no AF3 runner).

Pure-array helpers that turn AF3 predict-time confidence outputs into the
BindCraft/ColabDesign scalar design loss used to score semigreedy candidates
(note 014). These are FORWARD scoring utilities only: semigreedy is
gradient-free (013), so iptm can be used here as a plain scalar (it does not
need a differentiable head, unlike the rejected gradient mixed loss of 011 /
P2-F / P2-F2). NOTHING here runs AF3, diffusion or confidence; it consumes
arrays a caller obtained from a predict() forward.

Golden-source semantics (do NOT invent scoring heuristics, note 014 sections 4-5):
  - pLDDT is per-atom [.., N, A]; collapse to per-token with the ATOM MASK
    (never a raw dense-atom mean -- Codex P2-A caveat / note 014 R2).
  - pae       = symmetrized full_pae, binder rows x ALL valid cols, /31
                (ColabDesign get_pae_loss, af/loss.py:252; BindCraft `pae`).
  - i_pae     = symmetrized full_pae, binder rows x TARGET cols, /31
                (BindCraft `i_pae`, same mask as the design loss).
  - plddt_loss = 1 - binder_plddt/100      (af/loss.py:248).
  - iptm_loss  = 1 - iptm                   (add_i_ptm_loss, colabdesign_utils.py:393).
  - weights    = BindCraft default_4stage_multimer.json: con1 / i_con1 / pae0.4 /
                 i_pae0.1 / plddt0.1 / iptm0.05 / helix-0.3 (helix optional).
  - ALL confidence masks use `binder_mask` (the loss domain), NEVER `design_mask`
    (the mutation domain). For CDR-only design the binder is the whole antibody
    chain; for a small-molecule target the target is the is_ligand mask and the
    ligand never contributes to binder pLDDT (note 014 section 5).

`ranking_score` is a ranking AID only and must NOT be used as the design loss.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp

_EPS = 1e-8


def _reduce_samples(x: jnp.ndarray, sample_ndim: int) -> jnp.ndarray:
    """Average a leading diffusion-sample axis if present.

    predicted_lddt is [N, A] or [S, N, A]; full_pae is [N, N] or [S, N, N].
    `sample_ndim` is the rank WITHOUT a sample axis (2 for both here).
    """
    x = jnp.asarray(x)
    if x.ndim == sample_ndim + 1:
        return jnp.mean(x, axis=0)
    return x


def token_plddt(predicted_lddt: jnp.ndarray, atom_mask: jnp.ndarray) -> jnp.ndarray:
    """Collapse per-atom pLDDT to per-token using the atom mask.

    Args:
      predicted_lddt: [N, A] or [S, N, A] per-atom pLDDT (0-100).
      atom_mask: [N, A] real-atom mask (1 = real dense atom, 0 = padding).

    Returns [N] per-token pLDDT = atom-masked mean over real atoms. This is the
    mandated collapse; a raw `predicted_lddt.mean(-1)` would average padding
    atoms and is explicitly disallowed.
    """
    pl = _reduce_samples(predicted_lddt, 2)          # [N, A]
    am = jnp.asarray(atom_mask).astype(pl.dtype)     # [N, A]
    return jnp.sum(pl * am, axis=-1) / (jnp.sum(am, axis=-1) + _EPS)


def plddt_metrics(predicted_lddt, atom_mask, binder_mask, valid_mask) -> dict:
    """mean_plddt over valid tokens and binder_plddt over binder tokens.

    Both use the atom-masked per-token pLDDT. binder_mask is the loss domain
    (whole binder chain), NOT design_mask; ligand tokens are excluded from the
    binder by construction (note 012 / 014 section 5).
    """
    tok = token_plddt(predicted_lddt, atom_mask)
    v = jnp.asarray(valid_mask).astype(tok.dtype)
    b = jnp.asarray(binder_mask).astype(tok.dtype)
    return {
        "token_plddt": tok,
        "mean_plddt": float(jnp.sum(tok * v) / (jnp.sum(v) + _EPS)),
        "binder_plddt": float(jnp.sum(tok * b) / (jnp.sum(b) + _EPS)),
    }


def pae_metrics(full_pae, binder_mask, target_mask, valid_mask) -> dict:
    """Symmetrized PAE collapsed to BindCraft pae / i_pae losses (+ raw means).

    pae_loss   = mean(sym_pae over binder rows x ALL valid cols) / 31.
    i_pae_loss = mean(sym_pae over binder rows x TARGET cols) / 31.
    Raw `pae_mean` / `interface_pae_mean` are returned for reporting only; the
    scoring loss uses the normalized *_loss values (== raw_mean / 31).
    """
    pae = _reduce_samples(full_pae, 2)               # [N, N]
    pae_sym = (pae + pae.T) / 2.0
    b = jnp.asarray(binder_mask).astype(pae_sym.dtype)
    t = jnp.asarray(target_mask).astype(pae_sym.dtype)
    v = jnp.asarray(valid_mask).astype(pae_sym.dtype)
    m_pae = b[:, None] * v[None, :]                  # binder rows x all valid cols
    m_ipae = b[:, None] * t[None, :]                 # binder rows x target cols
    pae_mean = float(jnp.sum(pae_sym * m_pae) / (jnp.sum(m_pae) + _EPS))
    interface_pae_mean = float(jnp.sum(pae_sym * m_ipae) / (jnp.sum(m_ipae) + _EPS))
    return {
        "pae_loss": pae_mean / 31.0,
        "i_pae_loss": interface_pae_mean / 31.0,
        "pae_mean": pae_mean,
        "interface_pae_mean": interface_pae_mean,
        "pae_mask_count": int(jnp.sum(m_pae)),
        "i_pae_mask_count": int(jnp.sum(m_ipae)),
    }


@dataclasses.dataclass(frozen=True)
class ScoringWeights:
    """BindCraft default_4stage_multimer.json weights (note 014 section 4)."""

    con: float = 1.0
    i_con: float = 1.0
    pae: float = 0.4
    i_pae: float = 0.1
    plddt: float = 0.1
    iptm: float = 0.05
    helix: float = -0.3


def scoring_loss(con, i_con, pae_loss, i_pae_loss, binder_plddt, iptm,
                 helix=None, weights: ScoringWeights | None = None) -> tuple:
    """BindCraft semigreedy candidate scalar loss (forward scoring; min better).

    con / i_con come from the trunk distogram aux (013 con_loss); pae_loss /
    i_pae_loss / binder_plddt / iptm come from the confidence forward. helix
    (trunk distogram) is optional. ranking_score must NOT be passed here.

    plddt_loss = 1 - binder_plddt/100; iptm_loss = 1 - iptm. Returns
    (total_loss, components_dict).
    """
    w = weights or ScoringWeights()
    plddt_loss = 1.0 - binder_plddt / 100.0
    iptm_loss = 1.0 - iptm
    total = (w.con * con + w.i_con * i_con + w.pae * pae_loss
             + w.i_pae * i_pae_loss + w.plddt * plddt_loss + w.iptm * iptm_loss)
    comps = {
        "con": float(con), "i_con": float(i_con),
        "pae_loss": float(pae_loss), "i_pae_loss": float(i_pae_loss),
        "plddt_loss": float(plddt_loss), "iptm_loss": float(iptm_loss),
        "w_con_term": float(w.con * con), "w_icon_term": float(w.i_con * i_con),
        "w_pae_term": float(w.pae * pae_loss), "w_ipae_term": float(w.i_pae * i_pae_loss),
        "w_plddt_term": float(w.plddt * plddt_loss), "w_iptm_term": float(w.iptm * iptm_loss),
    }
    if helix is not None:
        total = total + w.helix * helix
        comps["helix"] = float(helix)
        comps["w_helix_term"] = float(w.helix * helix)
    return float(total), comps

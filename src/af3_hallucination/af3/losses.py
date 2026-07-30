"""Differentiable contact losses on the AF3 distogram.

Faithful port of ColabDesign get_con_loss / _get_con_loss / get_helix_loss
(external/colabdesign/colabdesign/af/loss.py:261-337). These implement the real
top-k (min_k) + cutoff + seqsep + binary/categorical semantics. This module does
NOT use a block mean as a stand-in for con/i_con.

Source <-> AF3 mapping (note 006 section 6.1 + ColabDesign af/loss.py):
  - ColabDesign dgram logits  -> AF3 outputs["distogram"]["distogram"]  ([N,N,64],
    available when the trunk forward is run with return_distogram=True).
  - ColabDesign dgram_bins    -> per-bin lower edges. For AF-style 64 bins this is
    append(0, linspace(2.3125, 21.6875, 63)); get_con_loss uses bins < cutoff.
  - AF3 default first_break/last_break/num_bins = 2.3125 / 21.6875 / 64.

If only contact_probs (not the full distogram) is available, the loss collapses
to the head's own 8.0 A binary contact and only the seqsep/min_k/mask machinery
applies; prefer return_distogram=True so cutoff is honoured.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def colabdesign_dgram_bins(bin_edges: jnp.ndarray) -> jnp.ndarray:
    """Reconstruct ColabDesign's 64 distogram bin lower edges from AF3 breaks.

    ColabDesign get_dgram_bins(af/loss.py:215-221) uses
    append(0, linspace(2.3125, 21.6875, 63)) for 64-bin distograms and then tests
    dgram_bins < cutoff. AF3 returns the same 63 break values as bin_edges, so the
    faithful loss basis is append(0, bin_edges), not the AF3 contact_probs upper
    edges append(bin_edges, bin_edges[-1] + step).
    """
    return jnp.append(jnp.asarray(0.0, dtype=bin_edges.dtype), bin_edges)


def _per_pair_con(dgram_logits: jnp.ndarray, dgram_bins: jnp.ndarray,
                  cutoff: float, binary: bool) -> jnp.ndarray:
    """ColabDesign _get_con_loss: [N,N,bins] logits -> [N,N] contact loss."""
    bins = dgram_bins < cutoff
    px = jax.nn.softmax(dgram_logits, axis=-1)
    not_bins = (~bins).astype(dgram_logits.dtype)
    px_ = jax.nn.softmax(dgram_logits - 1e7 * not_bins, axis=-1)
    cat_ent = -(px_ * jax.nn.log_softmax(dgram_logits, axis=-1)).sum(-1)
    bin_ent = -jnp.log((bins * px + 1e-8).sum(-1))
    return jnp.where(binary, bin_ent, cat_ent)


def _min_k(x: jnp.ndarray, k: int, mask: jnp.ndarray | None = None) -> jnp.ndarray:
    """ColabDesign min_k: mean of the k smallest entries along last axis.

    With a mask, only masked-in entries are eligible; the rest become nan and
    are excluded. k is clipped to >=1.
    """
    k = max(int(k), 1)
    y = x if mask is None else jnp.where(mask, x, jnp.nan)
    y = jnp.sort(y, axis=-1)
    k_mask = jnp.logical_and(jnp.arange(y.shape[-1]) < k, ~jnp.isnan(y))
    return jnp.where(k_mask, y, 0.0).sum(-1) / (k_mask.sum(-1) + 1e-8)


def _offset(residue_index: jnp.ndarray) -> jnp.ndarray:
    idx = residue_index.flatten()
    return idx[:, None] - idx[None, :]


def con_loss(dgram_logits, bin_edges, residue_index, spec,
             mask_1d=None, mask_1b=None, mask_2d=None) -> jnp.ndarray:
    """Generic contact objective (ColabDesign get_con_loss).

    spec is a ContactSpec (cutoff/num/num_pos/seqsep/binary). Used for both the
    intra-binder `con` (mask_1d = mask_1b = binder) and the interface `i_con`
    (mask_1d = binder, mask_1b = hotspot/target), matching BindCraft's two calls.
    """
    dgram_bins = colabdesign_dgram_bins(bin_edges)
    p = _per_pair_con(dgram_logits, dgram_bins, spec.cutoff, spec.binary)

    offset = _offset(residue_index)
    m = jnp.abs(offset) >= spec.seqsep if spec.seqsep else jnp.ones_like(offset, dtype=bool)

    n = m.shape[0]
    if mask_1d is None:
        mask_1d = jnp.ones((n,), dtype=bool)
    if mask_1b is None:
        mask_1b = jnp.ones((n,), dtype=bool)
    m = jnp.logical_and(m, mask_1b) if mask_2d is None else jnp.logical_and(m, mask_2d)

    p = _min_k(p, spec.num, m)                       # per-row top-k contact
    num_pos = spec.num_pos if spec.num_pos is not None else n
    return _min_k(p, num_pos, mask_1d)               # average over the worst rows


def helix_loss(dgram_logits, bin_edges, residue_index, seq_mask,
               cutoff: float = 6.0) -> jnp.ndarray:
    """ColabDesign get_helix_loss: contact bias at sequence offset == 3."""
    dgram_bins = colabdesign_dgram_bins(bin_edges)
    x = _per_pair_con(dgram_logits, dgram_bins, cutoff=cutoff, binary=True)
    offset = _offset(residue_index)
    mask_2d = seq_mask[:, None] * seq_mask[None, :]
    mask = jnp.where(mask_2d.astype(bool), offset == 3, False)
    return jnp.where(mask, x, 0.0).sum() / (mask.sum() + 1e-8)


def dgram_bin_centers(bin_edges: jnp.ndarray, clamp_max: float = 27.0) -> jnp.ndarray:
    """AF3 distance representatives for the ESMFold2 Rg formula, length 64.

    ESMFold2's globularity loss consumes a per-bin distance tensor clamped at
    27 A. AF3 provides 63 distogram break points, so the internal semantic port
    uses AF3-specific representatives: the first bin center just below the first
    break, midpoints for closed bins, and a capped 27 A representative for the
    final open bin.
    """
    edges = jnp.asarray(bin_edges, dtype=jnp.float32)
    step = edges[1] - edges[0]
    centers = jnp.concatenate(
        [
            jnp.asarray([edges[0] - 0.5 * step], dtype=edges.dtype),
            0.5 * (edges[:-1] + edges[1:]),
            jnp.asarray([clamp_max], dtype=edges.dtype),
        ]
    )
    return jnp.minimum(centers, jnp.asarray(clamp_max, dtype=edges.dtype))


def distogram_globularity_loss(dgram_logits, bin_edges, binder_mask) -> jnp.ndarray:
    """ESMFold2-style coordinate-free Rg/globularity loss on AF3 distograms.

    Semantic reproduction of
    external/biohub-esm/cookbook/tutorials/binder_design.py:324-339:

        ELU(sqrt(sum_{i>j} E[d_ij^2] / N^2) - 2.38 * N^0.365)

    The strict lower triangle excludes self-pairs and double counting. The term
    reads only trunk distogram logits and never diffusion coordinates.
    """
    calc_dtype = jnp.float32
    centers = dgram_bin_centers(bin_edges)                 # [64], capped at 27 A
    px = jax.nn.softmax(jnp.asarray(dgram_logits, dtype=calc_dtype), axis=-1)
    e_d2 = jnp.sum(px * jnp.square(centers), axis=-1)
    b = binder_mask.astype(calc_dtype)
    n = jnp.sum(b)
    pair_mask = jnp.tril(b[:, None] * b[None, :], k=-1)
    rg_proxy = jnp.sqrt(jnp.sum(e_d2 * pair_mask) / (n * n + 1e-8) + 1e-8)
    rg_threshold = 2.38 * jnp.power(n, 0.365)
    loss = jax.nn.elu(rg_proxy - rg_threshold)
    return jnp.where(n > 0, loss, jnp.asarray(0.0, dtype=calc_dtype))


def seq_ent_loss(pssm: jnp.ndarray, mask_1d: jnp.ndarray | None = None) -> jnp.ndarray:
    """Sequence entropy regulariser over design positions (ColabDesign).

    pssm is the per-position amino-acid distribution (softmax of logits).
    """
    ent = -(pssm * jnp.log(pssm + 1e-8)).sum(-1)
    if mask_1d is None:
        return ent.mean()
    return (ent * mask_1d).sum() / (mask_1d.sum() + 1e-8)


def total_loss(distogram, residue_index, masks, cfg) -> tuple[jnp.ndarray, dict]:
    """Weighted V0 objective from the AF3 distogram outputs.

    distogram: outputs["distogram"] with keys "distogram" (logits) and
    "bin_edges". masks: DesignMasks. cfg: DesignConfig. Returns (loss, aux).

    BindCraft minimises contact loss (smaller -> more contacts), so the weighted
    sum is the loss directly; callers wanting a maximise-contacts gradient use
    the same sign as ColabDesign.
    """
    logits = distogram["distogram"]
    bins = distogram["bin_edges"]
    w = cfg.weights

    aux = {}
    loss = jnp.array(0.0)

    if w.con:
        aux["con"] = con_loss(logits, bins, residue_index, cfg.con_spec,
                              mask_1d=masks.binder_mask, mask_1b=masks.binder_mask)
        loss = loss + w.con * aux["con"]
    if w.i_con:
        if getattr(masks, "has_hotspot", False):
            # ColabDesign _loss_binder hotspot branch flips row/column roles:
            # each hotspot target token should find PARTNER contacts. Round3: the
            # partner set is masks.i_con_partner_mask -- the design-CDR for the antibody
            # `target_hotspot_x_design_cdr` mode (hotspot x CDR only), or the whole binder
            # for legacy `target_hotspot_x_whole_binder`. Backward-compatible fallback to
            # binder_mask if an older DesignMasks (no partner field) is passed.
            i_con_partner = getattr(masks, "i_con_partner_mask", None)
            if i_con_partner is None:
                i_con_partner = masks.binder_mask
            aux["i_con"] = con_loss(
                logits, bins, residue_index, cfg.i_con_spec,
                mask_1d=masks.hotspot_mask, mask_1b=i_con_partner)
            # Germinal-style paratope localization modulation (note 191). When
            # framework_contact_loss is enabled, the positive i_con (hotspot x design-core)
            # is modulated by the off-design binder i_con (hotspot x off_design_binder)
            # to penalize cases where framework/non-design residues dominate the interface.
            # i_con_modulated = i_con * (i_con / max(off_design_i_con - offset, 1e-8)).
            # This strongly downweights solutions where the framework contributes
            # substantial hotspot contact. Only meaningful with explicit off_design mask.
            if getattr(w, "framework_contact_loss", False):
                off_mask = getattr(masks, "off_design_binder_mask", None)
                if off_mask is not None:
                    offset = float(getattr(w, "framework_contact_offset", 1.0))
                    aux["off_design_i_con"] = con_loss(
                        logits, bins, residue_index, cfg.i_con_spec,
                        mask_1d=masks.hotspot_mask, mask_1b=off_mask)
                    aux["i_con_raw"] = aux["i_con"]
                    aux["i_con"] = aux["i_con"] * (
                        aux["i_con"] / jnp.maximum(aux["off_design_i_con"] - offset, 1e-8))
        else:
            aux["i_con"] = con_loss(
                logits, bins, residue_index, cfg.i_con_spec,
                mask_1d=masks.binder_mask, mask_1b=masks.target_mask)
        loss = loss + w.i_con * aux["i_con"]
    if w.helix:
        aux["helix"] = helix_loss(logits, bins, residue_index, masks.binder_mask,
                                 cutoff=cfg.helix_cutoff)
        loss = loss + w.helix * aux["helix"]
    # Round13 globularity (Rg compactness) proxy: opt-in only. getattr keeps backward
    # compatibility with any older LossWeights without a `glob` field; default 0.0 = OFF,
    # so the trunk_only / BindCraft-parity objective is unchanged unless explicitly enabled.
    if getattr(w, "glob", 0.0):
        aux["glob"] = distogram_globularity_loss(logits, bins, masks.binder_mask)
        loss = loss + w.glob * aux["glob"]

    return loss, aux

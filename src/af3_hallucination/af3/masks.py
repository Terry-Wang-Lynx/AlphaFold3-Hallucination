"""Design / target / binder / hotspot masks for an AF3 batch.

BindCraft's binder protocol fixes the target and optimises a binder; the masks
below express that on AF3 tokens. AF3 has no native "design region" feature, so
masks are built from token-level fields (asym_id / token ranges / is_ligand),
per the designable-batch-probe conclusion (#6/#7), note 006 section 6.5, and the
mask-generalization source spec (writing/research-notes/012).

Masks are plain boolean/float arrays so they can be built from numpy on the host
and passed into the jitted forward as static fixed inputs.

Generalization (note 012): the original prototype assumed a single contiguous
binder range. build_masks() now accepts an explicit binder source (range |
asym_id | bool mask), an optional non-contiguous design-token-index set, and an
optional explicit target, so it covers (a) the legacy contiguous protein binder,
(b) a protein binder against a small-molecule (is_ligand) target, and (c) an
antibody chain whose design positions are a multi-segment CDR subset. The loss /
soft-sequence injection / MSA query-row adapter already work on bool masks and
are unchanged. Non-contiguous design positions are a semantic reproduction of
ColabDesign fix_pos/pos (af/inputs.py:35-47), not new design logic.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np


def _index_mask(name: str, indices, n: int) -> np.ndarray:
    """Convert token indices to a bool mask, with clear bounds errors."""
    idx = np.asarray(indices, dtype=int)
    if idx.size and (np.any(idx < 0) or np.any(idx >= n)):
        raise ValueError(f"build_masks: {name} contains token indices outside [0, {n}).")
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask


@dataclasses.dataclass
class DesignMasks:
    """All masks needed by the loss/feature layers.

    Shapes are [num_tokens] for 1d masks and [num_tokens, num_tokens] for the
    interface 2d mask. All in the bucketed token frame (length == bucket), with
    padding tokens masked out via `seq_mask`.
    """

    seq_mask: jnp.ndarray        # [N] valid (non-padding) tokens
    binder_mask: jnp.ndarray     # [N] full binder (whole chain; fed to loss)
    target_mask: jnp.ndarray     # [N] fixed target (protein/ligand/etc.) tokens
    hotspot_mask: jnp.ndarray    # [N] target hotspot subset (== target if none)
    has_hotspot: bool            # True only when the caller supplied hotspots
    design_mask: jnp.ndarray     # [N] positions whose aatype is mutable (subset of binder)
    design_indices: jnp.ndarray  # [L] sorted token indices where design_mask is True
    interface_2d: jnp.ndarray    # [N, N] binder x (hotspot|target) pairs
    # Round3: the binder-side PARTNER set for the i_con interface reward. For the
    # antibody mode `target_hotspot_x_design_cdr` it is the design-CDR mask (so i_con
    # couples hotspot x CDR only); for the legacy `target_hotspot_x_whole_binder` it is
    # the whole binder. losses.total_loss uses this as the i_con partner (mask_1b).
    i_con_partner_mask: jnp.ndarray = None   # type: ignore[assignment]
    interface_mode: str = "target_hotspot_x_whole_binder"
    # Germinal-style off-design paratope masks (note 191). off_design_binder_mask
    # covers ALL binder tokens outside the design set (fixed_cdr_support + framework).
    # Used by the framework_contact_loss term to penalise hotspot contact made by
    # non-design binder residues. Each is a subset of binder_mask and is disjoint
    # from design_mask. Default (=None for all three) when no explicit indices are
    # given: off_design_binder_mask = binder_mask & ~design_mask.
    fixed_cdr_support_mask: jnp.ndarray = None   # type: ignore[assignment]
    framework_mask: jnp.ndarray = None            # type: ignore[assignment]
    off_design_binder_mask: jnp.ndarray = None    # type: ignore[assignment]


def build_masks(
    seq_mask: jnp.ndarray,
    binder_token_range: tuple[int, int] | None = None,
    hotspot_token_indices: tuple[int, ...] = (),
    *,
    binder_mask: jnp.ndarray | None = None,
    binder_asym_id: int | None = None,
    asym_id: jnp.ndarray | None = None,
    is_ligand: jnp.ndarray | None = None,
    design_token_indices=None,
    target_token_indices=None,
    interface_mode: str = "target_hotspot_x_whole_binder",
    # Germinal-style paratope masks (note 191). Optional explicit indices for the
    # fixed-CDR-support and framework subsets of the binder. When both are given,
    # off_design_binder_mask = fixed_cdr_support_mask | framework_mask. When neither
    # is given (default), off_design_binder_mask = binder_mask & ~design_mask
    # (backward-compatible; no explicit separation between support and framework).
    fixed_support_token_indices: tuple[int, ...] | None = None,
    framework_token_indices: tuple[int, ...] | None = None,
) -> DesignMasks:
    """Build design/target/binder/hotspot masks from a generic binder source.

    Exactly ONE binder source must be given (priority is irrelevant since they
    are mutually exclusive):
      - `binder_token_range=(start, end)`: legacy contiguous protein binder.
      - `binder_asym_id=<chain id>` (+ `asym_id`): binder = that chain.
      - `binder_mask=<bool[N]>`: fully explicit.

    `design_token_indices` (int[L]) selects the mutable positions; default is the
    whole binder (V0 behaviour). `target_token_indices` (int[M]) gives an explicit
    target (e.g. is_ligand tokens); default is `valid & ~binder`. Hotspots, if
    given, must lie inside the target. residue_index is NOT consulted here; CDR
    residue-number ranges are resolved to token indices by `cdr_design_indices`.

    Returns a DesignMasks with `design_indices` = sorted nonzero(design_mask).
    Raises ValueError on any invariant violation (notably: ligand tokens or
    non-binder/padding tokens appearing in the design set).
    """
    valid = np.asarray(seq_mask).astype(bool)
    n = valid.shape[0]
    idx = np.arange(n)

    sources = [binder_mask is not None, binder_asym_id is not None,
               binder_token_range is not None]
    if sum(sources) != 1:
        raise ValueError(
            "build_masks: provide exactly one of binder_mask / binder_asym_id / "
            f"binder_token_range (got {sum(sources)}).")

    if binder_mask is not None:
        binder = np.asarray(binder_mask).astype(bool)
        if binder.shape != valid.shape:
            raise ValueError("build_masks: binder_mask must have shape [N].")
        if np.any(binder & ~valid):
            raise ValueError("build_masks: binder_mask must be a subset of valid tokens.")
    elif binder_asym_id is not None:
        if asym_id is None:
            raise ValueError("build_masks: binder_asym_id requires asym_id.")
        binder = (np.asarray(asym_id) == binder_asym_id) & valid
    else:
        start, end = binder_token_range
        binder = valid & (idx >= start) & (idx < end)

    if target_token_indices is not None:
        target = _index_mask("target_token_indices", target_token_indices, n)
        if np.any(target & ~valid):
            raise ValueError("build_masks: target_token_indices must be valid tokens.")
        if np.any(target & binder):
            raise ValueError("build_masks: target_mask must not overlap binder_mask.")
    else:
        target = valid & ~binder

    has_hotspot = len(hotspot_token_indices) > 0
    if has_hotspot:
        hotspot = _index_mask("hotspot_token_indices", hotspot_token_indices, n)
        if np.any(hotspot & ~target):
            raise ValueError("build_masks: hotspot_mask must be a subset of target_mask.")
    else:
        hotspot = target

    if design_token_indices is not None:
        design = _index_mask("design_token_indices", design_token_indices, n)
    else:
        design = binder.copy()

    # ---- invariants (note 012 section 3 + risks R1/R2/R3) ----
    if np.any(design & ~binder):
        raise ValueError(
            "build_masks: design_mask must be a subset of binder_mask "
            "(design positions must lie on the binder chain).")
    if np.any(binder & target):
        raise ValueError("build_masks: binder_mask and target_mask overlap.")
    if is_ligand is not None and np.any(design & np.asarray(is_ligand).astype(bool)):
        raise ValueError(
            "build_masks: ligand tokens must never be in design_mask "
            "(is_ligand tokens are per-atom and not designable).")

    design_indices = np.flatnonzero(design)
    # Round3: i_con interface partner set. target_hotspot_x_design_cdr -> the design-CDR
    # mask (i_con couples hotspot x CDR only; framework/target never partner); legacy
    # target_hotspot_x_whole_binder -> the whole binder. By construction partner is a
    # subset of the binder; in CDR mode it is also a subset of design (no framework leak).
    if interface_mode not in ("target_hotspot_x_whole_binder", "target_hotspot_x_design_cdr"):
        raise ValueError(f"build_masks: unknown interface_mode {interface_mode!r}")
    i_con_partner = design if interface_mode == "target_hotspot_x_design_cdr" else binder
    interface = i_con_partner[:, None] & hotspot[None, :]

    # ---- Germinal-style off-design paratope masks (note 191) ----
    # Build the off-design binder mask for the framework_contact_loss term. When
    # explicit fixed_support and framework indices are given, they must be subsets
    # of the binder and disjoint from the design set; off_design = support | framework.
    # When neither is given (default), off_design = binder & ~design (backward-compatible;
    # no explicit separation between support and framework). Providing exactly one of
    # the two is an error (they are a pair from the manifest).
    fixed_support_mask_np = None
    framework_mask_np = None
    has_explicit_off_design = (fixed_support_token_indices is not None
                               or framework_token_indices is not None)
    if has_explicit_off_design:
        if fixed_support_token_indices is None or framework_token_indices is None:
            raise ValueError(
                "build_masks: fixed_support_token_indices and framework_token_indices "
                "must both be provided together, or both be None (default).")
        fixed_support_mask_np = _index_mask("fixed_support_token_indices",
                                             fixed_support_token_indices, n)
        framework_mask_np = _index_mask("framework_token_indices",
                                         framework_token_indices, n)
        # Validate: subsets of binder, disjoint from design
        for name, m in (("fixed_support_mask", fixed_support_mask_np),
                        ("framework_mask", framework_mask_np)):
            if np.any(m & ~binder):
                raise ValueError(f"build_masks: {name} must be a subset of binder_mask.")
            if np.any(m & design):
                raise ValueError(f"build_masks: {name} must be disjoint from design_mask.")
        off_design = fixed_support_mask_np | framework_mask_np
        # Validate off_design disjoint from design
        if np.any(off_design & design):
            raise ValueError("build_masks: off_design_mask must be disjoint from design_mask.")
    else:
        off_design = binder & ~design

    return DesignMasks(
        seq_mask=jnp.asarray(valid),
        binder_mask=jnp.asarray(binder),
        target_mask=jnp.asarray(target),
        hotspot_mask=jnp.asarray(hotspot),
        has_hotspot=has_hotspot,
        design_mask=jnp.asarray(design),
        design_indices=jnp.asarray(design_indices),
        interface_2d=jnp.asarray(interface),
        i_con_partner_mask=jnp.asarray(i_con_partner),
        interface_mode=interface_mode,
        fixed_cdr_support_mask=(jnp.asarray(fixed_support_mask_np)
                                 if fixed_support_mask_np is not None else None),
        framework_mask=(jnp.asarray(framework_mask_np)
                        if framework_mask_np is not None else None),
        off_design_binder_mask=jnp.asarray(off_design),
    )


def cdr_design_indices(asym_id, residue_index, chain_asym_id,
                       residue_ranges) -> np.ndarray:
    """Map (antibody chain + CDR residue-number ranges) to design token indices.

    AF3 has no native CDR annotation; the design region must be given explicitly
    as residue-number ranges on a chain (note 012 section 5). `residue_index` is
    reset per chain (prep_inputs_field_probe), so we MUST filter by `asym_id`
    FIRST and only then by residue ranges, otherwise residue numbers shared
    across chains would cross-select (risk R2).

    Args:
      asym_id: [N] per-token chain id (from batch).
      residue_index: [N] per-token, per-chain-reset residue number (from batch).
      chain_asym_id: the binder/antibody chain id to restrict to.
      residue_ranges: iterable of inclusive (lo, hi) residue-number pairs (the
        CDR segments, e.g. H1/H2/H3).

    Returns a sorted numpy int array of token indices on that chain inside any
    range. Pass it to build_masks(design_token_indices=...).
    """
    asym = np.asarray(asym_id)
    res = np.asarray(residue_index)
    on_chain = asym == chain_asym_id
    in_range = np.zeros(asym.shape, dtype=bool)
    for lo, hi in residue_ranges:
        in_range |= (res >= lo) & (res <= hi)
    return np.flatnonzero(on_chain & in_range)

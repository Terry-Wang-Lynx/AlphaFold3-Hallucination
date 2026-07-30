"""Hard (discrete) candidate-sequence injection into an AF3 predict batch.

V0 token-only injection for the OFFICIAL AF3 predict path (note 015). A
semigreedy candidate is a discrete sequence on the design positions; unlike the
differentiable soft path (which must intercept create_target_feat /
create_msa_feat with floating one-hots via DesignEvoformer), a discrete sequence
can be written straight into the integer/float batch fields and fed to
ModelRunner.run_inference -- no custom Model wrapper.

Semantic reproduction of ColabDesign update_seq (query one-hot) + update_aatype
(discrete aatype), af/inputs.py:109-135, mapped to AF3's three token-level
sequence-identity fields (note 015 section 2):

    aatype[design_indices]  = seq_indices              (update_aatype)
    msa[0, design_indices]  = seq_indices  (query row) (update_seq one-hot row)
    profile[design_indices] = one_hot(seq_indices, 31) (update_seq seq_pssm; empty
                                                        MSA -> profile == query one-hot)

V0 APPROXIMATION (note 015 section 3) -- NOT a full re-featurisation:
  All ref_* / dense-atom layout / gather_idxs / layout-object fields are kept
  FIXED. AF3 precomputes atom layout per CCD residue at featurise time, so a hard
  aatype mutation does NOT update those fields the way AF2's update_aatype derives
  atom14/atom37 on the fly. token_only_bias.json shows this is ~loss-free on the
  TRUNK distogram (mean 0.0016 / max 0.064), but the diffusion+confidence
  predict-path bias is UNTESTED (note 015 R1) and must be quantified by a separate
  probe before this is used for full scoring. Do NOT claim full re-featurisation
  equivalence.

This helper is pure-numpy and runs nothing (no AF3, no diffusion/confidence, no
GPU). It deliberately mutates ONLY the three array fields and passes every other
field (arrays AND non-array layout objects) through by identity -- NEVER use
jax.tree_map / a whole-batch tree op, which would break on the AtomLayout-style
object fields (note 015 section 4 / R4).
"""

from __future__ import annotations

import numpy as np

# The three token-level sequence-identity fields that a hard candidate updates.
SEQUENCE_IDENTITY_FIELDS = ("aatype", "msa", "profile")


def inject_hard_sequence(batch, seq_indices, design_indices, *,
                         is_ligand=None, width: int = 31) -> dict:
    """Return a shallow-copied batch with a hard candidate written on design positions.

    Args:
      batch: featurised AF3 batch dict (must contain aatype [N], msa [num_msa, N],
        profile [N, width]). NOT mutated in place.
      seq_indices: int array-like [L] -- candidate aa ids on the design positions.
      design_indices: int array-like [L] -- token indices (may be non-contiguous,
        e.g. antibody CDR segments). seq_indices[i] is written at design_indices[i].
      is_ligand: optional [N] mask; if any design index is a ligand token this
        raises (note 012 invariant: ligand tokens are never designable).
      width: profile one-hot width (AF3 polymer token width, 31).

    Returns a NEW dict: aatype/msa/profile are fresh modified copies; every other
    field (including non-array layout objects) is the SAME object as in `batch`.
    """
    seq_indices = np.asarray(seq_indices)
    design_indices = np.asarray(design_indices)
    if seq_indices.ndim != 1 or design_indices.ndim != 1:
        raise ValueError("inject_hard_sequence: seq_indices and design_indices must be 1-D.")
    if not np.issubdtype(seq_indices.dtype, np.integer):
        raise ValueError("inject_hard_sequence: seq_indices must contain integer aa ids.")
    if not np.issubdtype(design_indices.dtype, np.integer):
        raise ValueError("inject_hard_sequence: design_indices must contain integer token ids.")
    if seq_indices.shape[0] != design_indices.shape[0]:
        raise ValueError(
            "inject_hard_sequence: len(seq_indices) "
            f"({seq_indices.shape[0]}) != len(design_indices) ({design_indices.shape[0]}).")
    for field in SEQUENCE_IDENTITY_FIELDS:
        if field not in batch:
            raise KeyError(f"inject_hard_sequence: batch is missing required field '{field}'.")

    aatype = np.asarray(batch["aatype"])
    n = aatype.shape[0]
    if design_indices.size and (np.any(design_indices < 0) or np.any(design_indices >= n)):
        raise ValueError(
            f"inject_hard_sequence: design_indices out of range [0, {n}).")
    if seq_indices.size and (np.any(seq_indices < 0) or np.any(seq_indices >= width)):
        raise ValueError(
            f"inject_hard_sequence: seq_indices out of aa range [0, {width}).")
    if is_ligand is not None:
        lig = np.asarray(is_ligand).astype(bool)
        if np.any(lig[design_indices]):
            raise ValueError(
                "inject_hard_sequence: design_indices include ligand tokens "
                "(is_ligand tokens are never designable).")

    # shallow copy: every value reference is shared, then we swap the three
    # sequence-identity fields for fresh modified copies. NO jax.tree_map.
    new_batch = dict(batch)

    new_aatype = np.array(batch["aatype"])           # writable copy
    new_aatype[design_indices] = seq_indices.astype(new_aatype.dtype)

    new_msa = np.array(batch["msa"])                 # [num_msa, N]
    new_msa[0, design_indices] = seq_indices.astype(new_msa.dtype)

    new_profile = np.array(batch["profile"])         # [N, width]
    onehot = np.eye(width, dtype=new_profile.dtype)[seq_indices]
    new_profile[design_indices] = onehot

    new_batch["aatype"] = new_aatype
    new_batch["msa"] = new_msa
    new_batch["profile"] = new_profile
    return new_batch

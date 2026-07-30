"""No-diffusion AF3 ConfidenceHead scoring on fixed anchor pseudo-beta geometry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def mean_masked(values, mask):
    import jax.numpy as jnp

    values = jnp.asarray(values, jnp.float32)
    mask = jnp.asarray(mask, bool)
    return jnp.sum(jnp.where(mask, values, 0.0)) / (jnp.sum(mask) + 1e-8)


def four_term_consistency_loss(result, batch, context: Mapping[str, Any]):
    """Return the validated CDR confidence/PAE/PDE consistency objective."""

    import jax.numpy as jnp

    atom_mask = jnp.asarray(batch["pred_dense_atom_mask"], jnp.float32)
    predicted = jnp.asarray(result["predicted_lddt"], jnp.float32)
    token_plddt = jnp.sum(predicted * atom_mask, axis=-1) / (
        jnp.sum(atom_mask, axis=-1) + 1e-8
    )
    cdr = jnp.asarray(context["cdr_mask"], bool)
    cdr3 = jnp.asarray(context.get("cdr3_mask", context["cdr_mask"]), bool)
    patch = jnp.asarray(context["patch_mask"], bool)
    pae = jnp.asarray(result["full_pae"], jnp.float32)
    pde = jnp.asarray(result["full_pde"], jnp.float32)
    pair = (cdr[:, None] & patch[None, :]) | (patch[:, None] & cdr[None, :])
    terms = {
        "cdr_plddt_loss": 1.0 - mean_masked(token_plddt / 100.0, cdr),
        "cdr3_plddt_loss": 1.0 - mean_masked(token_plddt / 100.0, cdr3),
        "cdr_patch_pae_loss": mean_masked(pae / 31.0, pair),
        "cdr_patch_pde_loss": mean_masked(pde / 31.0, pair),
    }
    total = sum(terms.values()) / 4.0
    return total, {
        "consistency_loss": total,
        **terms,
        "all_cdr_plddt": mean_masked(token_plddt, cdr),
        "cdr3_plddt": mean_masked(token_plddt, cdr3),
        "cdr_patch_pae_symmetric": mean_masked(pae, pair),
        "cdr_patch_pde_symmetric": mean_masked(pde, pair),
    }


def scatter_pseudo_beta(batch_obj, pseudo_beta):
    """Scatter one pseudo-beta coordinate per token into AF3 dense atom layout."""

    import numpy as np

    info = batch_obj.pseudo_beta_info.token_atoms_to_pseudo_beta
    gather_indices = np.asarray(info.gather_idxs, np.int64)
    gather_mask = np.asarray(info.gather_mask, bool)
    input_shape = tuple(int(value) for value in np.asarray(info.input_shape).tolist())
    dense = np.zeros(input_shape + (3,), np.float32)
    dense.reshape((-1, 3))[gather_indices[gather_mask]] = np.asarray(pseudo_beta)[gather_mask]
    return dense


def conditioned_confidence_model_class():
    """Build the optional AF3 module class lazily to keep core imports portable."""

    import jax
    import jax.numpy as jnp
    from alphafold3.model.components import mapping
    from alphafold3.model.network import confidence_head, distogram_head

    from .hybrid import _HybridEmbeddingModel

    class StructureConditionedConfidence(_HybridEmbeddingModel):
        def __call__(
            self,
            batch,
            pseudo_full,
            profile_full,
            dense_atom_positions,
            geometry_valid_mask,
            key=None,
        ):
            if key is None:
                key = jax.random.PRNGKey(0)
            batch_obj, embeddings = self._hybrid_embeddings(
                batch, pseudo_full, profile_full, key
            )
            positions = jnp.asarray(dense_atom_positions)
            confidence_mask = batch_obj.token_features.mask * jnp.asarray(
                geometry_valid_mask, batch_obj.token_features.mask.dtype
            )
            confidence_batched = mapping.sharded_map(
                lambda sample_positions: confidence_head.ConfidenceHead(
                    self.config.heads.confidence, self.global_config
                )(
                    dense_atom_positions=sample_positions,
                    embeddings=embeddings,
                    seq_mask=confidence_mask,
                    token_atoms_to_pseudo_beta=batch_obj.pseudo_beta_info.token_atoms_to_pseudo_beta,
                    asym_id=batch_obj.token_features.asym_id,
                ),
                in_axes=0,
            )(positions[None, ...])
            confidence = jax.tree_util.tree_map(lambda value: value[0], confidence_batched)
            distogram = distogram_head.DistogramHead(
                self.config.heads.distogram, self.global_config
            )(batch_obj, embeddings, return_distogram=True)
            return {"external_atom_positions": positions, "distogram": distogram, **confidence}

    return StructureConditionedConfidence

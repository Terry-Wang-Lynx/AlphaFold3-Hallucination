"""ColabDesign-like hybrid AF3 trunk and one-sample full gate.

Hard sequence re-featurisation supplies atom chemistry. Exact pseudo is written
to target aatype, the MSA query row and the profile channel, matching the
default ColabDesign update_seq path. Official AF3 diffusion and ConfidenceHead
are reused once per gate checkpoint.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.data import featurisation
from alphafold3.model import feat_batch
from alphafold3.model import model as af_model
from alphafold3.model.components import mapping, utils
from alphafold3.model.network import (
    atom_cross_attention,
    confidence_head,
    diffusion_head,
    distogram_head,
)

from .trunk import WIDTH, DesignEvoformer

PROTEIN_ALPHABET = "ARNDCQEGHILKMFPSTWYV"


class TemplateDetachedDesignEvoformer(DesignEvoformer):
    """Keep template forward context while excluding its backward tape.

    The full-AF3 gate is forward-only, so stop_gradient is numerically inert
    there. During AF3 sequence optimisation this prevents the known bucket-256
    template-pair backward allocation from exceeding a 24 GB GPU.
    """

    @hk.transparent
    def _embed_template_pair(self, batch, pair_activations, pair_mask, key):
        summed, key = super()._embed_template_pair(
            batch=batch,
            pair_activations=pair_activations,
            pair_mask=pair_mask,
            key=key,
        )
        template_delta = summed - pair_activations
        return pair_activations + jax.lax.stop_gradient(template_delta), key


def replace_first_a3m_sequence(a3m: str, sequence: str) -> str:
    lines = a3m.splitlines()
    if not lines:
        return f">query\n{sequence}\n"
    output = []
    replaced = False
    in_first = False
    for line in lines:
        if line.startswith(">"):
            if in_first and not replaced:
                output.append(sequence)
                replaced = True
            output.append(line)
            in_first = not replaced
        elif in_first:
            continue
        else:
            output.append(line)
    if in_first and not replaced:
        output.append(sequence)
    return "\n".join(output) + "\n"


def candidate_input(base_input: Path, hard_ids, output: Path, binder_chain="B") -> Path:
    obj = json.loads(base_input.read_text())
    sequence = "".join(PROTEIN_ALPHABET[int(index)] for index in np.asarray(hard_ids))
    found = False
    for entry in obj["sequences"]:
        protein = entry.get("protein")
        identifier = None if protein is None else protein.get("id")
        if protein and (
            identifier == binder_chain
            or (isinstance(identifier, list) and identifier == [binder_chain])
        ):
            protein["sequence"] = sequence
            protein["unpairedMsa"] = replace_first_a3m_sequence(
                protein.get("unpairedMsa", ""), sequence
            )
            protein["pairedMsa"] = ""
            found = True
            break
    if not found:
        raise KeyError(f"Binder chain {binder_chain!r} missing from {base_input}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(obj, indent=2) + "\n")
    return output


def featurise(path: Path, bucket: int):
    fold_input = list(folding_input.load_fold_inputs_from_path(path))[0]
    batch = featurisation.featurise_input(
        fold_input=fold_input, ccd=chemical_components.Ccd(), buckets=[bucket]
    )[0]
    batch = utils.remove_invalidly_typed_feats(batch)
    return jax.tree_util.tree_map(jnp.asarray, batch)


def binder_mask_from_batch(batch_dict, binder_asym_id=2):
    return (
        (np.asarray(batch_dict["asym_id"]) == int(binder_asym_id))
        & (np.asarray(batch_dict["seq_mask"]) > 0)
    )


class _HybridEmbeddingModel(af_model.Model):
    def __init__(
        self,
        config,
        remat_cfg,
        design_mask,
        *,
        num_design_recycles=0,
        detach_recycles=False,
        name="diffuser",
    ):
        super().__init__(config, name=name)
        self._remat_cfg = remat_cfg
        self._design_mask = design_mask
        self._num_design_recycles = int(num_design_recycles)
        self._detach_recycles = bool(detach_recycles)

    @hk.transparent
    def _hybrid_embeddings(self, batch_dict, pseudo_full, profile_full, key):
        batch = feat_batch.Batch.from_data_dict(batch_dict)
        evo_cfg = self.config.evoformer
        global_config = self.global_config
        dtype = jnp.bfloat16 if global_config.bfloat16 == "all" else jnp.float32

        with utils.bfloat16_context():
            aatype_onehot = jax.nn.one_hot(batch.token_features.aatype, WIDTH)
            mask = self._design_mask[:, None].astype(aatype_onehot.dtype)
            hybrid_aatype = aatype_onehot * (1.0 - mask) + pseudo_full * mask
            hybrid_profile = batch.msa.profile * (1.0 - mask) + profile_full * mask
            target_feat = jnp.concatenate(
                [hybrid_aatype, hybrid_profile, batch.msa.deletion_mean[..., None]], axis=-1
            ).astype(dtype)
            atom_encoding = atom_cross_attention.atom_cross_att_encoder(
                token_atoms_act=None,
                trunk_single_cond=None,
                trunk_pair_cond=None,
                config=evo_cfg.per_atom_conditioning,
                global_config=global_config,
                batch=batch,
                name="evoformer_conditioning",
            )
            target_feat = jnp.concatenate(
                [target_feat, atom_encoding.token_act], axis=-1
            ).astype(dtype)

        embedding_module = TemplateDetachedDesignEvoformer(
            evo_cfg,
            global_config,
            self._remat_cfg,
            pseudo_full,
            self._design_mask,
            True,
        )
        prev = {
            "pair": jnp.zeros(
                [batch.num_res, batch.num_res, evo_cfg.pair_channel], jnp.float32
            ),
            "single": jnp.zeros([batch.num_res, evo_cfg.seq_channel], jnp.float32),
            "target_feat": target_feat,
        }

        def body(previous, rng_key):
            rng_key, subkey = jax.random.split(rng_key)
            embeddings = embedding_module(
                batch=batch, prev=previous, target_feat=target_feat, key=subkey
            )
            embeddings = {
                "pair": embeddings["pair"].astype(jnp.float32),
                "single": embeddings["single"].astype(jnp.float32),
                "target_feat": target_feat,
            }
            return embeddings, rng_key

        embeddings = prev
        rng_key = key
        recycle_count = self._num_design_recycles + 1
        for recycle_index in range(recycle_count):
            if self._detach_recycles and recycle_index < recycle_count - 1:
                embeddings = {
                    name: jax.lax.stop_gradient(value) for name, value in embeddings.items()
                }
            embeddings, rng_key = body(embeddings, rng_key)
        return batch, embeddings


class HybridDesignTrunk(_HybridEmbeddingModel):
    """Hybrid AF3 trunk returning only model-native distogram logits."""

    def __call__(self, batch, pseudo_full, profile_full, key=None):
        if key is None:
            key = hk.next_rng_key()
        batch_obj, embeddings = self._hybrid_embeddings(batch, pseudo_full, profile_full, key)
        return distogram_head.DistogramHead(
            self.config.heads.distogram, self.global_config
        )(batch_obj, embeddings, return_distogram=True)


class HybridFullGate(_HybridEmbeddingModel):
    """One official AF3 diffusion sample plus ConfidenceHead for a hybrid input."""

    def __init__(self, *args, return_embeddings=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._return_embeddings = bool(return_embeddings)

    def __call__(self, batch, pseudo_full, profile_full, key=None, diffusion_key=None):
        if key is None:
            key = hk.next_rng_key()
        if diffusion_key is None:
            diffusion_key = jax.random.fold_in(key, 1)
        batch_obj, embeddings = self._hybrid_embeddings(batch, pseudo_full, profile_full, key)
        denoising_step = functools.partial(
            self.diffusion_module,
            batch=batch_obj,
            embeddings=embeddings,
            use_conditioning=True,
        )
        samples = diffusion_head.sample(
            denoising_step=denoising_step,
            batch=batch_obj,
            key=diffusion_key,
            config=self.config.heads.diffusion.eval,
        )
        positions = samples["atom_positions"][0]
        confidence_batched = mapping.sharded_map(
            lambda dense_atom_positions: confidence_head.ConfidenceHead(
                self.config.heads.confidence, self.global_config
            )(
                dense_atom_positions=dense_atom_positions,
                embeddings=embeddings,
                seq_mask=batch_obj.token_features.mask,
                token_atoms_to_pseudo_beta=batch_obj.pseudo_beta_info.token_atoms_to_pseudo_beta,
                asym_id=batch_obj.token_features.asym_id,
            ),
            in_axes=0,
        )(samples["atom_positions"])
        confidence = jax.tree_util.tree_map(lambda value: value[0], confidence_batched)
        distogram = distogram_head.DistogramHead(
            self.config.heads.distogram, self.global_config
        )(batch_obj, embeddings, return_distogram=True)
        output = {
            "diffusion_atom_positions": positions,
            "distogram": distogram,
            **confidence,
        }
        if self._return_embeddings:
            output["single_embeddings"] = embeddings["single"]
            output["pair_embeddings"] = embeddings["pair"]
        return output


class HybridInferenceModel(_HybridEmbeddingModel):
    """Hybrid-query trunk followed by official AF3 diffusion and confidence heads.

    Unlike :class:`HybridFullGate`, this class preserves the leading diffusion
    sample axis and the standard ``diffusion_samples`` key so AF3's official
    inference-result extraction and post-processing can be reused unchanged.
    """

    def __call__(self, batch, pseudo_full, profile_full, key=None, diffusion_key=None):
        if key is None:
            key = hk.next_rng_key()
        if diffusion_key is None:
            diffusion_key = jax.random.fold_in(key, 1)
        batch_obj, embeddings = self._hybrid_embeddings(
            batch, pseudo_full, profile_full, key
        )
        denoising_step = functools.partial(
            self.diffusion_module,
            batch=batch_obj,
            embeddings=embeddings,
            use_conditioning=True,
        )
        samples = diffusion_head.sample(
            denoising_step=denoising_step,
            batch=batch_obj,
            key=diffusion_key,
            config=self.config.heads.diffusion.eval,
        )
        confidence = mapping.sharded_map(
            lambda dense_atom_positions: confidence_head.ConfidenceHead(
                self.config.heads.confidence, self.global_config
            )(
                dense_atom_positions=dense_atom_positions,
                embeddings=embeddings,
                seq_mask=batch_obj.token_features.mask,
                token_atoms_to_pseudo_beta=(
                    batch_obj.pseudo_beta_info.token_atoms_to_pseudo_beta
                ),
                asym_id=batch_obj.token_features.asym_id,
            ),
            in_axes=0,
        )(samples["atom_positions"])
        distogram = distogram_head.DistogramHead(
            self.config.heads.distogram, self.global_config
        )(batch_obj, embeddings, return_distogram=True)
        return {"diffusion_samples": samples, "distogram": distogram, **confidence}


def full_sequence_features(batch, binder_mask, binder_pseudo, binder_profile=None):
    """Scatter binder distributions into padded full-token feature arrays."""
    binder_mask = np.asarray(binder_mask, bool)
    binder_pseudo = jnp.asarray(binder_pseudo)
    if binder_profile is None:
        binder_profile = binder_pseudo
    binder_profile = jnp.asarray(binder_profile)
    indices = jnp.asarray(np.flatnonzero(binder_mask), dtype=jnp.int32)
    n_tokens = int(np.asarray(batch["seq_mask"]).shape[0])
    pseudo_full = jnp.zeros((n_tokens, WIDTH), dtype=binder_pseudo.dtype).at[indices].set(
        binder_pseudo
    )
    profile_full = jnp.zeros((n_tokens, WIDTH), dtype=binder_profile.dtype).at[indices].set(
        binder_profile
    )
    return pseudo_full, profile_full


def binder_plddt(result, batch, binder_mask) -> float:
    predicted = np.asarray(result["predicted_lddt"], dtype=np.float32)
    atom_mask = np.asarray(batch["pred_dense_atom_mask"], dtype=np.float32)
    per_token = np.sum(predicted * atom_mask, axis=-1) / (np.sum(atom_mask, axis=-1) + 1e-8)
    valid = np.asarray(binder_mask, bool) & (np.asarray(batch["seq_mask"]) > 0)
    return float(np.mean(per_token[valid]))

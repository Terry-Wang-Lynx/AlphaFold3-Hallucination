"""AF3 trunk-only distogram forward for Linux/JAX design runs.

Ports experiment/runs/2026-06-26_af3-v0-diff-loop/scripts/diff_design_loop.py
into the package, with two prototype-grade improvements over that script:

  1. soft sequence is injected into aatype one-hot, msa.profile, AND the MSA
     query-row one-hot feature (ColabDesign update_seq semantics), not just the
     aatype channel.
  2. per-block remat is wired LOCALLY by subclassing Evoformer and wrapping the
     per-block fn of BOTH the pairformer stack (in __call__) and the MSA stack
     (in the _embed_process_msa override) in remat.maybe_remat - NO global
     monkeypatch of hk.experimental.layer_stack (which the experiment script
     used and which Codex disallowed in the prototype).

Requires jax + haiku + alphafold3 + af3.bin (trunk weights). Imports are kept at
module top-level so this file only loads where alphafold3 is installed.
DesignEvoformer.__call__ and _embed_process_msa are based on AF3 b2f3d45. The
intentional deviations are local remat wiring and the soft MSA query-row adapter;
if AF3 is bumped, both copies must be re-synced.
"""

from __future__ import annotations

import haiku as hk
import jax
import jax.numpy as jnp
from alphafold3.model import feat_batch
from alphafold3.model import model as af_model
from alphafold3.model.components import haiku_modules as hm
from alphafold3.model.components import utils
from alphafold3.model.network import (
    atom_cross_attention,
    distogram_head,
    featurization,
    modules,
)
from alphafold3.model.network import evoformer as evoformer_network

from . import features as featmod
from . import remat as rematmod
from . import sequence as seqmod

WIDTH = seqmod.AATYPE_WIDTH  # 31


class DesignEvoformer(evoformer_network.Evoformer):
    """Evoformer with design-specific sequence injection and local remat.

    Based on Evoformer.__call__ / _embed_process_msa (AF3 b2f3d45). Every
    embedding helper is inherited and called via self; name defaults to
    'evoformer' so the parameter scope matches af3.bin (subclass name would
    otherwise be 'design_evoformer').

    Intentional deviations:
      - wrap pairformer and MSA stack per-block fns in remat.maybe_remat;
      - optionally replace the MSA query-row one-hot feature with soft pseudo
        sequence, matching ColabDesign update_seq() semantics.
    """

    def __init__(self, config, global_config, remat_cfg, pseudo_full,
                 design_mask, soft_msa_query=True, name='evoformer'):
        super().__init__(config, global_config, name=name)
        self._remat_cfg = remat_cfg
        self._pseudo_full = pseudo_full
        self._design_mask = design_mask
        self._soft_msa_query = soft_msa_query

    @hk.transparent
    def _embed_process_msa(self, msa_batch, pair_activations, pair_mask, key,
                           target_feat):
        """Process MSA with local remat and optional soft query-row injection.

        @hk.transparent is preserved so submodule param scopes (msa_activations,
        extra_msa_target_feat, msa_stack) stay identical to af3.bin.
        """
        dtype = pair_activations.dtype
        if self._soft_msa_query:
            # Match ColabDesign update_seq() semantics without writing soft
            # values into AF3's integer msa.rows. Build the floating msa_feat
            # before shuffle, inject query row 0, then apply the same shuffle
            # and truncation order to both msa_batch and msa_feat.
            key, sample_key = jax.random.split(key)
            logits = (jnp.clip(jnp.sum(msa_batch.mask, axis=-1), 0.0, 1.0) - 1.0) * 1e6
            index_order = featurization.gumbel_argsort_sample_idx(sample_key, logits)
            msa_feat = featmod.create_msa_feat_with_soft_query(
                msa_batch, self._pseudo_full, self._design_mask)
            msa_batch = msa_batch.index_msa_rows(index_order)
            msa_feat = msa_feat[index_order]
            indices = jnp.arange(self.config.num_msa)
            msa_batch = msa_batch.index_msa_rows(indices)
            msa_feat = msa_feat[indices].astype(dtype)
        else:
            msa_batch, key = featurization.shuffle_msa(key, msa_batch)
            msa_batch = featurization.truncate_msa_batch(msa_batch, self.config.num_msa)
            msa_feat = featurization.create_msa_feat(msa_batch).astype(dtype)

        msa_activations = hm.Linear(
            self.config.msa_channel, name='msa_activations')(msa_feat)
        msa_activations += hm.Linear(
            self.config.msa_channel, name='extra_msa_target_feat')(target_feat)[None]
        msa_mask = msa_batch.mask.astype(dtype)

        evoformer_input = {'msa': msa_activations, 'pair': pair_activations}
        masks = {'msa': msa_mask, 'pair': pair_mask}

        def evoformer_fn(x):
            return modules.EvoformerIteration(
                self.config.msa_stack, self.global_config, name='msa_stack')(
                    activations=x, masks=masks)

        # ---- LOCAL remat wiring (the only change vs upstream) ----
        evoformer_fn = rematmod.maybe_remat(evoformer_fn, self._remat_cfg)
        evoformer_stack = hk.experimental.layer_stack(
            self.config.msa_stack.num_layer)(evoformer_fn)
        # ----------------------------------------------------------

        evoformer_output = evoformer_stack(evoformer_input)
        return evoformer_output['pair'], key

    def __call__(self, batch, prev, target_feat, key):
        assert self.global_config.bfloat16 in {'all', 'none'}
        num_residues = target_feat.shape[0]
        assert batch.token_features.aatype.shape == (num_residues,)
        dtype = jnp.bfloat16 if self.global_config.bfloat16 == 'all' else jnp.float32

        with utils.bfloat16_context():
            pair_activations, pair_mask = self._seq_pair_embedding(
                batch.token_features, target_feat)

            pair_activations += hm.Linear(
                pair_activations.shape[-1], name='prev_embedding',
                initializer=self.global_config.final_init,
            )(hm.LayerNorm(name='prev_embedding_layer_norm')(
                prev['pair'].astype(pair_activations.dtype)))

            pair_activations = self._relative_encoding(batch, pair_activations)
            pair_activations = self._embed_bonds(
                batch=batch, pair_activations=pair_activations)
            pair_activations, key = self._embed_template_pair(
                batch=batch, pair_activations=pair_activations,
                pair_mask=pair_mask, key=key)
            pair_activations, key = self._embed_process_msa(
                msa_batch=batch.msa, pair_activations=pair_activations,
                pair_mask=pair_mask, key=key, target_feat=target_feat)
            del key

            single_activations = hm.Linear(
                self.config.seq_channel, name='single_activations')(target_feat)
            single_activations += hm.Linear(
                single_activations.shape[-1], name='prev_single_embedding',
                initializer=self.global_config.final_init,
            )(hm.LayerNorm(name='prev_single_embedding_layer_norm')(
                prev['single'].astype(single_activations.dtype)))

            def pairformer_fn(x):
                pairformer_iteration = modules.PairFormerIteration(
                    self.config.pairformer, self.global_config,
                    with_single=True, name='trunk_pairformer')
                pair_act, single_act = x
                return pairformer_iteration(
                    act=pair_act, single_act=single_act, pair_mask=pair_mask,
                    seq_mask=batch.token_features.mask.astype(dtype))

            # ---- LOCAL remat wiring (the only change vs upstream __call__) ----
            pairformer_fn = rematmod.maybe_remat(pairformer_fn, self._remat_cfg)
            pairformer_stack = hk.experimental.layer_stack(
                self.config.pairformer.num_layer)(pairformer_fn)
            # -------------------------------------------------------------------

            pair_activations, single_activations = pairformer_stack(
                (pair_activations, single_activations))

            output = {
                'single': single_activations,
                'pair': pair_activations,
                'target_feat': target_feat,
            }
        return output


class TrunkDistogram(af_model.Model):
    """Trunk-only forward returning the distogram head outputs (with logits).

    Subclasses af_model.Model (name='diffuser') so the evoformer/distogram
    parameter scopes match af3.bin. Skips diffusion + confidence entirely.

    __call__(batch_dict, pseudo_full):
      - batch_dict: the raw featurised dict (host arrays already on device).
      - pseudo_full: [N, WIDTH] soft/pseudo sequence over ALL tokens. Design
        rows are injected into aatype/profile/MSA query-row features.
    Returns the DistogramHead dict {bin_edges, contact_probs, distogram}.
    """

    def __init__(self, config, remat_cfg, design_mask, soft_msa_query=True,
                 num_recycles=0, name='diffuser'):
        super().__init__(config, name=name)
        self._remat_cfg = remat_cfg
        self._design_mask = design_mask
        self._soft_msa_query = soft_msa_query
        self._num_recycles = int(num_recycles)

    def __call__(self, batch, pseudo_full, key=None):
        if key is None:
            key = hk.next_rng_key()
        batch = feat_batch.Batch.from_data_dict(batch)
        evo_cfg = self.config.evoformer
        gconf = self.global_config
        dtype = jnp.bfloat16 if gconf.bfloat16 == 'all' else jnp.float32

        with utils.bfloat16_context():
            aatype_oh = jax.nn.one_hot(batch.token_features.aatype, WIDTH)
            # inject soft sequence into aatype + profile at design tokens
            new_aatype, new_profile = featmod.inject_soft_sequence(
                aatype_oh, batch.msa.profile, pseudo_full, self._design_mask)
            tf = [new_aatype, new_profile, batch.msa.deletion_mean[..., None]]
            target_feat = jnp.concatenate(tf, axis=-1).astype(dtype)
            enc = atom_cross_attention.atom_cross_att_encoder(
                token_atoms_act=None, trunk_single_cond=None, trunk_pair_cond=None,
                config=evo_cfg.per_atom_conditioning, global_config=gconf,
                batch=batch, name='evoformer_conditioning')
            target_feat = jnp.concatenate(
                [target_feat, enc.token_act], axis=-1).astype(dtype)

        embedding_module = DesignEvoformer(
            evo_cfg, gconf, self._remat_cfg, pseudo_full, self._design_mask,
            self._soft_msa_query)
        num_res = batch.num_res
        prev = {
            'pair': jnp.zeros([num_res, num_res, evo_cfg.pair_channel], jnp.float32),
            'single': jnp.zeros([num_res, evo_cfg.seq_channel], jnp.float32),
            'target_feat': target_feat,
        }
        def recycle_body(prev_embeddings, rng_key):
            rng_key, subkey = jax.random.split(rng_key)
            embeddings = embedding_module(
                batch=batch, prev=prev_embeddings, target_feat=target_feat, key=subkey)
            embeddings = {
                'pair': embeddings['pair'].astype(jnp.float32),
                'single': embeddings['single'].astype(jnp.float32),
            }
            return embeddings, rng_key

        if hk.running_init():
            emb, _ = recycle_body(prev, key)
        else:
            # Semantic reproduction of ColabDesign recycle_mode="last": recycle
            # passes update prev without gradient, then the final pass carries
            # the gradient to the sequence logits.
            emb, rng_key = prev, key
            for _ in range(self._num_recycles):
                emb = {k: jax.lax.stop_gradient(v) for k, v in emb.items()}
                emb, rng_key = recycle_body(emb, rng_key)
            emb = {k: jax.lax.stop_gradient(v) for k, v in emb.items()}
            emb, _ = recycle_body(emb, rng_key)

        dgram = distogram_head.DistogramHead(self.config.heads.distogram, gconf)(
            batch, emb, return_distogram=True)
        return dgram

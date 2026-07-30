"""AF3DesignModel: differentiable AF3 trunk facade for Hallucination research.

This is the BindCraft/ColabDesign-shaped facade over the AF3 trunk. It is a
wires together config / sequence / masks / losses / features / remat. Importing
this runner requires JAX; full forward/optimize also needs an independently
installed AlphaFold 3 runtime and parameters.

Interface mirrors ColabDesign mk_afdesign_model(protocol="binder") (note 006
sections 4.2, 6.5):

    m = AF3DesignModel(cfg)
    m.prep_inputs(fold_input_json, binder_token_range, hotspot=...)
    m.set_sequence(length=..., omit_aa=...)        # init design logits
    loss, aux = m.loss(m.forward(m.logits))        # one differentiable eval
    m.optimize(stage)                              # optax loop over a stage

V0 implements the differentiable `soft`/`hard` path plus a gradient-free
semigreedy prototype. Semigreedy remains experiment-only and scores hard
candidates through the official predict() wrapper.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from . import losses as lossmod
from . import masks as maskmod
from . import sequence as seqmod
from .config import BINDCRAFT_4STAGE, DesignConfig, StageConfig

# AF3/Haiku/Optax imports are deferred to call time. JAX is still needed by
# sequence/masks/losses at module import time; package-level __init__ keeps only
# config import-safe on machines without JAX.


@dataclasses.dataclass
class DesignState:
    """Mutable state of a design run (ColabDesign self._params/_inputs/_tmp)."""

    logits: Any = None          # [L_design, width] optimisation variable
    bias: Any = None            # [width] banned-aa bias
    opt: Any = None             # seqmod.SeqOpt for the current stage
    batch: Any = None           # fixed featurised AF3 batch (host arrays)
    masks: Any = None           # maskmod.DesignMasks
    model_params: Any = None    # frozen AF3 haiku params
    residue_index: Any = None   # [N] token residue_index for seqsep/offset
    binder_range: Any = None    # (start, end) legacy contiguous range (compat field)
    design_indices: Any = None  # [L] sorted design token indices (scatter target)
    fold_input_path: Any = None # source fold input (predict() re-featurises at predict_bucket)
    prep_spec: Any = None       # mask spec used in prep_inputs (predict() rebuilds masks)
    model_dir: Any = None       # af3.bin dir (predict()'s ModelRunner loads weights)
    best: dict = dataclasses.field(default_factory=dict)
    traj: list = dataclasses.field(default_factory=list)
    global_step: int = 0


class AF3DesignModel:
    """BindCraft-shaped facade over the AF3 trunk (prototype)."""

    def __init__(self, cfg: DesignConfig | None = None, *, loss_fn=None):
        self.cfg = cfg or DesignConfig()
        self.state = DesignState()
        self._custom_loss_fn = loss_fn
        self._forward_apply = None   # cached hk.transform(...).apply
        self._grad_fn = None         # cached jit(value_and_grad)

    # ---- prep_inputs -----------------------------------------------------
    def prep_inputs(self, fold_input_path: str,
                    binder_token_range: tuple[int, int] | None = None,
                    hotspot_token_indices: tuple[int, ...] = (), *,
                    binder_asym_id: int | None = None,
                    design_token_indices=None,
                    target_token_indices=None,
                    ligand_target: bool = False) -> None:
        """Featurise a fixed AF3 input and build design/target/binder masks.

        Reuses the experiment featurisation path (diff_design_loop.py:98-102):
        load folding_input -> featurise_input(buckets=[cfg.bucket]) -> strip
        invalid feats -> to device arrays. Then build masks via masks.build_masks.

        Binder source (exactly one): legacy contiguous `binder_token_range`, a
        protein `binder_asym_id` (chain id), or rely on build_masks' explicit
        mask. `design_token_indices` selects non-contiguous design positions
        (e.g. antibody CDR segments via masks.cdr_design_indices); default is the
        whole binder. `ligand_target=True` makes the target the is_ligand tokens
        (small-molecule target); `target_token_indices` gives an explicit target.

        The imports below require `alphafold3` on `PYTHONPATH`; the public core
        package remains importable without that optional runtime.
        """
        import pathlib

        import jax  # noqa: F401  (deferred AF3-coupled import)
        import jax.numpy as jnp
        import numpy as np
        from alphafold3.common import folding_input
        from alphafold3.constants import chemical_components
        from alphafold3.data import featurisation
        from alphafold3.model.components import utils

        fi = list(folding_input.load_fold_inputs_from_path(pathlib.Path(fold_input_path)))[0]
        batch = featurisation.featurise_input(
            fold_input=fi, ccd=chemical_components.Ccd(), buckets=[self.cfg.bucket])[0]
        batch = utils.remove_invalidly_typed_feats(batch)
        batch = jax.tree_util.tree_map(jnp.asarray, batch)

        # validation/prep_inputs_field_probe confirmed the raw featurised dict
        # carries top-level "seq_mask" (shape [bucket], sum == #valid tokens) and
        # "residue_index" (per-chain, NOT global arange); no Batch.from_data_dict
        # needed here. profile is [bucket, 31] == sequence.AATYPE_WIDTH. is_ligand
        # / asym_id are per-token TokenFeatures (note 012 section 1).
        seq_mask = jnp.asarray(batch["seq_mask"]) if "seq_mask" in batch else None
        asym_id = jnp.asarray(batch["asym_id"]) if "asym_id" in batch else None
        is_ligand = jnp.asarray(batch["is_ligand"]) if "is_ligand" in batch else None

        if ligand_target and target_token_indices is None:
            if is_ligand is None:
                raise ValueError(
                    "prep_inputs: ligand_target=True needs is_ligand in the batch.")
            target_token_indices = np.flatnonzero(np.asarray(is_ligand).astype(bool))

        self.state.batch = batch
        self.state.residue_index = jnp.asarray(batch["residue_index"])
        self.state.binder_range = binder_token_range  # legacy compat field
        self.state.masks = maskmod.build_masks(
            seq_mask=seq_mask, binder_token_range=binder_token_range,
            hotspot_token_indices=hotspot_token_indices,
            binder_asym_id=binder_asym_id, asym_id=asym_id, is_ligand=is_ligand,
            design_token_indices=design_token_indices,
            target_token_indices=target_token_indices)
        self.state.design_indices = self.state.masks.design_indices
        # predict() re-featurises this fold input at predict_bucket and rebuilds
        # masks from the same spec (it must NOT reuse the design-loop bucket batch).
        self.state.fold_input_path = fold_input_path
        self.state.prep_spec = {
            "binder_token_range": binder_token_range,
            "hotspot_token_indices": hotspot_token_indices,
            "binder_asym_id": binder_asym_id,
            "design_token_indices": (None if design_token_indices is None
                                     else [int(x) for x in np.asarray(design_token_indices)]),
            "target_token_indices": (None if target_token_indices is None
                                     else [int(x) for x in np.asarray(target_token_indices)]),
        }
        # Any existing predict() context was built from a previous fold input or
        # mask spec; invalidate it so validation/scoring cannot use stale masks.
        self._predict_ctx = None
        self._predict_ctx_key = None

    # ---- set_sequence ----------------------------------------------------
    def set_sequence(self, length: int | None = None,
                     omit_aa_indices: tuple[int, ...] = (),
                     init_scale: float = 0.0, seed: int | None = None) -> None:
        """Initialise the design logits + banned-aa bias (ColabDesign set_seq).

        `length` is the number of design tokens. When None it defaults to the
        number of design positions (`len(design_indices)`), which is the only
        consistent choice for non-contiguous design sets; if given explicitly it
        must equal `len(design_indices)`. Non-saturated init by default
        (init_scale=0) per af3-bc-backprop-parity.
        """
        import jax
        if self.state.design_indices is None:
            raise RuntimeError("set_sequence(): call prep_inputs() first.")
        n_design = int(self.state.design_indices.shape[0])
        if length is None:
            length = n_design
        elif int(length) != n_design:
            raise ValueError(
                f"set_sequence: length={length} != number of design tokens "
                f"{n_design} (len(design_indices)).")
        key = jax.random.PRNGKey(self.cfg.seed if seed is None else seed)
        width = seqmod.AATYPE_WIDTH
        self.state.logits = seqmod.init_logits(length, width, init_scale, key)
        self.state.bias = seqmod.aa_bias(width, omit_aa_indices)

    # ---- weights + compiled forward --------------------------------------
    def load_params(self, model_dir: str) -> None:
        """Load frozen AF3 trunk weights (af3.bin) once.

        model_dir is the directory holding af3.bin (e.g. ~/alphafold3/models).
        Only the trunk + distogram params are used; diffusion/confidence params
        are loaded but never invoked.
        """
        import pathlib

        from alphafold3.model import params as af_params
        self.state.model_dir = str(pathlib.Path(model_dir).expanduser())
        # Changing model_dir also changes the official ModelRunner used by
        # predict(); force a fresh predict context on the next call.
        self._predict_ctx = None
        self._predict_ctx_key = None
        self.state.model_params = af_params.get_model_haiku_params(
            model_dir=pathlib.Path(model_dir).expanduser())

    def _model_config(self):
        """AF3 model config with official recycles off; distogram requested.

        Uses run_alphafold.make_model_config (needs ~/alphafold3 on PYTHONPATH).
        return_distogram is irrelevant here because TrunkDistogram calls the head
        with return_distogram=True directly. The design-loop recycle count is
        implemented in TrunkDistogram so it can use ColabDesign-style
        recycle_mode="last" gradient semantics.
        """
        import run_alphafold as ra
        return ra.make_model_config(num_recycles=0, return_distogram=True)

    def _ensure_forward(self):
        """Lazily build the hk.transform'd trunk forward and cache its apply.

        The transform closes over the (static) model config, remat config and
        design_mask; the differentiable inputs (batch, pseudo_full) are apply
        arguments so jax.value_and_grad/jit see them explicitly.
        """
        if self._forward_apply is not None:
            return
        import haiku as hk

        from . import trunk as trunkmod

        model_cfg = self._model_config()
        remat_cfg = self.cfg.remat
        design_mask = self.state.masks.design_mask
        soft_msa_query = self.cfg.soft_msa_query
        design_num_recycles = self.cfg.design_num_recycles

        @hk.transform
        def _fwd(batch, pseudo_full):
            return trunkmod.TrunkDistogram(
                model_cfg, remat_cfg, design_mask, soft_msa_query,
                num_recycles=design_num_recycles)(
                    batch, pseudo_full)

        self._forward_apply = _fwd.apply

    def _pseudo_full(self, logits, opt):
        """Scatter per-design soft sequence into a full [N, WIDTH] array.

        Soft rows are scattered to `design_indices` (sorted token indices, which
        may be non-contiguous, e.g. antibody CDR segments). logits row i maps to
        design_indices[i]. Rows outside the design set stay zero and are ignored
        by inject_soft_sequence, which only writes design_mask rows; non-design
        binder positions (e.g. antibody framework) keep their featurised aatype.
        """
        import jax.numpy as jnp
        soft = seqmod.soft_seq(logits, opt, self.state.bias)["pseudo"]  # [L, W]
        n = self.state.residue_index.shape[0]
        width = seqmod.AATYPE_WIDTH
        base = jnp.zeros((n, width), dtype=soft.dtype)
        return base.at[self.state.design_indices].set(soft)

    # ---- forward ---------------------------------------------------------
    def forward(self, logits, batch=None, opt: seqmod.SeqOpt | None = None,
                key=None):
        """Differentiable trunk-only forward returning DistogramHead outputs.

        Pipeline (note 006 section 6.4):
          soft_seq(logits) -> inject into target_feat aatype + profile
          -> DesignEvoformer trunk (LOCAL block_remat via remat.maybe_remat)
          -> DistogramHead(return_distogram=True) -> {bin_edges, contact_probs,
             distogram}

        batch defaults to the prepped batch; opt defaults to state.opt. Requires
        load_params() to have been called.
        """
        import jax
        if self.state.model_params is None:
            raise RuntimeError("forward(): call load_params(model_dir) first.")
        self._ensure_forward()
        opt = opt if opt is not None else self.state.opt
        if opt is None:
            opt = seqmod.SeqOpt(soft=1.0, temp=1.0)
        batch = batch if batch is not None else self.state.batch
        if key is None:
            key = jax.random.PRNGKey(self.cfg.seed)
        pseudo_full = self._pseudo_full(logits, opt)
        return self._forward_apply(self.state.model_params, key, batch, pseudo_full)

    # ---- loss ------------------------------------------------------------
    def loss(self, distogram) -> tuple[Any, dict]:
        """Weighted V0 contact objective from distogram outputs (losses.total_loss)."""
        if self._custom_loss_fn is not None:
            return self._custom_loss_fn(
                distogram,
                self.state.residue_index,
                self.state.masks,
                self.cfg,
            )
        return lossmod.total_loss(
            distogram, self.state.residue_index, self.state.masks, self.cfg)

    def _loss_from_logits(self, logits, batch):
        """value_and_grad target: logits -> distogram -> scalar loss.

        Keep `batch` an explicit arg, never a jit closure: the diff-loop run
        found a closed-over batch gets folded to all-zero contact_probs.
        """
        opt = self.state.opt
        # forward returns the DistogramHead dict {bin_edges, contact_probs,
        # distogram}; total_loss consumes that dict whole (reads ["distogram"]
        # logits + ["bin_edges"]).
        head = self.forward(logits, batch=batch, opt=opt)
        loss, aux = self.loss(head)
        soft = seqmod.soft_seq(logits, opt, self.state.bias)
        if self.cfg.weights.seq_ent:
            aux["seq_ent"] = lossmod.seq_ent_loss(soft["pssm"])
            loss = loss + self.cfg.weights.seq_ent * aux["seq_ent"]
        return loss, aux

    def _loss_from_logits_dyn(self, logits, batch, soft, temp, hard):
        """value_and_grad target with soft/temp/hard as DYNAMIC jit args.

        optimize() ramps soft/temp/hard every step. If they were read from
        self.state.opt (Python floats) the jitted grad fn would bake the first
        step's values as constants. Passing them as traced scalars keeps one
        compiled grad fn for the whole stage. alpha is static (cfg.alpha).
        """
        opt = seqmod.SeqOpt(soft=soft, temp=temp, hard=hard, alpha=self.cfg.alpha)
        head = self.forward(logits, batch=batch, opt=opt)
        loss, aux = self.loss(head)
        if self.cfg.weights.seq_ent:
            sq = seqmod.soft_seq(logits, opt, self.state.bias)
            aux["seq_ent"] = lossmod.seq_ent_loss(sq["pssm"])
            loss = loss + self.cfg.weights.seq_ent * aux["seq_ent"]
        return loss, aux

    # ---- optimize --------------------------------------------------------
    # ColabDesign optimizer registry (af/model.py set_optimizer). SGD is the
    # BindCraft-parity default; adam etc. are optional but NOT parity.
    _OPTAX = {"sgd": "sgd", "adam": "adam", "adabelief": "adabelief",
              "rmsprop": "rmsprop"}

    def optimize(self, stage: StageConfig, *, step_callback=None, should_stop=None) -> list[dict]:
        """Run one BindCraft/ColabDesign stage over the design logits.

        Semantic reproduction of ColabDesign af/design.py:design()+step():
          - per-step ramp of soft/hard/step (linear) and temp (quadratic decay),
          - lr_scale = step * ((1-soft) + soft*temp),
          - optional norm_seq_grad before the optimizer (cfg.norm_seq_grad),
          - optimizer built with lr=1.0 (cfg.optimizer, default SGD); the real
            lr = cfg.learning_rate * lr_scale is applied manually:
            params -= lr * grad  with grad = -optax_updates  (==  params += lr*updates).

        V0 executes differentiable logits/soft/hard stages. "semigreedy" raises.
        """
        if stage.name == "semigreedy":
            raise NotImplementedError(
                "semigreedy stage: gradient-free search, not in soft-gradient V0.")

        import jax
        import jax.numpy as jnp
        import optax

        from .config import ramp_value

        cfg = self.cfg
        optax_ctor = {"sgd": optax.sgd, "adam": optax.adam,
                      "adabelief": optax.adabelief, "rmsprop": optax.rmsprop}
        optimizer = optax_ctor[cfg.optimizer](1.0)  # lr baked to 1.0; scaled below

        logits = self.state.logits
        opt_state = optimizer.init(logits)
        base_lr = stage.learning_rate if stage.learning_rate is not None else cfg.learning_rate

        # ramp endpoints (None -> constant = start)
        e_soft = stage.soft if stage.e_soft is None else stage.e_soft
        e_temp = stage.temp if stage.e_temp is None else stage.e_temp
        e_hard = stage.hard if stage.e_hard is None else stage.e_hard
        e_step = stage.step if stage.e_step is None else stage.e_step

        # one compiled grad fn for the whole stage; soft/temp/hard are dynamic args
        grad_fn = jax.jit(jax.value_and_grad(self._loss_from_logits_dyn,
                                             argnums=0, has_aux=True))
        n = stage.iterations
        traj = []
        best_logits = None
        best_loss = None
        for i in range(n):
            soft = ramp_value(stage.soft, e_soft, i, n)
            temp = ramp_value(stage.temp, e_temp, i, n, quadratic=True)
            hard = ramp_value(stage.hard, e_hard, i, n)
            step_v = ramp_value(stage.step, e_step, i, n)
            eval_logits = logits
            (loss, aux), grad = grad_fn(
                logits, self.state.batch,
                jnp.float32(soft), jnp.float32(temp), jnp.float32(hard))
            loss_f = float(loss)
            if best_loss is None or loss_f < best_loss:
                best_loss = loss_f
                best_logits = eval_logits
            if cfg.norm_seq_grad:
                grad = seqmod.norm_seq_grad(grad)
            updates, opt_state = optimizer.update(grad, opt_state, logits)
            lr = base_lr * (step_v * ((1.0 - soft) + soft * temp))
            # ColabDesign: grad_ret = -updates; params -= lr*grad_ret == params += lr*updates
            logits = logits + lr * updates
            row = {"stage": stage.name, "step": i,
                         "global_step": self.state.global_step,
                         "soft": float(soft), "temp": float(temp),
                         "hard": float(hard), "lr": float(lr), "loss": loss_f,
                         **{k: float(v) for k, v in aux.items()}}
            traj.append(row)
            if step_callback is not None:
                step_callback(dict(row), eval_logits)
            self.state.global_step += 1
            if should_stop is not None and should_stop(dict(row)):
                logits = eval_logits
                break
        self.state.logits = logits
        self.state.best = {
            "stage": stage.name,
            "loss": best_loss,
            "logits": best_logits,
        }
        # keep state.opt in sync with the final ramp values for any later forward()
        self.state.opt = seqmod.SeqOpt(soft=float(traj[-1]["soft"]) if traj else stage.soft,
                                       temp=float(traj[-1]["temp"]) if traj else stage.temp,
                                       hard=float(traj[-1]["hard"]) if traj else stage.hard,
                                       alpha=cfg.alpha)
        self.state.traj.extend(traj)
        return traj

    def run_schedule(
        self,
        stages: tuple[StageConfig, ...] = BINDCRAFT_4STAGE,
        *,
        step_callback=None,
        should_stop=None,
    ) -> list[dict]:
        """Run a sequence of stages (BindCraft 4-stage by default).

        V0 executes differentiable logits/soft/hard stages and stops at the
        first unimplemented semigreedy stage rather than failing the schedule.
        """
        out = []
        for stage in stages:
            try:
                out.extend(
                    self.optimize(
                        stage,
                        step_callback=step_callback,
                        should_stop=should_stop,
                    )
                )
            except NotImplementedError:
                break
            if should_stop is not None and out and should_stop(dict(out[-1])):
                break
        return out

    # ---- ColabDesign-style stage convenience methods ---------------------
    # Mirror ColabDesign af/design.py design_logits/soft/hard signatures so the
    # BindCraft 4-stage flow can be expressed call-for-call. semigreedy is a
    # declared placeholder (gradient-free, V1).
    def design_logits(self, iters: int = 50, e_soft: float | None = None,
                      learning_rate: float | None = None) -> list[dict]:
        """ColabDesign design_logits: ramp soft 0 -> e_soft (default constant 0)."""
        return self.optimize(StageConfig(
            name="logits", iterations=iters, soft=0.0, e_soft=e_soft,
            temp=1.0, hard=0.0,
            learning_rate=self.cfg.learning_rate if learning_rate is None else learning_rate))

    def design_soft(self, iters: int = 45, temp: float = 1.0,
                    e_temp: float | None = 1e-2,
                    learning_rate: float | None = None) -> list[dict]:
        """ColabDesign design_soft: soft=1, temp ramps temp -> e_temp (quadratic)."""
        return self.optimize(StageConfig(
            name="soft", iterations=iters, soft=1.0, temp=temp, e_temp=e_temp,
            hard=0.0,
            learning_rate=self.cfg.learning_rate if learning_rate is None else learning_rate))

    def design_hard(self, iters: int = 5, temp: float = 1e-2,
                    learning_rate: float | None = None) -> list[dict]:
        """ColabDesign design_hard: soft=1, hard=1, fixed low temp, dropout off."""
        return self.optimize(StageConfig(
            name="hard", iterations=iters, soft=1.0, temp=temp, hard=1.0,
            dropout=False,
            learning_rate=self.cfg.learning_rate if learning_rate is None else learning_rate))

    def design_semigreedy(self, *, iters: int = 15, tries: int | None = None,
                          e_tries: int | None = None, seq_logits=None,
                          seq_design_start=None, policy: str | None = None,
                          model_dir=None, mutation_rate: int = 1, seed: int | None = None,
                          predict_kwargs: dict | None = None) -> dict:
        """ColabDesign design_semigreedy (gradient-free) THIN DELEGATE (note 018).

        Hill-climbs the hard binder sequence by scoring single-point mutants with
        the gradient-free predict() path. Does NOT call forward()/_loss_from_logits()
        /optimize() or the DesignEvoformer soft trunk; the loop / best-of-tries /
        save_best / trajectory live in semigreedy.run_semigreedy. Each candidate is
        scored by self.predict(seq_indices=cand)["scoring_loss"]; position prior uses
        predict(...)["per_token_plddt"]["design"]/100 (NOT scalar binder_plddt).

        Start: seq_design_start (int[L]) if given, else state.logits.argmax(-1) (the
        hard-stage end sequence). Raises if neither is available (no silent guess).
        tries defaults to BindCraft greedy_tries = ceil(L * 1/100). seq_logits=0 is
        the BindCraft 4-stage (soft_iters=0) path; PSSM soft_iters>0 is not in V1.
        Writes the best hard sequence back as a saturated one-hot in state.logits
        and records state.best / state.traj (see validation/README semigreedy notes).
        That write-back is for terminal hard output / predict(); do not run a later
        gradient stage from these saturated logits without reinitialising logits.
        """
        import math

        import numpy as np

        from . import semigreedy as sgmod

        if self.state.design_indices is None:
            raise RuntimeError("design_semigreedy(): call prep_inputs() first.")
        design_indices = np.asarray(self.state.design_indices)
        L = int(design_indices.shape[0])

        if seq_design_start is not None:
            seq_start = np.asarray(seq_design_start, dtype=int)
        elif self.state.logits is not None:
            import jax.numpy as jnp
            # hard-stage end: argmax of the design logits over the design positions.
            seq_start = np.asarray(jnp.argmax(self.state.logits, axis=-1)).astype(int)
        else:
            raise RuntimeError(
                "design_semigreedy(): no starting sequence; pass seq_design_start or "
                "run a hard stage (set_sequence()/optimize()) first. Refusing to guess.")
        bad = seq_start[(seq_start < 0) | (seq_start >= sgmod.PROTEIN_ALPHABET_SIZE)]
        if bad.size:
            raise ValueError(
                "design_semigreedy(): starting sequence contains non-protein aa ids "
                f"{bad[:5].tolist()}; semigreedy uses the protein alphabet [0, "
                f"{sgmod.PROTEIN_ALPHABET_SIZE}).")

        policy = policy if policy is not None else self.cfg.predict_policy
        tries = tries if tries is not None else max(1, math.ceil(L * 1 / 100))
        pk = dict(predict_kwargs or {})

        def predict_fn(cand):
            return self.predict(seq_indices=np.asarray(cand).astype(int), policy=policy,
                                model_dir=model_dir, **pk)

        rng = np.random.RandomState(self.cfg.seed if seed is None else seed)
        aa_bias = None if self.state.bias is None else np.asarray(self.state.bias)
        out = sgmod.run_semigreedy(
            predict_fn, design_indices=design_indices, seq_design_start=seq_start,
            iters=iters, tries=tries, e_tries=e_tries, seq_logits=seq_logits,
            aa_bias=aa_bias, mutation_rate=mutation_rate, rng=rng)

        # Write best hard sequence back as a saturated one-hot so subsequent
        # predict()/output uses it. Semigreedy is terminal in BindCraft's 4-stage
        # schedule; reinitialise logits before any later gradient stage.
        best_ids = np.asarray(out["best_seq_design"], dtype=int)
        width = seqmod.AATYPE_WIDTH
        onehot = np.full((L, width), -1e4, dtype=np.float32)
        onehot[np.arange(L), best_ids] = 1e4
        import jax.numpy as jnp
        self.state.logits = jnp.asarray(onehot)
        self.state.best = {"seq_design": out["best_seq_design"],
                           "scoring_loss": out["best_scoring_loss"]}
        self.state.traj.extend(out["trajectory"])
        return out

    # ---- validation / scoring (gradient-free official path) --------------
    def predict(self, *, seq_indices=None, policy: str | None = None,
                num_samples: int | None = None, diffusion_steps: int | None = None,
                num_recycles: int | None = None,
                bucket: int | None = None, flash_attention: str | None = None,
                seed: int | None = None, key=None, weights=None,
                binder_chain_id=None, model_dir=None, return_raw: bool = False,
                return_arrays: bool = False) -> dict:
        """Gradient-free official-AF3 predict for ONE hard candidate (note 017).

        THIN DELEGATE to af3design.predict. This is the BindCraft predict(
        backprop=False) analogue: it runs the OFFICIAL ModelRunner path (trunk +
        diffusion + ConfidenceHead + ranking) and assembles con/i_con (distogram)
        + atom-masked pLDDT + masked PAE/i_PAE + iptm + scoring_loss. It does NOT
        call forward()/_loss_from_logits()/optimize() or the DesignEvoformer soft
        trunk; it shares only masks + config + af3.bin with the differentiable
        loop. `ranking_score` is report-only (never in scoring_loss); `iptm` is a
        forward scalar only (never a gradient mixed-loss term).

        seq_indices: int[L] hard candidate aa ids on the design positions; None ->
          current hard sequence state.logits.argmax(-1). policy/bucket/steps etc.
          default to DesignConfig.predict_*.
        """
        import numpy as np

        from . import predict as predmod
        cfg = self.cfg
        if self.state.fold_input_path is None or self.state.prep_spec is None:
            raise RuntimeError("predict(): call prep_inputs() first.")
        mdir = model_dir if model_dir is not None else self.state.model_dir
        if mdir is None:
            raise RuntimeError("predict(): model_dir required (pass model_dir= or call load_params()).")
        policy = policy if policy is not None else cfg.predict_policy
        bucket = bucket if bucket is not None else cfg.predict_bucket
        diffusion_steps = diffusion_steps if diffusion_steps is not None else cfg.predict_diffusion_steps
        num_samples = num_samples if num_samples is not None else cfg.predict_num_samples
        num_recycles = num_recycles if num_recycles is not None else cfg.predict_num_recycles
        flash_attention = flash_attention if flash_attention is not None else cfg.predict_flash_attention
        seed = cfg.seed if seed is None else seed
        if seq_indices is None and self.state.logits is not None:
            import jax.numpy as jnp
            logits = self.state.logits
            if self.state.bias is not None:
                logits = logits + self.state.bias
            seq_indices = np.asarray(jnp.argmax(logits, axis=-1)).astype(np.int32)

        def _freeze(v):
            if v is None or isinstance(v, (str, int, float, bool)):
                return v
            if isinstance(v, dict):
                return tuple(sorted((k, _freeze(val)) for k, val in v.items()))
            if isinstance(v, (list, tuple)):
                return tuple(_freeze(x) for x in v)
            try:
                arr = np.asarray(v)
                return tuple(arr.reshape(-1).tolist())
            except Exception:  # noqa: BLE001
                return repr(v)
        prep_key = _freeze(self.state.prep_spec)
        # cache key MUST include num_recycles: scorer (recycle 0) and final
        # validation (recycle 3) share a model but need distinct compiled contexts.
        ckey = (str(mdir), self.state.fold_input_path, prep_key,
                bucket, diffusion_steps, num_samples, num_recycles, flash_attention)
        if getattr(self, "_predict_ctx", None) is None or self._predict_ctx_key != ckey:
            self._predict_ctx = predmod.build_context(
                fold_input_path=self.state.fold_input_path, prep_spec=self.state.prep_spec,
                design_cfg=cfg, model_dir=mdir, bucket=bucket,
                diffusion_steps=diffusion_steps, num_samples=num_samples,
                num_recycles=num_recycles, flash_attention=flash_attention)
            self._predict_ctx_key = ckey
        return predmod.predict_candidate(
            self._predict_ctx, seq_indices=seq_indices, policy=policy,
            binder_chain_id=binder_chain_id, seed=seed, key=key, weights=weights,
            return_raw=return_raw, return_arrays=return_arrays)

    def validate(self, *, seq_indices=None, policy: str | None = None,
                 filters: dict | None = None, model_dir=None, seed: int | None = None,
                 binder_chain_id=None,
                 return_raw: bool = False) -> dict:
        """Final-validation predict + BindCraft confidence gate/filter (THIN wrapper).

        Semantic reproduction of BindCraft's design-vs-validation split (note
        recycle/dropout/model-sampling spec): the cheap semigreedy scorer ranks
        candidates at the predict_* config (steps50/recycle0), but the final
        confidence FILTER thresholds must be applied to a HIGHER-FIDELITY predict.
        validate() calls self.predict() with the validate_* config
        (validate_diffusion_steps / validate_num_recycles / validate_num_samples;
        bucket + policy reuse predict_*) and then applies
        validation_filters.final_trajectory_gate + apply_confidence_filters to that
        OUTPUT (never to a scorer output).

        Returns {predict, gate, filters, pass}. `pass` = gate did not terminate AND
        the confidence filters passed. `ranking_score` is report-only (it is never
        read by the gate or the filters).
        """
        from . import validation_filters as vfmod
        cfg = self.cfg
        predict_out = self.predict(
            seq_indices=seq_indices,
            policy=policy if policy is not None else cfg.predict_policy,
            diffusion_steps=cfg.validate_diffusion_steps,
            num_recycles=cfg.validate_num_recycles,
            num_samples=cfg.validate_num_samples,
            binder_chain_id=binder_chain_id,
            model_dir=model_dir, seed=seed, return_raw=return_raw)
        gate = vfmod.final_trajectory_gate(predict_out)
        filt = vfmod.apply_confidence_filters(predict_out, filters)
        overall = bool((gate["terminate"] is False) and filt["pass"])
        return {"predict": predict_out, "gate": gate, "filters": filt, "pass": overall}

    def save_pdb(self, *args, **kwargs):
        raise NotImplementedError("save_pdb(): depends on diffusion sample (V1).")

"""Public AF3/JAX Hallucination engine with exact checkpoint callbacks."""

from __future__ import annotations

import dataclasses
import hashlib
import math
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..config import HallucinationSpec, LossSpec, StageSpec, validate_stopper
from ..errors import ConfigurationError
from .config import ContactSpec, DesignConfig, LossWeights, RematConfig, StageConfig

AF3_AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
AA_INDEX = {aa: index for index, aa in enumerate(AF3_AA_ORDER)}


def _contact_spec(loss: LossSpec, defaults: ContactSpec) -> ContactSpec:
    params = dict(loss.parameters)
    allowed = {"cutoff", "num", "num_pos", "binary", "seqsep"}
    localization = {"framework_contact_loss", "framework_contact_offset"}
    unknown = set(params) - allowed - localization
    if unknown:
        raise ConfigurationError(f"unknown {loss.type} parameters: {sorted(unknown)}")
    values = dataclasses.asdict(defaults)
    values.update({key: value for key, value in params.items() if key in allowed})
    return ContactSpec(**values)


def design_config_from_public(spec: HallucinationSpec, *, remat: bool = True) -> DesignConfig:
    """Translate public loss/stage knobs into the reviewed AF3 design config."""

    weights = {
        "con": 0.0,
        "i_con": 0.0,
        "helix": 0.0,
        "seq_ent": 0.0,
        "glob": 0.0,
        "framework_contact_loss": False,
        "framework_contact_offset": 1.0,
    }
    con = ContactSpec(cutoff=14.0, num=2, seqsep=9, binary=False)
    i_con = ContactSpec(cutoff=20.0, num=2, seqsep=0, binary=False)
    mapping = {
        "intra_contact": "con",
        "interface_contact": "i_con",
        "helix": "helix",
        "sequence_entropy": "seq_ent",
        "globularity": "glob",
    }
    for loss in spec.losses:
        if loss.type not in mapping:
            raise ConfigurationError(
                f"AF3 built-in backend does not know loss {loss.type!r}; "
                "provide a custom loss_fn through the Python API"
            )
        field = mapping[loss.type]
        weights[field] = loss.weight
        if loss.type == "intra_contact":
            con = _contact_spec(loss, con)
        elif loss.type == "interface_contact":
            i_con = _contact_spec(loss, i_con)
            if "framework_contact_loss" in loss.parameters:
                weights["framework_contact_loss"] = bool(loss.parameters["framework_contact_loss"])
            if "framework_contact_offset" in loss.parameters:
                weights["framework_contact_offset"] = float(loss.parameters["framework_contact_offset"])
    return DesignConfig(
        con_spec=con,
        i_con_spec=i_con,
        weights=LossWeights(**weights),
        remat=RematConfig(enabled=remat),
        bucket=spec.bucket,
        seed=spec.seed,
        optimizer=spec.optimizer,
        learning_rate=spec.learning_rate,
        alpha=spec.alpha,
        design_num_recycles=spec.design_recycles,
    )


def stage_config_from_public(stage: StageSpec, default_lr: float) -> StageConfig:
    if stage.soft_start is None or stage.soft_end is None:
        raise ConfigurationError("stage soft ramp endpoints are missing")
    if stage.temperature_start is None or stage.temperature_end is None:
        raise ConfigurationError("stage temperature ramp endpoints are missing")
    if stage.hard_start is None or stage.hard_end is None:
        raise ConfigurationError("stage hard ramp endpoints are missing")
    return StageConfig(
        name=stage.type,
        iterations=stage.steps,
        learning_rate=stage.learning_rate or default_lr,
        soft=stage.soft_start,
        e_soft=stage.soft_end,
        temp=stage.temperature_start,
        e_temp=stage.temperature_end,
        hard=stage.hard_start,
        e_hard=stage.hard_end,
        step=stage.step_start,
        e_step=stage.step_end,
        dropout=stage.dropout,
    )


@dataclasses.dataclass(frozen=True)
class PreparedDesign:
    input_json: Path
    binder_chain_id: str
    binder_asym_id: int
    binder_sequence: str
    design_token_indices: tuple[int, ...]
    hotspot_token_indices: tuple[int, ...]
    target_token_indices: tuple[int, ...] | None = None


class AF3JaxHallucinationEngine:
    """Latest hybrid-query Hallucination path used by the public backend.

    Every evaluation re-featurises the hard argmax carrier so atom chemistry is
    legal, while CDR/design rows of target aatype, MSA query, and profile receive
    the differentiable pseudo/query state.
    """

    def __init__(
        self,
        public_spec: HallucinationSpec,
        prepared: PreparedDesign,
        *,
        model_dir: str | Path,
        remat: bool = True,
        loss_fn: Callable | None = None,
    ) -> None:
        self.public_spec = public_spec
        self.cfg = design_config_from_public(public_spec, remat=remat)
        self.prepared = prepared
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.custom_loss_fn = loss_fn
        self.trajectory: list[dict[str, Any]] = []
        self.stage_summaries: list[dict[str, Any]] = []
        self.global_step = 0
        self._prepare()

    def _prepare(self) -> None:
        import haiku as hk
        import jax
        import jax.numpy as jnp
        import run_alphafold as run_af3
        from alphafold3.model import params as af_params

        from . import losses as loss_module
        from . import masks as mask_module
        from . import sequence as sequence_module
        from .hybrid import HybridDesignTrunk, binder_mask_from_batch, featurise
        from .trunk import WIDTH

        self._jax = jax
        self._jnp = jnp
        self._sequence_module = sequence_module
        self._hybrid = __import__(
            "af3_hallucination.af3.hybrid", fromlist=["candidate_input"]
        )
        self._width = WIDTH
        self.base_batch = featurise(self.prepared.input_json, self.cfg.bucket)
        valid = np.asarray(self.base_batch["seq_mask"]) > 0
        binder_mask = binder_mask_from_batch(self.base_batch, self.prepared.binder_asym_id)
        binder_indices = np.flatnonzero(binder_mask)
        if binder_indices.size != len(self.prepared.binder_sequence):
            raise ValueError("binder token count differs from binder sequence length")
        design_indices = np.asarray(self.prepared.design_token_indices, np.int32)
        if design_indices.size == 0 or len(np.unique(design_indices)) != design_indices.size:
            raise ValueError("design_token_indices must be non-empty and unique")
        if np.any(~binder_mask[design_indices]):
            raise ValueError("every design token must belong to the binder")
        local_lookup = {int(token): index for index, token in enumerate(binder_indices)}
        self.design_local_indices = np.asarray([local_lookup[int(token)] for token in design_indices], np.int32)
        self.design_indices = design_indices
        target_indices = (
            np.asarray(self.prepared.target_token_indices, np.int32)
            if self.prepared.target_token_indices is not None
            else np.flatnonzero(valid & ~binder_mask)
        )
        self.masks = mask_module.build_masks(
            seq_mask=valid,
            binder_mask=binder_mask,
            hotspot_token_indices=self.prepared.hotspot_token_indices,
            design_token_indices=design_indices,
            target_token_indices=target_indices,
            asym_id=np.asarray(self.base_batch["asym_id"]),
            is_ligand=np.asarray(self.base_batch["is_ligand"]),
            interface_mode="target_hotspot_x_design_cdr",
        )
        self.residue_index = jnp.asarray(self.base_batch["residue_index"])
        self.binder_indices = binder_indices
        self.fixed_binder_ids = np.asarray(
            [AA_INDEX[aa] for aa in self.prepared.binder_sequence], np.int32
        )
        omitted = tuple(AA_INDEX[aa] for aa in self.public_spec.omit_amino_acids)
        self.bias = sequence_module.aa_bias(20, omitted)
        key = jax.random.PRNGKey(self.public_spec.seed)
        self.logits = sequence_module.init_logits(
            len(design_indices), 20, self.public_spec.init_scale, key
        )
        model_config = run_af3.make_model_config(
            num_recycles=0,
            return_distogram=True,
            flash_attention_implementation=str(
                self.public_spec.backend_config.get("flash_attention", "xla")
            ),
        )
        design_mask = jnp.asarray(np.asarray(self.masks.design_mask), bool)

        @hk.transform
        def forward_model(batch, pseudo, profile, key_value):
            return HybridDesignTrunk(
                model_config,
                self.cfg.remat,
                design_mask,
                num_design_recycles=self.public_spec.design_recycles,
                detach_recycles=True,
            )(batch, pseudo, profile, key_value)

        self.parameters = af_params.get_model_haiku_params(model_dir=self.model_dir)
        self._forward_apply = forward_model.apply

        def objective(logits, batch, soft, temperature, hard, key_value):
            _, pseudo, profile = self._feature_state(logits, batch, soft, temperature, hard)
            distogram = self._forward_apply(
                self.parameters, key_value, batch, pseudo, profile, key_value
            )
            if self.custom_loss_fn is not None:
                total, aux = self.custom_loss_fn(
                    distogram, self.residue_index, self.masks, self.cfg
                )
            else:
                total, aux = loss_module.total_loss(
                    distogram, self.residue_index, self.masks, self.cfg
                )
            if self.cfg.weights.seq_ent:
                state = sequence_module.soft_seq(
                    logits,
                    sequence_module.SeqOpt(
                        soft=soft,
                        temp=temperature,
                        hard=hard,
                        alpha=self.public_spec.alpha,
                    ),
                    self.bias,
                )
                aux = {**aux, "seq_ent": loss_module.seq_ent_loss(state["pssm"])}
                total = total + self.cfg.weights.seq_ent * aux["seq_ent"]
            return total, aux

        self._value_and_grad = jax.jit(jax.value_and_grad(objective, has_aux=True))

    def _sequence_state(self, logits, soft, temperature, hard):
        return self._sequence_module.soft_seq(
            logits,
            self._sequence_module.SeqOpt(
                soft=soft,
                temp=temperature,
                hard=hard,
                alpha=self.public_spec.alpha,
            ),
            self.bias,
        )

    def _carrier_batch(self, logits, soft, temperature, hard, scratch: Path):
        state = self._sequence_state(logits, soft, temperature, hard)
        pseudo = np.asarray(state["pseudo"], np.float32)
        binder_ids = self.fixed_binder_ids.copy()
        binder_ids[self.design_local_indices] = np.argmax(pseudo, axis=-1)
        candidate_path = scratch / "current_carrier.json"
        self._hybrid.candidate_input(
            self.prepared.input_json,
            binder_ids,
            candidate_path,
            binder_chain=self.prepared.binder_chain_id,
        )
        batch = self._hybrid.featurise(candidate_path, self.cfg.bucket)
        binder_mask = self._hybrid.binder_mask_from_batch(batch, self.prepared.binder_asym_id)
        if not np.array_equal(np.flatnonzero(binder_mask), self.binder_indices):
            raise ValueError("hard carrier changed the token layout")
        if not np.array_equal(np.asarray(batch["aatype"])[binder_mask], binder_ids):
            raise ValueError("hard carrier atom chemistry is stale")
        return batch, binder_ids

    def _feature_state(self, logits, batch, soft, temperature, hard):
        jnp = self._jnp
        state = self._sequence_state(logits, soft, temperature, hard)
        pseudo20 = state["pseudo"]
        query20 = hard * state["hard"] + (1.0 - hard) * state["soft"]
        pseudo31 = jnp.pad(pseudo20, ((0, 0), (0, self._width - 20)))
        query31 = jnp.pad(query20, ((0, 0), (0, self._width - 20)))
        hard_ids = jnp.asarray(batch["aatype"])[self.design_indices]
        hard_onehot = self._jax.nn.one_hot(hard_ids, self._width)
        hard_profile = jnp.asarray(batch["profile"])[self.design_indices]
        query_weight = jnp.sum(hard_profile * hard_onehot, axis=-1, keepdims=True)
        expected_profile = hard_profile + query_weight * (query31 - hard_onehot)
        n_tokens = int(batch["seq_mask"].shape[0])
        pseudo_full = jnp.zeros((n_tokens, self._width), pseudo31.dtype).at[
            self.design_indices
        ].set(pseudo31)
        profile_full = jnp.zeros((n_tokens, self._width), expected_profile.dtype).at[
            self.design_indices
        ].set(expected_profile)
        return state, pseudo_full, profile_full

    def run(
        self,
        scratch_dir: str | Path,
        *,
        step_callback: Callable[[dict[str, Any], np.ndarray], None] | None = None,
        should_stop: Callable[[dict[str, Any]], bool] | None = None,
        semigreedy_scorer: Callable | None = None,
    ) -> list[dict[str, Any]]:
        import optax

        scratch = Path(scratch_dir).resolve()
        scratch.mkdir(parents=True, exist_ok=True)
        optimizer_factory = {
            "sgd": optax.sgd,
            "adam": optax.adam,
            "adabelief": optax.adabelief,
            "rmsprop": optax.rmsprop,
        }[self.public_spec.optimizer]
        for public_stage in self.public_spec.stages:
            if public_stage.steps == 0:
                continue
            if public_stage.type == "semigreedy":
                if semigreedy_scorer is None:
                    raise NotImplementedError(
                        "semigreedy requires a candidate scorer supplied through the Python API"
                    )
                from .semigreedy import run_semigreedy

                stage_name = public_stage.name or public_stage.type
                started = time.time()
                design_start = np.asarray(
                    self._jnp.argmax(self.logits + self.bias, axis=-1), np.int32
                )

                def predict_fn(design_ids):
                    values = np.asarray(design_ids, np.int32)
                    binder_ids = self.fixed_binder_ids.copy()
                    binder_ids[self.design_local_indices] = values
                    result = semigreedy_scorer(values, binder_ids)
                    if not isinstance(result, Mapping):
                        raise TypeError("semigreedy_scorer must return a mapping")
                    return dict(result)

                outcome = run_semigreedy(
                    predict_fn,
                    design_indices=self.design_indices,
                    seq_design_start=design_start,
                    iters=public_stage.steps,
                    tries=public_stage.tries or 1,
                    e_tries=public_stage.tries or 1,
                    seq_logits=np.asarray(self.logits, np.float32),
                    aa_bias=np.asarray(self.bias, np.float32),
                    rng=np.random.RandomState(self.public_spec.seed + self.global_step),
                )
                best_ids = np.asarray(outcome["best_seq_design"], np.int32)
                saturated = np.full(best_ids.shape + (20,), -1e4, np.float32)
                saturated[np.arange(best_ids.size), best_ids] = 1e4
                self.logits = self._jnp.asarray(saturated)
                for semigreedy_row in outcome["trajectory"][1:]:
                    accepted_ids = np.asarray(semigreedy_row["seq_design"], np.int32)
                    binder_ids = self.fixed_binder_ids.copy()
                    binder_ids[self.design_local_indices] = accepted_ids
                    row = {
                        "stage": stage_name,
                        "stage_type": "semigreedy",
                        "stage_step": int(semigreedy_row["step"]),
                        "global_step": self.global_step,
                        "evaluation_forward": None,
                        "soft": 1.0,
                        "temperature": float(public_stage.temperature_end or 0.01),
                        "hard": 1.0,
                        "learning_rate": 0.0,
                        "loss": float(semigreedy_row["scoring_loss"]),
                        "gradient_norm": 0.0,
                        "hard_sequence": "".join(
                            AF3_AA_ORDER[int(value)] for value in binder_ids
                        ),
                        "semigreedy_tries": int(semigreedy_row["num_tries"]),
                        "best_scoring_loss": float(
                            semigreedy_row["best_scoring_loss"]
                        ),
                    }
                    self.trajectory.append(row)
                    self.global_step += 1
                self.stage_summaries.append(
                    {
                        "stage": stage_name,
                        "stage_type": "semigreedy",
                        "completed_steps": public_stage.steps,
                        "best_loss": float(outcome["best_scoring_loss"]),
                        "best_evaluation_forward": None,
                        "runtime_seconds": round(time.time() - started, 6),
                        "candidate_scorer": "python_api",
                    }
                )
                continue
            stage = stage_config_from_public(public_stage, self.public_spec.learning_rate)
            optimizer = optimizer_factory(1.0)
            optimizer_state = optimizer.init(self.logits)
            stage_best: dict[str, Any] | None = None
            stage_stopped = False
            started = time.time()
            from .config import ramp_value

            for index in range(stage.iterations):
                soft = ramp_value(stage.soft, stage.e_soft if stage.e_soft is not None else stage.soft, index, stage.iterations)
                temperature = ramp_value(stage.temp, stage.e_temp if stage.e_temp is not None else stage.temp, index, stage.iterations, quadratic=True)
                hard = ramp_value(stage.hard, stage.e_hard if stage.e_hard is not None else stage.hard, index, stage.iterations)
                step_scale = ramp_value(stage.step, stage.e_step if stage.e_step is not None else stage.step, index, stage.iterations)
                eval_logits = self.logits
                batch, binder_ids = self._carrier_batch(
                    eval_logits, soft, temperature, hard, scratch
                )
                key = self._jax.random.fold_in(
                    self._jax.random.PRNGKey(self.public_spec.seed), self.global_step
                )
                (loss, aux), gradient = self._value_and_grad(
                    eval_logits,
                    batch,
                    self._jnp.float32(soft),
                    self._jnp.float32(temperature),
                    self._jnp.float32(hard),
                    key,
                )
                loss_value = float(loss)
                gradient_np = np.asarray(gradient, np.float32)
                if not np.isfinite(loss_value) or not np.isfinite(gradient_np).all():
                    raise FloatingPointError("non-finite AF3 Hallucination loss or gradient")
                if self.cfg.norm_seq_grad:
                    gradient = self._sequence_module.norm_seq_grad(gradient)
                updates, optimizer_state = optimizer.update(
                    gradient, optimizer_state, eval_logits
                )
                lr = (stage.learning_rate or self.public_spec.learning_rate) * step_scale * (
                    (1.0 - soft) + soft * temperature
                )
                row = {
                    "stage": public_stage.name or public_stage.type,
                    "stage_type": public_stage.type,
                    "stage_step": index,
                    "global_step": self.global_step,
                    "evaluation_forward": self.global_step + 1,
                    "soft": float(soft),
                    "temperature": float(temperature),
                    "hard": float(hard),
                    "learning_rate": float(lr),
                    "loss": loss_value,
                    "gradient_norm": float(np.linalg.norm(gradient_np)),
                    "hard_sequence": "".join(AF3_AA_ORDER[int(x)] for x in binder_ids),
                    **{name: float(value) for name, value in aux.items()},
                }
                if stage_best is None or loss_value < stage_best["loss"]:
                    stage_best = {
                        "loss": loss_value,
                        "evaluation_forward": self.global_step + 1,
                        "logits": np.asarray(eval_logits, np.float32).copy(),
                    }
                self.trajectory.append(row)
                if step_callback is not None:
                    step_callback(dict(row), np.asarray(eval_logits, np.float32))
                self.global_step += 1
                if should_stop is not None and should_stop(dict(row)):
                    self.logits = eval_logits
                    stage_stopped = True
                    break
                self.logits = eval_logits + lr * updates
            if stage_best is None:
                raise RuntimeError("a non-empty gradient stage produced no evaluations")
            self.stage_summaries.append(
                {
                    "stage": public_stage.name or public_stage.type,
                    "stage_type": public_stage.type,
                    "completed_steps": index + 1,
                    "best_loss": stage_best["loss"],
                    "best_evaluation_forward": stage_best["evaluation_forward"],
                    "runtime_seconds": round(time.time() - started, 6),
                }
            )
            if stage_stopped:
                break
        return list(self.trajectory)

    @property
    def parameter_identifier(self) -> str | None:
        meta = self.parameters.get("__meta__", {})
        raw = meta.get("__identifier__")
        if raw is None:
            return None
        value = np.asarray(raw).tobytes()
        try:
            return value.decode().rstrip("\x00")
        except UnicodeDecodeError:
            return hashlib.sha256(value).hexdigest()


def compile_stopper(spec: Mapping[str, Any]) -> Callable[[Mapping[str, Any]], bool]:
    """Compile a simple, explicit metric-based stopping rule."""

    normalized = validate_stopper(spec)
    stopper_type = normalized["type"]
    if stopper_type == "none":
        return lambda row: False
    compiled = []
    for condition in normalized["conditions"]:
        compiled.append(
            (condition["metric"], condition["operator"], condition["value"])
        )

    def evaluate(row: Mapping[str, Any]) -> bool:
        outcomes = []
        for metric, operator, threshold in compiled:
            if metric not in row:
                outcomes.append(False)
                continue
            try:
                value = float(row[metric])
            except (TypeError, ValueError):
                outcomes.append(False)
                continue
            if not math.isfinite(value):
                outcomes.append(False)
                continue
            outcomes.append(
                {
                    "<=": value <= threshold,
                    "<": value < threshold,
                    ">=": value >= threshold,
                    ">": value > threshold,
                }[operator]
            )
        return all(outcomes) if stopper_type == "all" else any(outcomes)

    return evaluate

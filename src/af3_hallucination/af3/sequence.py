"""Sequence parameterization: logits -> soft / hard / pseudo.

Direct port of the model-agnostic core of ColabDesign `soft_seq()`
(external/colabdesign/colabdesign/shared/model.py:194-223). This part is not
AF2/AF3-specific and is reused verbatim in spirit: only the alphabet width and
the chain/design masking are project choices.

The optimisation variable is `logits` over the declared design positions only.
The binder protocol keeps the target and non-design binder positions fixed, so
the gradient flows only into those explicitly mutable positions.

AF3 polymer alphabet width is residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP
(= 31), the same width create_target_feat() one-hots aatype to.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

# AF3 token aatype one-hot width. Imported lazily so config/tests don't need
# alphafold3 on the path; assert against the real constant inside the runner.
AATYPE_WIDTH = 31


@dataclasses.dataclass
class SeqOpt:
    """Per-step sequence options, mirroring ColabDesign opt[...].

    alpha default is 2.0 to match ColabDesign af/model.py:49 (`opt["alpha"]=2.0`),
    used as the logit scale in soft_seq (`logits = input * alpha`).
    """

    soft: float = 0.0     # 0 -> raw logits path, 1 -> full softmax
    temp: float = 1.0     # softmax temperature
    hard: float = 0.0     # 0 -> soft, 1 -> straight-through one-hot
    alpha: float = 2.0    # logit scale on the input (ColabDesign default 2.0)


def soft_seq(logits: jnp.ndarray, opt: SeqOpt, bias: jnp.ndarray | None = None) -> dict:
    """Turn raw sequence logits into the soft/hard/pseudo representations.

    Returns a dict with keys logits / pssm / soft / hard / pseudo, all shape
    [..., width]. `pseudo` is what gets injected into features. This follows
    ColabDesign exactly:

      pseudo = opt.soft * soft + (1 - opt.soft) * input
      pseudo = opt.hard * hard + (1 - opt.hard) * pseudo

    where `input` is the raw optimisation variable before alpha/bias. This
    matters for BindCraft's logits stage (`soft < 1`) and hard stage; replacing
    the raw branch with pssm would silently change the design schedule.
    """
    seq_input = logits
    x = seq_input * opt.alpha
    if bias is not None:
        x = x + bias

    pssm = jax.nn.softmax(x, axis=-1)
    soft = jax.nn.softmax(x / opt.temp, axis=-1)

    # straight-through one-hot: forward = one_hot(argmax), backward = soft grad
    hard = jax.nn.one_hot(jnp.argmax(soft, axis=-1), soft.shape[-1])
    hard = jax.lax.stop_gradient(hard - soft) + soft

    pseudo = opt.soft * soft + (1 - opt.soft) * seq_input
    pseudo = opt.hard * hard + (1 - opt.hard) * pseudo

    return {"logits": x, "pssm": pssm, "soft": soft, "hard": hard, "pseudo": pseudo}


def norm_seq_grad(grad: jnp.ndarray) -> jnp.ndarray:
    """Sequence-gradient normalisation, ColabDesign `_norm_seq_grad`.

    Verbatim port of shared/model.py:128-132 (numpy -> jnp). Rescales the
    gradient so its Frobenius norm over the (length, alphabet) axes equals
    sqrt(eff_L), where eff_L is the number of positions with a non-zero gradient:

        eff_L = (g**2.sum(-1) > 0).sum(-2)
        gn    = ||g||_(L,A)
        g_new = g * sqrt(eff_L) / (gn + 1e-7)

    Operates on the last two axes, so it handles both [L, A] design logits and
    ColabDesign's [num_seq, L, A]. Applied to the design-logit gradient BEFORE
    the optimizer update (af/design.py:219).
    """
    g = grad
    eff_L = (jnp.square(g).sum(-1, keepdims=True) > 0).sum(-2, keepdims=True)
    gn = jnp.linalg.norm(g, axis=(-1, -2), keepdims=True)
    return g * jnp.sqrt(eff_L.astype(g.dtype)) / (gn + 1e-7)


def init_logits(length: int, width: int = AATYPE_WIDTH, scale: float = 0.0,
                key: jax.Array | None = None) -> jnp.ndarray:
    """Initialise design logits.

    scale=0 -> zeros (uniform / non-saturated). The af3-bc-backprop-parity run
    showed saturated one-hot init gives ~1e-8 gradients under bf16; non-saturated
    init gives healthy ~1e-4. Default to non-saturated.
    """
    if key is not None and scale > 0:
        return scale * jax.random.normal(key, (length, width))
    return jnp.zeros((length, width), dtype=jnp.float32)


def aa_bias(width: int, omit_aa_indices: tuple[int, ...] = (), value: float = -1e7) -> jnp.ndarray:
    """Banned amino-acid bias (ColabDesign rm_aa): large negative logit bias.

    Returns shape [width]; broadcast over positions at soft_seq time.
    """
    bias = jnp.zeros((width,), dtype=jnp.float32)
    if omit_aa_indices:
        idx = jnp.asarray(omit_aa_indices, dtype=jnp.int32)
        bias = bias.at[idx].set(value)
    return bias

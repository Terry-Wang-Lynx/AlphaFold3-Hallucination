"""Local block rematerialisation adapter for the AF3 trunk.

The experiment prototype (experiment/runs/2026-06-26_af3-v0-diff-loop) proved
per-block remat can shrink the bucket-256 design backward pass from OOM to
3.64 GB when every relevant layer_stack is wrapped, but it did so by GLOBALLY
monkeypatching hk.experimental.layer_stack. That is an experiment-only hack and
is explicitly disallowed in code/ (task acceptance criteria + Codex interface
decision 2026-06-26 20:33).

This module provides the local alternative: a small wrapper a *subclassed* trunk
forward calls on its per-layer function before handing it to layer_stack. No
global symbol is patched; each stack must opt in by routing its block fn through
`maybe_remat`. Current prototype coverage is pairformer-only; MSA stack support
requires a separate `_embed_process_msa` override.

Wiring sketch (in the runner's trunk adapter, NOT here):

    from alphafold3.model.network import evoformer as ev
    class DesignTrunk(ev.Evoformer):
        def _build_stack(self, block_fn, num_layers):
            block_fn = maybe_remat(block_fn, self._remat_cfg)
            return hk.experimental.layer_stack(num_layers)(block_fn)

AF3 already exposes Evoformer.Config.block_remat / remat_block_size that nothing
reads; a clean V1 wires those fields here instead of carrying a separate
RematConfig. For V0 the adapter keeps its own config so code/ does not depend on
patching the external Evoformer.
"""

from __future__ import annotations

from collections.abc import Callable

import haiku as hk

from .config import RematConfig


def maybe_remat(block_fn: Callable, cfg: RematConfig) -> Callable:
    """Return block_fn wrapped in hk.remat when cfg.enabled, else unchanged.

    block_size is reserved for grouped remat (wrap every Nth block); V0 supports
    per-block only and treats any block_size as per-block, logging via the
    docstring contract rather than silently grouping.
    """
    if not cfg.enabled:
        return block_fn
    return hk.remat(block_fn)

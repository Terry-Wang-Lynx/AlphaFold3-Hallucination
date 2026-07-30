# Antibody Workflow

The antibody workflow is a resumable five-step pipeline. Every step receives
the complete artifact registry and publishes named, hashed artifacts.

## Required invariants

- `cdr_only: true`
- `framework_fixed: true`
- `antigen_fixed: true`

Adapters must independently verify these invariants rather than relying on the
configuration declaration alone.

## Plugins

`mock` is deterministic and supports macOS/Linux integration testing.

`command` launches an external adapter as an argument list without a shell. It
can reference `{step_dir}`, `{output_dir}`, `{run_id}`, `{seed}`, and artifacts
published by previous steps. The process must exit zero and create every
declared artifact.

Python users may register in-process plugins through `PluginRegistry`.

The first-party Linux/AF3 plugins are:

- `hallucination/af3_jax`: differentiable trunk/distogram Hallucination.
- `diffusion/af3`: exact checkpoint decoding through official AF3 diffusion.
- `inverse_folding/candidate_command`: external inverse folder with strict
  candidate validation.
- `inverse_folding/frozen_candidates`: interface and frozen-sequence testing only.
- `consistency/af3_fixed_geometry`: fixed-pseudo-beta ConfidenceHead scoring,
  without diffusion.
- `final_evaluation/af3`: independent full AF3 evaluation over explicit seeds.

The first-party model plugins are imported lazily. Listing plugins or planning a
workflow therefore does not import JAX or AF3 on a laptop.

## Inverse-folding contract

An external inverse folder must emit JSON containing `candidates` (the legacy
key `sequences` is also accepted). Each item is either a complete antibody-chain
sequence or an object with `id` and `sequence`. The adapter rejects noncanonical
residues, incorrect lengths, duplicate-only pools, and any mutation outside the
declared design indices. A minimal output is:

```json
{
  "schema_version": "af3h_candidate_pool_v1",
  "candidates": [
    {"id": "sample_0001", "sequence": "FULL_ANTIBODY_SEQUENCE"}
  ]
}
```

## Consistency Gate contract

The no-diffusion Gate combines:

- the candidate sequence in the AF3 trunk inputs; and
- fixed anchor pseudo-beta geometry in the official ConfidenceHead input.

It does not require inverse-folding side-chain coordinates and must report zero
diffusion calls. It is a ranking/filtering stage; full AF3 with independent
seeds remains the final structural evaluation.

This coordinate reduction follows the audited official AF3
`ConfidenceHead._embed_features()` implementation: the dense coordinate array
is converted with `token_atoms_to_pseudo_beta` before its distance features are
formed. Per-atom pLDDT logits are predicted from the single representation and
use the dense-atom dimension only to define output shape. The release adapter
therefore scatters the fixed anchor pseudo-beta coordinates into the expected
dense layout and masks tokens without valid geometry; it does not invent
inverse-folding side chains. See [references.md](references.md) for the audited
AF3 source revision.

The built-in hard thresholds are optional and explicit:
`consistency_loss_max`, `all_cdr_plddt_min`, `cdr3_plddt_min`,
`cdr_patch_pae_max`, and `cdr_patch_pde_max`. If no candidate satisfies all
configured thresholds, the workflow becomes `rejected` and the final AF3 step
is recorded as `skipped` rather than silently selecting a low-scoring sequence.

## Resume

Each completed step records its plugin configuration hash, artifacts, hashes,
metrics, and runtime. `--resume` skips only steps marked completed under the
same complete configuration hash. Configuration drift fails closed.

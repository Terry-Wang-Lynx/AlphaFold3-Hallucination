# Architecture

The project separates research logic from model runtimes.

## Core layer

The import-light core contains configuration parsing, schedule expansion,
plugin discovery, artifact hashing, workflow state, resume checks, local/SSH
execution, and result inspection. It does not import JAX or AlphaFold 3.

## AF3/JAX layer

`af3_hallucination.af3` is optional. It provides:

- ColabDesign-compatible sequence parameterization.
- AF3 target/MSA/profile soft-query injection.
- A hard-carrier re-featurization boundary for legal atom chemistry.
- Differentiable AF3 trunk and distogram execution.
- Built-in contact, helix, globularity, and entropy losses.
- Custom loss injection through the Python API.
- Exact float32 checkpoint callbacks and metric-based stopping.
- Fixed-pseudo-beta, no-diffusion Consistency scoring primitives.
- Official AF3 checkpoint diffusion and independent final-evaluation adapters.

The package calls an independently installed official AF3 runtime. It does not
vendor AF3 or its parameters.

## Antibody workflow

The workflow engine owns orchestration, provenance, and resume behavior. Each
operation is a plugin:

1. Hallucination
2. Diffusion anchor generation
3. CDR inverse folding
4. Consistency Gate
5. Full AF3 evaluation

The public antibody contract fixes the antigen and antibody framework and
allows changes only at declared CDR tokens. Alternative inverse folders and
evaluators are command or Python plugins rather than hard-coded dependencies.

All five built-in workflow boundaries publish versioned manifests plus their
declared array or structure artifacts. The orchestrator records SHA-256, byte
count, producer, runtime, and complete configuration hash. A rejected
Consistency Gate stops the workflow fail-closed.

## Pocket placeholder

The pocket-redesign namespace is intentionally non-functional. Calling it
returns `not_implemented`; no model or scientific capability is implied.

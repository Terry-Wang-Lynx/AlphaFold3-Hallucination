# AlphaFold3 Hallucination

> [!WARNING]
> **Antibody-design validation status:** The effectiveness of the antibody-design
> workflow has not been rigorously validated experimentally. Its outputs must be
> treated as computational research hypotheses, not as evidence of binding,
> therapeutic efficacy, or safety.

An extensible JAX research framework for AlphaFold 3 sequence hallucination,
plus a modular antibody-design workflow.

> Status: public research release candidate `0.1.0rc2`. This repository does not
> include AlphaFold 3 parameters, genetic databases,
> third-party inverse-folding model weights, or precomputed biological inputs.

## What is included

- A configurable Hallucination stage engine with `logits`, `soft`, `hard`, and
  `semigreedy` sequence states.
- BindCraft/ColabDesign-compatible stage and temperature scheduling semantics.
- AF3 soft-query feature injection, differentiable trunk/distogram execution,
  contact losses, checkpoints, observers, and stopping plugins.
- A resumable antibody workflow:

  ```text
  Hallucination -> AF3 diffusion anchor -> CDR inverse folding
                -> fixed-geometry Consistency Gate -> full AF3 evaluation
  ```

- A clearly marked, non-functional small-molecule pocket-redesign placeholder.

The AF3-backed Hallucination, checkpoint-diffusion, fixed-geometry Consistency,
and final-evaluation plugins are included. Inverse folding remains an explicit
adapter boundary so ProteinMPNN, antibody-specific MPNNs, or locally trained
models can be exchanged without changing the workflow engine.

## Platform support

| Platform | Supported use |
| --- | --- |
| macOS 13+ | Install the core package; validate and expand configs; run dry-runs and mock workflows; inspect results; submit work through the optional SSH executor. |
| Linux x86_64 | All macOS capabilities plus local AF3/JAX execution. An NVIDIA GPU and an independently installed AlphaFold 3 runtime are required for model execution. |

The official AF3 model path is intentionally a runtime boundary. Native AF3
GPU inference is not claimed on Apple Silicon.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'

af3h doctor
af3h config validate configs/hallucination/fast20.yaml
af3h schedule show configs/hallucination/fast20.yaml
af3h antibody plan configs/antibody/example.yaml
af3h antibody run configs/antibody/mock.yaml --output-dir /tmp/af3h-mock
af3h audit run /tmp/af3h-mock
```

For a Linux AF3 environment:

```bash
python -m pip install -e .
export PYTHONPATH=/path/to/alphafold3:/path/to/alphafold3/src
af3h doctor --require-af3 --require-jax
af3h hallucinate run configs/hallucination/fast20.yaml \
  --input-json /path/to/populated_input.json \
  --model-dir /path/to/af3/models \
  --output-dir runs/example
```

Install this package into an independently prepared official AF3 environment.
The project deliberately does not provide an `af3` dependency extra because
official AF3 releases own a coupled JAX/Haiku/Optax dependency set; installing
generic pins over that environment can make an otherwise valid runtime
inconsistent.

See [docs/configuration.md](docs/configuration.md),
[docs/antibody-workflow.md](docs/antibody-workflow.md), and
[docs/platforms.md](docs/platforms.md). The exact release-candidate test scope
and its interpretation boundary are recorded in
[docs/validation.md](docs/validation.md).

## Scientific boundary

The package is a research tool. Confidence metrics and computational structure
recurrence are screening evidence, not experimental binding or therapeutic
validation. The bundled defaults reproduce specific implementation semantics;
they are not asserted to be universally optimal.

## License and attribution

Project code is provided under Apache-2.0. AlphaFold 3, BindCraft,
ColabDesign, ProteinMPNN, and optional inverse-folding tools retain their own
licenses and must be installed separately. See [NOTICE](NOTICE) and
[docs/references.md](docs/references.md).

The canonical source repository is
[Terry-Wang-Lynx/AlphaFold3-Hallucination](https://github.com/Terry-Wang-Lynx/AlphaFold3-Hallucination).
See [release/RELEASE_CHECKLIST.md](release/RELEASE_CHECKLIST.md) for the audited
release scope.

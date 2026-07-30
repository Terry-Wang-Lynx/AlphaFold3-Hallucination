# Platforms

## macOS

The core package supports macOS for configuration, scheduling, workflow
orchestration, dry-runs, mock end-to-end tests, output review, and SSH
submission. This is the recommended laptop workflow.

The project does not claim native official AF3 GPU execution on Apple Silicon.
Use `SSHExecutor` or ordinary SSH to run the AF3/JAX backend in a prepared Linux
environment.

## Linux with NVIDIA GPU

Full model execution requires:

- an independently installed official AlphaFold 3 source tree;
- legally obtained AF3 model parameters;
- compatible JAX, Haiku, Optax, CUDA, and NVIDIA driver versions; and
- a populated AF3 input produced by the official data pipeline.

Expose the AF3 source through `PYTHONPATH`, install this package in that same
environment, and run `af3h doctor --require-jax --require-af3` before a smoke.

When AF3 is provided through Apptainer/Singularity, bind the project source and
set `PYTHONPATH=/path/to/af3-hallucination/src:/path/to/alphafold3:/path/to/alphafold3/src`.
The package neither downloads nor discovers model parameters automatically.

The `af3` optional extra is a convenience pin for the release-candidate runtime
(`jax 0.6.1`, `dm-haiku 0.0.14`, `optax 0.2.8`). It is not a substitute for the
official AF3 installation and should not overwrite another AF3 release's own
dependency constraints.

## Resource policy

The framework does not select GPUs automatically. Operators must set
`CUDA_VISIBLE_DEVICES` or use their scheduler. Run manifests preserve the
visible-device value and detected GPU inventory.

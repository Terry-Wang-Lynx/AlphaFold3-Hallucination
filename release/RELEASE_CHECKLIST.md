# Release checklist: 0.1.0rc2

Release-candidate review date: 2026-07-31.

- [x] Product scope contains the open AF3/JAX Hallucination framework.
- [x] Product scope contains the five-step, CDR-only antibody workflow.
- [x] Small-molecule pocket redesign is marked `not_implemented`.
- [x] Core package imports without JAX or AlphaFold 3.
- [x] Strict configuration, schedule, plugin, candidate, resume, and audit tests pass.
- [x] macOS arm64 editable install and clean-wheel validation pass.
- [x] Linux x86_64 editable install and clean-wheel validation pass.
- [x] Linux RTX 4090 AF3/JAX gradient and two-schedule smokes pass.
- [x] Diffusion, no-diffusion Consistency, and multi-seed full AF3 smokes pass.
- [x] First-party five-step workflow, resume, and independent hash audit pass.
- [x] Explicit CDR-only and candidate-eligibility invariants fail closed.
- [x] Checkpoint, anchor, Consistency, and final-evaluation manifests are hash-linked.
- [x] Workflow resume rejects changed upstream artifacts and malformed state.
- [x] GitHub Actions are least-privilege, pinned, and checked with `actionlint`.
- [x] Secret, dependency, static-security, metadata, and wheel-content scans pass.
- [x] A second clean macOS/Linux validation and standalone source-archive replay pass.
- [x] License, NOTICE, citations, contribution, and security documents are present.
- [x] Release tree contains no parameters, databases, populated inputs, raw structures, private paths, or large outputs.
- [x] Wheel and sdist contents have been inspected.
- [x] Publish the audited source tree to GitHub.
- [ ] Publish a package to PyPI or another package index.

The source repository is published at
<https://github.com/Terry-Wang-Lynx/AlphaFold3-Hallucination>. Package-index
publication remains a separate, unauthorised action.

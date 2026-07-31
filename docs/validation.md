# Release-candidate validation

Version `0.1.0rc2` was reviewed on 2026-07-31. This page records what was
actually exercised. It is not a biological benchmark and does not claim that
the bundled example schedule is optimal.

## Platform matrix

| Platform | Environment | Result |
| --- | --- | --- |
| macOS arm64 | Python 3.14.6, clean editable and wheel environments | Core install, static checks, unit/integration tests, CLI, mock workflow, resume, audit, wheel and sdist passed. |
| Linux x86_64 | Python 3.10.12, clean editable and wheel environments | The same core and packaging checks passed. |
| Linux x86_64 + RTX 4090 | Official AF3 3.0.1 runtime, JAX 0.6.1, Haiku 0.0.14, NumPy 2.1.3 | The initial AF3/JAX gradient, two-schedule, diffusion, fixed-geometry Consistency, multi-seed final AF3, and five-step workflow acceptance passed on one visible GPU. |
| Linux x86_64 + RTX 4090 | Official AF3 3.0.3 at `7b197fe`, JAX 0.9.1, Haiku 0.0.16, NumPy 2.4.1 | Final `0.1.0rc2` source repeated a one-forward gradient run and the complete five-step workflow on physical GPU 0; audit and resume passed with 13 linked artifacts. |

The macOS claim is intentionally limited to the import-light research and
orchestration layer. Native official AF3 CUDA inference is not claimed on
Apple Silicon.

## Independent release-readiness audit

The second audit treated `code/` as a standalone GitHub repository rather than
inheriting the first candidate's conclusion. It added fail-closed checks
for explicit chain-local CDR indices, inverse-folding candidate validation,
environment expansion, checkpoint path containment, workflow resume integrity,
artifact/config/input hash linkage, terminal-state consistency, numeric inputs,
external-command timeouts, and final-candidate eligibility. The corresponding
regression suite contains 57 tests.

CI action references are pinned to immutable commits. `actionlint`, `gitleaks`,
Ruff, `compileall`, Bandit, dependency vulnerability inspection, package
metadata validation, wheel-content inspection, and relative-link checks were
also included in the release review. Exact final commands and artifact hashes
are recorded in the repository-level audit run rather than claimed by this
standalone source tree.

## GPU scientific-contract checks

- A one-forward Hallucination run produced finite loss `8.1014109` and finite,
  nonzero gradient norm `6.5906138`. Its checkpoint was exact `float32` with
  shape `[37, 20]`.
- A second schedule containing one `logits` and one `hard` forward completed
  with finite losses `[5.5764532, 6.0057459]` and gradient norms
  `[3.1648872, 16.1317177]`. Both checkpoints were finite `float32 [37, 20]`.
- Checkpoint diffusion produced a valid structure and fixed pseudo-beta
  artifact. The smoke anchor had pTM `0.52597`, ipTM `0.17879`, ranking score
  `0.24823`, and no clash.
- The no-diffusion Consistency Gate evaluated two CDR-only candidates in two
  model forwards and zero diffusion calls. Losses were `0.40539` and `0.47335`;
  the lower-loss candidate was selected.
- Full AF3 evaluated that candidate with independent seeds 31 and 37. The two
  binder pLDDT values were `77.56` and `75.76`; CDR pLDDT values were `52.14`
  and `47.56`. Both outputs were clash-free.
- A first-party `af3h antibody run` completed all five plugin boundaries,
  emitted 13 hashed artifacts, resumed without changing state, and passed the
  independent audit.
- The final `0.1.0rc2` replay on AF3 3.0.3 produced finite one-forward loss
  `8.09603`, an exact finite `float32 [37,20]` checkpoint, a zero-diffusion
  two-candidate Consistency ranking, and one full-AF3 final evaluation. This
  replay also exposed and fixed an argument-isolation incompatibility between
  the argparse CLI and a lazily initialized Abseil/Tokamax runtime.

These deliberately tiny runs validate execution, gradients, masks, fixed
geometry, call separation, artifact contracts, seeds, and provenance. Their
low interface-confidence values are not evidence of a successful antibody
design. Biological performance remains a separate, larger evaluation.

## Runtime provenance

The AF3 source contract was audited at commit
`b2f3d45fbfcacc5183bd5345d15df93571b8437f`. The Linux runtime image did not
contain Git metadata, so the actual imported modules were independently hashed:

| Module | SHA-256 |
| --- | --- |
| `run_alphafold` | `90bfc2615a7d51e7989b39311cb20e4b8ce1b9b8d6cc5354c2e9d0a26ee155bd` |
| `alphafold3.model.model` | `164a2d6550c5eaaab2a0c44e7060d550d4b9b9f0943cc47a149d1668895bcc87` |
| `confidence_head` | `1368027986f851b67717df0bc842f107d28ed09503eda49f6df6bbff403492ae` |
| `diffusion_head` | `282f72f5400f8dd63bb53df504cbe25233857e28936e02fec85ae27edd54de60` |

The AF3 parameter identifier was
`25505ca01c7e2b507045f0464a97ebfc45f456cb03c9855416e6f6dab16c2cf9`.
Model parameters and raw structures are not distributed.

Compatibility was additionally executed against official AF3 3.0.3 commit
`7b197fe859790fc3e04d03ea70dd0b9ba48881c9`. Its imported modules are recorded
by runtime SHA-256 in the generated run manifest. This compatibility replay
does not redefine the older commit used as the semantic source basis for the
ported trunk code.

## Remaining boundaries

- The `candidate_command` inverse-folding adapter was contract-tested with an
  executable fixture. No third-party inverse-folding weights are bundled, and
  this release does not claim a scientific benchmark of one inverse folder.
- `frozen_candidates` is for tests and fixed-sequence evaluations, not a design
  model.
- Consistency thresholds in examples are explicit research settings, not
  universally calibrated clinical or experimental criteria.
- The small-molecule pocket namespace is a non-functional placeholder.

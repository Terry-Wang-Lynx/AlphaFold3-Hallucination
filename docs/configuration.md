# Configuration

Configuration uses strict YAML with `schema_version: 1`. Unknown keys fail
validation, while project-specific metadata may be stored below `extensions`.

## Hallucination stages

Stages may be repeated and ordered freely. Every stage exposes:

- `type`: `logits`, `soft`, `hard`, or `semigreedy`
- `steps`
- `learning_rate`
- `soft_start`, `soft_end`
- `temperature_start`, `temperature_end`
- `hard_start`, `hard_end`
- `step_start`, `step_end`
- `dropout`
- `tries` for semigreedy search

Temperature follows the audited ColabDesign quadratic schedule. Soft, hard,
and step values use linear schedules. The effective gradient learning rate is:

```text
learning_rate * step_scale * ((1 - soft) + soft * temperature)
```

`semigreedy` is gradient-free. The Python engine requires an explicit candidate
scorer returning the BindCraft-style forward-scoring fields; the CLI refuses to
invent a confidence scorer when none is supplied.

`dropout` is retained in the portable schedule schema for source parity, but
`dropout: true` currently fails validation because the official AF3 trunk adapter
does not expose a validated inference-dropout switch. It is never silently ignored.

## Built-in losses

- `intra_contact`
- `interface_contact`
- `helix`
- `globularity`
- `sequence_entropy`

Weights may be zero or negative. A custom Python `loss_fn` may replace the
built-in total without changing the stage engine.

## Design and hotspot indices

The AF3 backend accepts either padded global token indices or indices local to
one chain:

```yaml
backend_config:
  binder_chain_id: B
  binder_asym_id: 2
  target_asym_id: 1
  design_local_indices: [26, 27, 28]
  hotspot_local_indices: [10, 11]
```

For production antibody runs, derive and freeze CDR indices with a documented
numbering scheme before launching the model. The package deliberately does not
guess CDR boundaries from sequence alone.

## Checkpoints and stopping

`checkpoints` are one-based evaluation-forward numbers. Saved values are exact
float32 logits evaluated at that forward, before the optimizer update.

A metric stopper may require all or any conditions:

```yaml
stopper:
  type: all
  conditions:
    - {metric: i_con, operator: "<=", value: 0.3}
    - {metric: con, operator: "<=", value: 0.5}
```

Loss thresholds are operational rules, not proof that a decoded structure is
unique or experimentally functional.

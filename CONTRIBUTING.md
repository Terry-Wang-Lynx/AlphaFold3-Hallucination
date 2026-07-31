# Contributing

Use Python 3.10 or newer. Keep the core importable without JAX or AlphaFold 3.

```bash
python -m pip install -e '.[dev,release]'
ruff check src tests
pytest
python -m build
twine check dist/*
check-wheel-contents dist/*.whl
```

New AF3 behavior must identify whether it directly reuses, semantically
reproduces, or newly implements behavior relative to AlphaFold 3,
BindCraft/ColabDesign, and must include a focused contract test.

Do not commit parameters, databases, populated AF3 inputs, structures, private
paths, credentials, or large generated outputs.

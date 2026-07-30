from pathlib import Path

import pytest

from af3_hallucination.config import ConfigurationError, load_config, parse_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "path",
    sorted((ROOT / "configs").glob("**/*.yaml")),
)
def test_all_bundled_configs_validate(path):
    config = load_config(path)
    assert len(config.sha256) == 64
    assert config.to_dict()["schema_version"] == 1


def test_unknown_top_level_key_fails():
    with pytest.raises(ConfigurationError, match="unknown top-level"):
        parse_config(
            {
                "schema_version": 1,
                "kind": "hallucination",
                "hallucination": {
                    "stages": [{"type": "logits", "steps": 1}],
                    "losses": [],
                },
                "typo": True,
            }
        )


def test_checkpoint_cannot_exceed_gradient_steps():
    with pytest.raises(ConfigurationError, match="checkpoint exceeds"):
        parse_config(
            {
                "schema_version": 1,
                "kind": "hallucination",
                "hallucination": {
                    "stages": [{"type": "logits", "steps": 1}],
                    "losses": [],
                    "checkpoints": [2],
                },
            }
        )


def test_unsupported_dropout_is_not_silently_ignored():
    with pytest.raises(ConfigurationError, match="dropout=true is not supported"):
        parse_config(
            {
                "schema_version": 1,
                "kind": "hallucination",
                "hallucination": {
                    "stages": [{"type": "logits", "steps": 1, "dropout": True}],
                    "losses": [],
                },
            }
        )


def test_antibody_invariants_fail_closed():
    value = load_config(ROOT / "configs/antibody/mock.yaml").to_dict()
    value["antibody"]["framework_fixed"] = False
    with pytest.raises(ConfigurationError, match="currently requires"):
        parse_config(value)

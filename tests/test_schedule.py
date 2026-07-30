import pytest

from af3_hallucination.config import HallucinationSpec
from af3_hallucination.schedule import bindcraft_four_stage, expand_schedule


def test_bindcraft_four_stage_counts_and_boundaries():
    spec = HallucinationSpec(
        backend="af3_jax",
        stages=bindcraft_four_stage(),
        losses=(),
    )
    rows = expand_schedule(spec)
    assert len(rows) == 140
    assert sum(row["gradient_enabled"] for row in rows) == 125
    assert rows[0]["stage"] == "g1_logits"
    assert rows[0]["soft"] == pytest.approx(0.9 / 50)
    assert rows[49]["soft"] == pytest.approx(0.9)
    assert rows[50]["stage"] == "additional_logits"
    assert rows[50]["soft"] == pytest.approx(1.0 / 25)
    assert rows[74]["soft"] == pytest.approx(1.0)
    assert rows[75]["stage"] == "g2_soft"
    assert rows[75]["temperature"] == pytest.approx(0.01 + 0.99 * (44 / 45) ** 2)
    assert rows[119]["temperature"] == pytest.approx(0.01)
    assert rows[120]["hard"] == pytest.approx(1.0)
    assert not rows[125]["gradient_enabled"]


def test_stage_order_is_user_defined():
    stages = bindcraft_four_stage()
    spec = HallucinationSpec(
        backend="af3_jax",
        stages=(stages[2], stages[0]),
        losses=(),
    )
    rows = expand_schedule(spec)
    assert rows[0]["stage"] == "g2_soft"
    assert rows[45]["stage"] == "g1_logits"

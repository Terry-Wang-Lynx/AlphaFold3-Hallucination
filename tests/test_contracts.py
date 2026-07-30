import pytest

from af3_hallucination.contracts import normalize_candidate_pool
from af3_hallucination.errors import PluginError


def test_candidate_pool_enforces_cdr_only_and_deduplicates():
    rows = normalize_candidate_pool(
        {
            "candidates": [
                {"id": "first", "sequence": "AANAA"},
                {"id": "duplicate", "sequence": "AANAA"},
                "AARAA",
            ]
        },
        reference_sequence="AAAAA",
        design_local_indices=[2],
    )
    assert [row["sequence"] for row in rows] == ["AANAA", "AARAA"]
    assert rows[0]["id"] == "first"


def test_candidate_pool_rejects_framework_change():
    with pytest.raises(PluginError, match="fixed antibody framework"):
        normalize_candidate_pool(
            {"candidates": ["VAAAA"]},
            reference_sequence="AAAAA",
            design_local_indices=[2],
        )


def test_candidate_pool_rejects_duplicate_ids():
    with pytest.raises(PluginError, match="duplicated"):
        normalize_candidate_pool(
            {
                "candidates": [
                    {"id": "same", "sequence": "AANAA"},
                    {"id": "same", "sequence": "AARAA"},
                ]
            },
            reference_sequence="AAAAA",
            design_local_indices=[2],
        )

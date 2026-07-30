import pytest

np = pytest.importorskip("numpy")
run_semigreedy = pytest.importorskip(
    "af3_hallucination.af3.semigreedy"
).run_semigreedy


def test_semigreedy_executes_and_preserves_sequence_trace():
    def scorer(sequence):
        sequence = np.asarray(sequence)
        loss = float(np.sum(sequence != 1))
        return {
            "scoring_loss": loss,
            "con": loss,
            "i_con": 0.0,
            "pae": 0.0,
            "i_pae": 0.0,
            "binder_plddt": 80.0,
            "iptm": 0.7,
            "per_token_plddt": {"design": [50.0] * len(sequence)},
        }

    result = run_semigreedy(
        scorer,
        design_indices=np.arange(4),
        seq_design_start=np.zeros(4, dtype=int),
        iters=3,
        tries=2,
        rng=np.random.RandomState(9),
    )
    assert len(result["trajectory"]) == 4
    assert all(len(row["seq_design"]) == 4 for row in result["trajectory"])
    assert result["best_scoring_loss"] <= result["trajectory"][0]["scoring_loss"]

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from af3_hallucination.af3.runtime import (
    decode_checkpoint,
    evaluate_selected_candidates,
    score_consistency_pool,
)
from af3_hallucination.config import load_config
from af3_hallucination.hashing import canonical_sha256, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def _anchor(tmp_path):
    candidate_input = tmp_path / "anchor_input.json"
    candidate_input.write_text(
        json.dumps(
            {
                "sequences": [
                    {"protein": {"id": "A", "sequence": "GG"}},
                    {"protein": {"id": "B", "sequence": "AAAAA"}},
                ]
            }
        )
    )
    anchor_path = tmp_path / "anchor_manifest.json"
    anchor = {
        "schema_version": "af3h_anchor_v1",
        "status": "completed",
        "candidate_input_path": str(candidate_input),
        "candidate_input_sha256": sha256_file(candidate_input),
        "hard_sequence": "AAAAA",
        "binder_chain_id": "B",
        "binder_asym_id": 2,
        "design_token_indices": [4],
        "design_local_indices": [2],
        "hotspot_token_indices": [0],
        "bucket": 32,
    }
    anchor_path.write_text(json.dumps(anchor))
    return anchor_path, candidate_input


def test_decode_checkpoint_rejects_path_escape_before_optional_runtime_import(tmp_path):
    escaped = tmp_path / "escaped.npy"
    np.save(escaped, np.zeros((1, 20), np.float32), allow_pickle=False)
    run = tmp_path / "run"
    run.mkdir()
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps({"sequences": [{"protein": {"id": "B", "sequence": "A"}}]})
    )
    config = load_config(ROOT / "configs/hallucination/fast20.yaml")
    checkpoints = run / "checkpoints.json"
    checkpoints.write_text(
        json.dumps(
            {
                "schema_version": "af3h_checkpoints_v1",
                "hallucination_sha256": canonical_sha256(
                    dataclasses.asdict(config.hallucination)
                ),
                "input_sha256": sha256_file(input_path),
                "checkpoints": [
                    {
                        "evaluation_forward": 1,
                        "path": "../escaped.npy",
                        "sha256": sha256_file(escaped),
                        "sequence_options": {
                            "soft": 0.1,
                            "temperature": 1.0,
                            "hard": 0.0,
                        },
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="escapes the Hallucination run"):
        decode_checkpoint(
            public_spec=config.hallucination,
            input_json=input_path,
            checkpoints_json=checkpoints,
            model_dir=tmp_path / "models",
            output_dir=tmp_path / "output",
            checkpoint_forward=1,
            trunk_seed=0,
            diffusion_seed=0,
            diffusion_steps=1,
            num_recycles=0,
        )


def test_decode_checkpoint_rejects_duplicate_forwards_before_optional_runtime_import(
    tmp_path,
):
    run = tmp_path / "run"
    run.mkdir()
    logits = run / "f1.npy"
    np.save(logits, np.zeros((1, 20), np.float32), allow_pickle=False)
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps({"sequences": [{"protein": {"id": "B", "sequence": "A"}}]})
    )
    config = load_config(ROOT / "configs/hallucination/fast20.yaml")
    record = {
        "evaluation_forward": 1,
        "path": "f1.npy",
        "sha256": sha256_file(logits),
        "sequence_options": {"soft": 0.1, "temperature": 1.0, "hard": 0.0},
    }
    checkpoints = run / "checkpoints.json"
    checkpoints.write_text(
        json.dumps(
            {
                "schema_version": "af3h_checkpoints_v1",
                "hallucination_sha256": canonical_sha256(
                    dataclasses.asdict(config.hallucination)
                ),
                "input_sha256": sha256_file(input_path),
                "checkpoints": [record, record],
            }
        )
    )

    with pytest.raises(ValueError, match="duplicate checkpoint evaluation_forward"):
        decode_checkpoint(
            public_spec=config.hallucination,
            input_json=input_path,
            checkpoints_json=checkpoints,
            model_dir=tmp_path / "models",
            output_dir=tmp_path / "output",
            checkpoint_forward=1,
            trunk_seed=0,
            diffusion_seed=0,
            diffusion_steps=1,
            num_recycles=0,
        )


def test_consistency_rejects_tampered_anchor_input_before_model_import(tmp_path):
    anchor_path, candidate_input = _anchor(tmp_path)
    candidate_input.write_text("tampered\n")
    pool = tmp_path / "candidates.json"
    pool.write_text(json.dumps({"candidates": ["AANAA"]}))

    with pytest.raises(ValueError, match="candidate input hash mismatch"):
        score_consistency_pool(
            anchor_manifest=anchor_path,
            candidate_pool=pool,
            model_dir=tmp_path / "models",
            output_dir=tmp_path / "output",
            scorer_seed=0,
            design_recycles=0,
            cdr3_local_indices=(),
            thresholds={},
            top_k=1,
        )


def test_final_evaluation_rejects_mismatched_consistency_anchor_before_model_import(
    tmp_path,
):
    anchor_path, _ = _anchor(tmp_path)
    consistency = tmp_path / "consistency.json"
    consistency.write_text(
        json.dumps(
            {
                "schema_version": "af3h_consistency_v1",
                "status": "completed",
                "anchor_manifest_sha256": "wrong",
                "selected": [{"id": "c0", "sequence": "AANAA", "eligible": True}],
            }
        )
    )

    with pytest.raises(ValueError, match="different anchor manifest"):
        evaluate_selected_candidates(
            anchor_manifest=anchor_path,
            consistency_results=consistency,
            model_dir=tmp_path / "models",
            output_dir=tmp_path / "output",
            seeds=(1,),
            diffusion_steps=1,
            num_recycles=0,
        )


def test_final_evaluation_revalidates_selected_candidate_eligibility(tmp_path):
    anchor_path, _ = _anchor(tmp_path)
    consistency = tmp_path / "consistency.json"
    consistency.write_text(
        json.dumps(
            {
                "schema_version": "af3h_consistency_v1",
                "status": "completed",
                "anchor_manifest_sha256": sha256_file(anchor_path),
                "selected": [{"id": "c0", "sequence": "AANAA", "eligible": False}],
            }
        )
    )

    with pytest.raises(ValueError, match="explicitly eligible"):
        evaluate_selected_candidates(
            anchor_manifest=anchor_path,
            consistency_results=consistency,
            model_dir=tmp_path / "models",
            output_dir=tmp_path / "output",
            seeds=(1,),
            diffusion_steps=1,
            num_recycles=0,
        )

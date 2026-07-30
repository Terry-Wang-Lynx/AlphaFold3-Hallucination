import json
from pathlib import Path

import pytest

from af3_hallucination.config import load_config
from af3_hallucination.errors import ResumeError
from af3_hallucination.plugins import PluginResult, default_registry
from af3_hallucination.workflow import AntibodyWorkflow

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_mock_antibody_workflow_and_resume(tmp_path):
    config = load_config(ROOT / "configs/antibody/mock.yaml")
    workflow = AntibodyWorkflow(config)
    state = workflow.run(tmp_path / "run")
    assert state["status"] == "completed"
    assert len(state["steps"]) == 5
    assert len(state["artifacts"]) == 5
    first_state = json.loads((tmp_path / "run/run_state.json").read_text())
    resumed = workflow.run(tmp_path / "run", resume=True)
    assert resumed["completed_at_unix"] == first_state["completed_at_unix"]


def test_existing_run_requires_resume(tmp_path):
    config = load_config(ROOT / "configs/antibody/mock.yaml")
    workflow = AntibodyWorkflow(config)
    workflow.run(tmp_path / "run")
    with pytest.raises(ResumeError, match="use --resume"):
        workflow.run(tmp_path / "run")


def test_resume_fails_if_recorded_artifact_changed(tmp_path):
    config = load_config(ROOT / "configs/antibody/mock.yaml")
    workflow = AntibodyWorkflow(config)
    output = tmp_path / "run"
    state = workflow.run(output)
    first = next(iter(state["artifacts"].values()))
    Path(first["path"]).write_text("tampered\n")
    with pytest.raises(ResumeError, match="artifact .* changed"):
        workflow.run(output, resume=True)


def test_rejected_gate_skips_final_evaluation(tmp_path):
    class Reject:
        def run(self, *, context, config):
            del config
            artifact = context.step_dir / "rejected.json"
            artifact.write_text('{"status":"rejected"}\n')
            return PluginResult("rejected", {"decision": artifact}, {"eligible": 0})

    config = load_config(ROOT / "configs/antibody/mock.yaml")
    registry = default_registry()
    registry.register("consistency", "mock", Reject, replace=True)
    state = AntibodyWorkflow(config, registry=registry).run(tmp_path / "rejected")
    assert state["status"] == "rejected"
    assert state["steps"]["consistency"]["status"] == "rejected"
    assert state["steps"]["final_evaluation"]["status"] == "skipped"

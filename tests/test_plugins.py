import json
import sys
from pathlib import Path

import pytest

from af3_hallucination.builtin_plugins import (
    AF3JaxHallucinationPlugin,
    CandidateCommandPlugin,
)
from af3_hallucination.config import load_config
from af3_hallucination.errors import PluginError
from af3_hallucination.plugins import CommandPlugin, PluginRegistry, default_registry
from af3_hallucination.workflow import WorkflowContext


def test_registry_rejects_duplicate_and_unknown():
    registry = PluginRegistry()
    registry.register("kind", "name", object)
    with pytest.raises(PluginError, match="already registered"):
        registry.register("kind", "name", object)
    with pytest.raises(PluginError, match="unknown plugin"):
        registry.create("kind", "missing")


def test_first_party_plugin_inventory_is_import_light():
    inventory = default_registry().inventory()
    assert "af3_jax" in inventory["hallucination"]
    assert "af3" in inventory["diffusion"]
    assert "candidate_command" in inventory["inverse_folding"]
    assert "command" not in inventory["inverse_folding"]
    assert "af3_fixed_geometry" in inventory["consistency"]
    assert "af3" in inventory["final_evaluation"]


def test_command_plugin_without_shell(tmp_path):
    script = tmp_path / "adapter.py"
    script.write_text(
        "from pathlib import Path\n"
        "import json, sys\n"
        "root=Path(sys.argv[1]); (root/'artifact.txt').write_text('ok\\n')\n"
        "(root/'result.json').write_text(json.dumps({'status':'completed','metrics':{'ok':True}}))\n"
    )
    context = WorkflowContext(
        output_dir=tmp_path,
        run_id="run",
        seed=1,
        artifacts={},
        step_name="step",
        step_dir=tmp_path / "step",
    )
    result = CommandPlugin().run(
        context=context,
        config={
            "command": [sys.executable, str(script), "{step_dir}"],
            "artifacts": {"value": "artifact.txt"},
        },
    )
    assert result.status == "completed"
    assert result.metrics["ok"] is True
    assert result.artifacts["value"].read_text() == "ok\n"
    assert json.loads((context.step_dir / "result.json").read_text())["status"] == "completed"


def test_builtin_plugins_reject_unknown_configuration_keys(tmp_path):
    context = WorkflowContext(
        output_dir=tmp_path,
        run_id="run",
        seed=1,
        artifacts={},
        step_name="step",
        step_dir=tmp_path / "step",
    )
    with pytest.raises(PluginError, match="unknown command plugin keys"):
        CommandPlugin().run(
            context=context,
            config={"command": [sys.executable, "-c", "pass"], "typo": True},
        )


def test_command_plugin_timeout_fails_closed_and_writes_logs(tmp_path):
    context = WorkflowContext(
        output_dir=tmp_path,
        run_id="run",
        seed=1,
        artifacts={},
        step_name="step",
        step_dir=tmp_path / "step",
    )
    with pytest.raises(PluginError, match="timed out"):
        CommandPlugin().run(
            context=context,
            config={
                "command": [sys.executable, "-c", "import time; time.sleep(1)"],
                "timeout_seconds": 0.01,
            },
        )
    assert (context.step_dir / "stdout.log").is_file()
    assert (context.step_dir / "stderr.log").is_file()


def test_candidate_command_executes_and_validates_cdr_only_pool(tmp_path):
    anchor = tmp_path / "anchor.json"
    anchor.write_text(json.dumps({"hard_sequence": "AAAAA", "design_local_indices": [2]}))
    script = tmp_path / "inverse_folder.py"
    script.write_text(
        "from pathlib import Path\n"
        "import json, sys\n"
        "out=Path(sys.argv[1]); out.write_text(json.dumps("
        "{'schema_version':'af3h_candidate_pool_v1','candidates':["
        "{'id':'s0','sequence':'AANAA'},{'id':'s1','sequence':'AARAA'}]}))\n"
        "Path('result.json').write_text(json.dumps({'status':'completed'}))\n"
    )
    step = tmp_path / "step"
    step.mkdir()
    context = WorkflowContext(
        output_dir=tmp_path,
        run_id="run",
        seed=1,
        artifacts={"anchor_manifest": anchor},
        step_name="inverse_folding",
        step_dir=step,
    )
    result = CandidateCommandPlugin().run(
        context=context,
        config={
            "command": [sys.executable, str(script), "{step_dir}/raw.json"],
            "artifacts": {"candidate_pool": "raw.json"},
        },
    )
    validated = json.loads(result.artifacts["candidate_pool"].read_text())
    assert [row["sequence"] for row in validated["candidates"]] == ["AANAA", "AARAA"]


def test_first_party_antibody_hallucination_requires_chain_local_design_indices(tmp_path):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/antibody/mock.yaml")
    context = WorkflowContext(
        output_dir=tmp_path,
        run_id="run",
        seed=1,
        artifacts={},
        project_config=config,
        config_dir=tmp_path,
        step_name="hallucination",
        step_dir=tmp_path / "step",
    )
    with pytest.raises(PluginError, match="CDR-only.*design_local_indices"):
        AF3JaxHallucinationPlugin().run(context=context, config={})


def test_first_party_runtime_path_rejects_unset_environment_variable(tmp_path, monkeypatch):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/antibody/example.yaml")
    context = WorkflowContext(
        output_dir=tmp_path,
        run_id="run",
        seed=1,
        artifacts={},
        project_config=config,
        config_dir=tmp_path,
        step_name="hallucination",
        step_dir=tmp_path / "step",
    )
    monkeypatch.delenv("AF3H_MISSING_INPUT", raising=False)
    with pytest.raises(PluginError, match="unset environment variables"):
        AF3JaxHallucinationPlugin().run(
            context=context,
            config={"input_json": "${AF3H_MISSING_INPUT}", "model_dir": "/unused"},
        )

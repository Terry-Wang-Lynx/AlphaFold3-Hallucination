import json
import sys

import pytest

from af3_hallucination.builtin_plugins import CandidateCommandPlugin
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


def test_candidate_command_executes_and_validates_cdr_only_pool(tmp_path):
    anchor = tmp_path / "anchor.json"
    anchor.write_text(
        json.dumps({"hard_sequence": "AAAAA", "design_local_indices": [2]})
    )
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

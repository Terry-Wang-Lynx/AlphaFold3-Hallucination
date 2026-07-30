import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_core_import_does_not_import_jax_or_af3():
    code = (
        "import sys, af3_hallucination, af3_hallucination.config, "
        "af3_hallucination.workflow; "
        "assert 'jax' not in sys.modules; assert 'alphafold3' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_public_typing_module_imports():
    subprocess.run(
        [sys.executable, "-c", "import af3_hallucination.typing"],
        check=True,
    )


def test_release_tree_has_no_private_paths_or_af4_placeholder():
    forbidden = (
        "/home/" + "wangty",
        "/Users/" + "lynx",
        "gpu" + "09",
        "alpha" + "fold4",
        "alpha" + "Fold4",
    )
    for path in ROOT.rglob("*"):
        parts = path.relative_to(ROOT).parts
        if not path.is_file() or any(
            part.startswith(".") or part.endswith(".egg-info") for part in parts
        ):
            continue
        if path.suffix in {".pyc", ".whl", ".gz"}:
            continue
        text = path.read_text(errors="ignore")
        for value in forbidden:
            assert value not in text, f"forbidden release string {value!r} in {path}"

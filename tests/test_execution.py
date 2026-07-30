import sys

import pytest

from af3_hallucination.execution import LocalExecutor, SSHExecutor


def test_local_executor():
    result = LocalExecutor().run([sys.executable, "-c", "print('ok')"])
    assert result.return_code == 0
    assert result.stdout.strip() == "ok"


def test_ssh_executor_rejects_whitespace_host():
    try:
        SSHExecutor("bad host")
    except ValueError:
        pass
    else:
        raise AssertionError("whitespace host was accepted")


def test_ssh_executor_rejects_option_host_and_invalid_environment_name():
    with pytest.raises(ValueError):
        SSHExecutor("-oProxyCommand=bad")
    executor = SSHExecutor("example")
    with pytest.raises(ValueError, match="environment variable"):
        executor.run(["true"], env={"BAD-NAME": "value"})

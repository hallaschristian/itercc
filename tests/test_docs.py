"""The Python snippets of README.md and llms.txt run as written (with a smaller
basis set): the usage an AI assistant copies from llms.txt must not go stale."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def snippets(name):
    text = (ROOT / name).read_text()
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    return [b for b in blocks if "cuda" not in b]          # GPU snippets need a device


@pytest.mark.parametrize("name", ["README.md", "llms.txt"])
def test_documented_snippets_run(name, capsys):
    pytest.importorskip("pyscf")
    code = "\n".join(snippets(name)).replace('"cc-pvdz"', '"6-31g"')
    assert "RestrictedCC" in code
    namespace = {}
    exec(compile(code, name, "exec"), namespace)
    assert namespace["ground"].converged
    if "result" in namespace:                               # llms.txt: CCSDTQ
        assert namespace["result"].converged
        assert namespace["result"].energy < namespace["ground"].energy

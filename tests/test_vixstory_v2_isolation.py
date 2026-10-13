"""The V2 gate path must not import Reader code or require provider credentials."""

import os
import subprocess
import sys
from pathlib import Path


def test_gate_imports_without_reader_or_api_credentials():
    root = Path(__file__).resolve().parents[1]
    script = '''
import importlib.abc
import sys
class NoReader(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.startswith("src.qa") or fullname in {
            "scripts.vixstory_reader_eval", "scripts.vixstory_cloud_diagnose",
            "src.pipeline.vixstory_worlds", "src.rl.vixstory_env",
        }):
            raise AssertionError("V2 attempted Reader/V1 import: " + fullname)
sys.meta_path.insert(0, NoReader())
from src.decision import schema_v2, vixstory_state_text_v2
from src.rl import vixstory_env_v2
from src.pipeline import vixstory_worlds_v2, vixstory_teacher_v2, vixstory_eval_v2, vixstory_arms_v2
from src.retrieval import dense_v2
from scripts import vixstory_v2_prepare, vixstory_v2_audit
assert schema_v2.ACTION_ORDER == ("STOP", "SEARCH_BM25", "SEARCH_DENSE")
print("V2 imports without Reader or credentials")
'''
    env = {key: value for key, value in os.environ.items()
           if not any(word in key.upper() for word in ("API_KEY", "TOKEN", "CREDENTIAL", "SECRET"))}
    result = subprocess.run([sys.executable, "-c", script], cwd=root, env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "without Reader or credentials" in result.stdout

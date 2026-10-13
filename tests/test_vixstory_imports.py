"""VixSTORY entry points remain independent of the retired ViQuAD pipeline."""

import os
import subprocess
import sys
from pathlib import Path


def test_vixstory_entrypoints_import_without_legacy_pipeline():
    root = Path(__file__).resolve().parents[1]
    script = '''
import importlib
import importlib.abc
import sys
from pathlib import Path

class NoLegacyPipeline(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.startswith("src.pipeline.")
                and not fullname.startswith("src.pipeline.vixstory_")
                and fullname != "src.pipeline.train_laya") or fullname in {
            "src.rl.env", "src.rl.reward", "src.decision.gliner_adapter",
            "src.data.loader", "src.data.artifacts", "src.retrieval.engine",
            "src.eval.correctness", "src.eval.rollout", "src.eval.splitting",
        }:
            raise AssertionError("VixSTORY imported retired code: " + fullname)

sys.meta_path.insert(0, NoLegacyPipeline())
for path in sorted(Path("scripts").glob("vixstory_*.py")):
    importlib.import_module("scripts." + path.stem)
from src.pipeline import train_laya
from src.eval import policies, reporting
from src.rl import reinforce
assert callable(train_laya.materialize_checkpoint)
assert policies.legal_actions([False, True, False, True]) == [1, 3]
assert callable(reporting.parse_laya_losses)
assert reinforce.discounted_returns([1.0, 2.0], 1.0).tolist() == [3.0, 2.0]
'''
    env = {key: value for key, value in os.environ.items()
           if not any(word in key.upper() for word in ("API_KEY", "TOKEN", "CREDENTIAL", "SECRET"))}
    env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", script], cwd=root, env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr

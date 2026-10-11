"""Guarded GPU scoring for the paired GLiNER/Laya pilot, using frozen upstream worlds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.decision.schema import ACTION_ORDER
from src.decision.state_text import state_hash
from src.utils.config import add_profile_args, config_from_args, load_config, resolve
from src.utils.hashing import sha256_file
from src.utils.io import atomic_write_json, read_jsonl

GLINER_BASE = Path(__file__).resolve().parents[2] / "outputs/experiments/2026-10-09-gliner-p0/hf/model"
REVISION = "e173bca1f0c4217d7a55b3b0c705d5ece8a0218d"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_profile_args(parser)
    parser.add_argument("--backend", choices=("laya", "gliner"), required=True)
    parser.add_argument("--part", choices=("states", "episodes"), required=True)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    output = resolve(cfg, "diagnostics") / f"{args.backend}_val_{args.part}.jsonl"
    identity = output.with_suffix(".identity.json")
    if output.exists() or identity.exists():
        raise FileExistsError(f"pilot score file already exists: {output}")
    if args.part == "states":
        source = resolve(cfg, "outputs").parent / "fixtures/val_states.jsonl"
        inputs = [(row["qid"], row["state_hash"], row["state"])
                  for row in read_jsonl(source)]
        if not inputs or any(state_hash(text) != digest for _, digest, text in inputs):
            raise ValueError("pilot validation states missing or hashes do not match")
    else:
        source = resolve(cfg, "outputs").parent / "fixtures/episode_requests.jsonl"
        inputs = [(row["qid"], row["state_hash"], row["state"])
                  for row in read_jsonl(source)]
        if not inputs or any(state_hash(text) != digest for _, digest, text in inputs):
            raise ValueError("pilot reachable states missing or hashes do not match")
    if args.backend == "laya":
        from src.decision.laya_adapter import LayaAdapter
        adapter = LayaAdapter.load(resolve(cfg, "models") / "laya_ce", device="cuda", batch_size=2)
    else:
        from src.decision.gliner_adapter import GLiNERAdapter
        adapter = GLiNERAdapter.load(GLINER_BASE,
            adapter_dir=resolve(cfg, "models") / "gliner_sft/final",
            revision=REVISION, device="cuda", batch_size=2)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            for start in range(0, len(inputs), 32):
                chunk = inputs[start:start + 32]
                probs = adapter.predict_probs([text for _, _, text in chunk])
                if probs.shape != (len(chunk), len(ACTION_ORDER)) or not np.all(np.isfinite(probs)):
                    raise ValueError("decision adapter returned invalid scores")
                for (qid, digest, _), vector in zip(chunk, probs):
                    handle.write(json.dumps({"qid": qid, "state_hash": digest,
                        "probs": vector.tolist()}, ensure_ascii=False) + "\n")
        result = {"backend": args.backend, "part": args.part,
                  "rows": len(inputs), "model_hash": adapter.model_hash(),
                  "output": str(output), "sha256": sha256_file(output)}
        atomic_write_json(identity, result)
        print(json.dumps(result))
    finally:
        adapter.close()


if __name__ == "__main__":
    main()

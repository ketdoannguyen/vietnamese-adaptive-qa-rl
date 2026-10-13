"""Reader cache identity shared by VixSTORY prediction caches."""

from __future__ import annotations

from typing import Sequence

from src.utils.hashing import sha256_json


def reader_cache_key(qid: str, visible_doc_ids: Sequence[str], reader_sha: str,
                     tokenizer_sha: str, inference_config_sha: str) -> str:
    """Key predictions by question, ordered evidence and reader configuration.

    Never key a reader prediction on bare ``qid``: different seen-document sets produce
    different candidates, and keying on qid alone silently mixes them.
    """
    return sha256_json({
        "qid": qid,
        "visible_doc_ids_ordered": list(visible_doc_ids),
        "reader_sha": reader_sha,
        "tokenizer_sha": tokenizer_sha,
        "reader_inference_config_sha": inference_config_sha,
    })

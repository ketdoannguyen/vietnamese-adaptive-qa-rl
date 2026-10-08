"""A tiny synthetic UIT-ViQuAD2.0-shaped raw dataset for the P1 tests.

It is *not* a copy of any real data: it is written here so the pipeline can be exercised
end to end (including its failure and recovery paths) on a few dozen rows. The fixture
deliberately contains:

* a positive with exact offsets, and a duplicate ``id`` of it (checksum-suffixed qid);
* a positive whose stored ``answer_start`` is one *inside* a repeated answer
  (recoverable, space-prefix rule);
* a positive whose answer occurs exactly once but at a different offset
  (recoverable, unique-match rule);
* a positive whose answer text is absent from the context (unrecoverable);
* impossible (negative) rows with ``plausible_answers``;
* an answerable row with ``answers=None`` (malformed) and a row with an empty question;
* an NFD + double-space context (display mapping must keep offsets valid);
* two titles that share the *same* context document (split isolation must not break);
* a title whose contexts are near-duplicates (cosine >= 0.90) of another title's
  contexts, so the near-duplicate group merge is exercised;
* an upstream ``validation`` split that overlaps ``train`` ids, and an upstream ``test``
  split that ships without labels (like the real dataset).

Offsets are computed from the text with ``str.index`` so the fixture can never encode a
hand-counted offset, except where a wrong offset is the point of the row (marked ``start=``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.hashing import sha256_file
from src.utils.io import atomic_write_json

ANSWERS_TYPE = pa.struct([("text", pa.list_(pa.string())), ("answer_start", pa.list_(pa.int64()))])
FEATURES = pa.schema([
    ("id", pa.string()), ("title", pa.string()), ("context", pa.string()),
    ("question", pa.string()), ("answers", ANSWERS_TYPE),
    ("is_impossible", pa.bool_()), ("plausible_answers", ANSWERS_TYPE),
])
REVISION = "fixture-rev"

# ------------------------------------------------------------------ shared texts

ALPHA_CTX = "Hà Nội là thủ đô của Việt Nam. Hà Nội nằm ở miền Bắc."
BETA_PHO = "Phở là món ăn truyền thống của Việt Nam, nước dùng được ninh từ xương bò trong nhiều giờ."
BETA_HA_LONG = ("Vịnh Hạ Long nằm ở tỉnh Quảng Ninh và được UNESCO công nhận là "
                "di sản thiên nhiên thế giới.")
BETA_HONG = "Sông Hồng bắt nguồn từ Trung Quốc và đổ ra biển Đông qua đồng bằng Bắc Bộ."
BETA_CAFE = "Cà phê được du nhập vào Việt Nam từ thế kỷ 19 và Đắk Lắk là vùng trồng lớn nhất."
#: NFD ("Hoà" decomposed) with a double space — display mapping must preserve offsets.
NFD_CTX = "Hoàng Sa  là quần đảo."


def qa(rid: str, title: str, context: str, question: str, answer: str | None = None,
       *, start: int | None = None, impossible: bool = False,
       plausible: str | None = None) -> dict[str, Any]:
    answers = None
    if answer is not None:
        at = context.index(answer) if start is None else int(start)
        answers = {"text": [answer], "answer_start": [at]}
    plaus = None if plausible is None else {"text": [plausible], "answer_start": [0]}
    return {"id": rid, "title": title, "context": context, "question": question,
            "answers": answers, "is_impossible": impossible, "plausible_answers": plaus}


# ------------------------------------------------------------------ raw train split

TRAIN_ROWS: list[dict[str, Any]] = [
    # -- Alpha: exact offsets, a duplicate id, both recovery rules, one failure
    qa("1", "Alpha", ALPHA_CTX, "Thủ đô của Việt Nam là gì?", "Hà Nội"),
    qa("1", "Alpha", ALPHA_CTX, "Thủ đô của Việt Nam là gì?", "Hà Nội"),  # duplicate id
    qa("2", "Alpha", "Việt Nam Việt Nam đẹp lắm.", "Từ nào được lặp lại?", "Việt Nam", start=8),
    qa("3", "Alpha", "Sông Mê Kông chảy qua nhiều nước.",
       "Con sông nào chảy qua nhiều nước?", "Mê Kông", start=11),
    qa("4", "Alpha", "Trời hôm nay rất đẹp.", "Thành phố nào lớn nhất?", "Sài Gòn", start=3),
    qa("5", "Alpha", "Truyện Kiều là tác phẩm nổi tiếng của văn học Việt Nam.",
       "Ai là tác giả của Truyện Kiều?", impossible=True, plausible="Nguyễn Du"),
    qa("6", "Alpha", "Việt Nam có nhiều ngọn núi cao ở phía bắc.",
       "Núi nào cao nhất Việt Nam?", impossible=True, plausible="Phan Xi Păng"),
    qa("7", "Alpha", NFD_CTX, "Quần đảo Hoàng Sa nằm ở đâu?", "Hoàng Sa"),
    qa("8", "Alpha", "Câu này thiếu đáp án vàng.", "Câu hỏi thiếu đáp án?"),  # answers=None
    qa("9", "Alpha", "Câu hỏi rỗng.", "", "Câu hỏi rỗng."),  # empty question
    # -- Beta: eight plain rows
    qa("20", "Beta", BETA_PHO, "Phở là món ăn của nước nào?", "Việt Nam"),
    qa("21", "Beta", BETA_HA_LONG, "Vịnh Hạ Long thuộc tỉnh nào?", "Quảng Ninh"),
    qa("22", "Beta", BETA_HONG, "Sông Hồng bắt nguồn từ đâu?", "Trung Quốc"),
    qa("23", "Beta", BETA_CAFE, "Vùng trồng cà phê lớn nhất Việt Nam?", "Đắk Lắk"),
    qa("24", "Beta", "Đà Lạt nổi tiếng với khí hậu mát mẻ và nhiều loài hoa ôn đới.",
       "Đà Lạt có bao nhiêu hồ?", impossible=True, plausible="Than Thở"),
    qa("25", "Beta", "Áo dài là trang phục truyền thống, thường được may bằng lụa hoặc gấm.",
       "Áo dài ra đời năm nào?", impossible=True, plausible="1744"),
    qa("26", "Beta", "Ruộng bậc thang Sa Pa thu hút nhiều du khách mỗi năm.",
       "Sa Pa có bao nhiêu dân?", impossible=True, plausible="60.000"),
    qa("27", "Beta", "Chợ Bến Thành được xây dựng từ năm 1912 và là biểu tượng của Sài Gòn.",
       "Chợ Bến Thành được xây dựng từ năm nào?", "1912"),
    # -- Gamma
    qa("40", "Gamma", "Trường Sa là quần đảo thuộc Biển Đông, gồm nhiều đảo nhỏ và bãi san hô.",
       "Trường Sa thuộc biển nào?", "Biển Đông"),
    qa("41", "Gamma", "Cốm làng Vòng là đặc sản của Hà Nội, làm từ lúa nếp non.",
       "Cốm làng Vòng làm từ gì?", "lúa nếp non"),
    qa("42", "Gamma", "Động Phong Nha nằm trong vườn quốc gia Phong Nha – Kẻ Bàng, tỉnh Quảng Bình.",
       "Động Phong Nha thuộc tỉnh nào?", "Quảng Bình"),
    qa("43", "Gamma", "Nhã nhạc cung đình Huế được công nhận là di sản phi vật thể.",
       "Nhã nhạc Huế có bao nhiêu nhạc công?", impossible=True, plausible="40"),
    qa("44", "Gamma", "Hồ Hoàn Kiếm gắn với truyền thuyết vua Lê Lợi trả gươm thần.",
       "Ai trả gươm thần ở hồ Hoàn Kiếm?", "Lê Lợi"),
    qa("45", "Gamma", "Bánh mì Việt Nam là món ăn đường phố nổi tiếng.",
       "Bánh mì có từ năm nào?", impossible=True, plausible="1958"),
    # -- Delta
    qa("60", "Delta", "Thăng Long tứ trấn gồm bốn ngôi đền trấn giữ bốn hướng của kinh thành.",
       "Thăng Long tứ trấn gồm mấy đền?", "bốn"),
    qa("61", "Delta", "Văn Miếu Quốc Tử Giám được xây dựng năm 1070 dưới thời vua Lý Thánh Tông.",
       "Văn Miếu được xây dựng năm nào?", "1070"),
    qa("62", "Delta", "Mộc bản triều Nguyễn là di sản tư liệu thế giới, hiện lưu giữ tại Đà Lạt.",
       "Mộc bản triều Nguyễn lưu giữ ở đâu?", "Đà Lạt"),
    qa("63", "Delta", "Hát xoan là di sản văn hóa phi vật thể của Phú Thọ.",
       "Hát xoan có bao nhiêu làn điệu?", impossible=True, plausible="ba"),
    qa("64", "Delta", "Cầu Thê Húc nằm trên hồ Hoàn Kiếm, sơn màu đỏ.",
       "Cầu Thê Húc sơn màu gì?", "đỏ"),
    qa("65", "Delta", "Đờn ca tài tử phát triển ở Nam Bộ từ cuối thế kỷ 19.",
       "Đờn ca tài tử có bao nhiêu nhạc cụ?", impossible=True, plausible="bốn"),
    # -- Epsilon: shares Alpha's exact context document under a different title
    qa("80", "Epsilon", ALPHA_CTX, "Hà Nội nằm ở miền nào?", "miền Bắc"),
    qa("81", "Epsilon", "Gió mùa đông bắc mang không khí lạnh về miền Bắc.",
       "Gió mùa đông bắc có bao nhiêu đợt?", impossible=True, plausible="ba"),
    # -- Zeta: near-duplicates (one word changed) of four Beta contexts
    qa("90", "Zeta", "Phở là món ăn truyền thống của Việt Nam, nước dùng được ninh từ xương bò "
                     "trong nhiều ngày.", "Nước dùng phở được ninh từ gì?", "xương bò"),
    qa("91", "Zeta", "Vịnh Hạ Long nằm ở tỉnh Quảng Ninh và được UNESCO công nhận là "
                     "di sản thiên nhiên nhân loại.", "Vịnh Hạ Long được UNESCO công nhận là gì?",
       "di sản thiên nhiên nhân loại"),
    qa("92", "Zeta", "Sông Hồng bắt nguồn từ Trung Quốc và chảy ra biển Đông qua đồng bằng Bắc Bộ.",
       "Sông Hồng đổ ra biển nào?", "biển Đông"),
    qa("93", "Zeta", "Cà phê được đưa vào Việt Nam từ thế kỷ 19 và Đắk Lắk là vùng trồng lớn nhất.",
       "Vùng trồng cà phê lớn nhất?", "Đắk Lắk"),
]

# --------------------------------------------------- upstream validation / test splits

VALIDATION_ROWS: list[dict[str, Any]] = [
    qa("1", "Alpha", ALPHA_CTX, "Thủ đô của Việt Nam là gì?", "Hà Nội"),          # overlaps train
    qa("2", "Alpha", "Việt Nam Việt Nam đẹp lắm.", "Từ nào được lặp lại?", "Việt Nam"),  # overlaps train
    qa("v3", "Beta", BETA_PHO, "Phở là gì?", "món ăn"),
    qa("v4", "Gamma", "Hồ Tây là hồ lớn nhất Hà Nội.", "Hồ nào lớn nhất Hà Nội?", "Hồ Tây"),
]

TEST_ROWS: list[dict[str, Any]] = [
    # like the real dataset: the upstream test split ships without labels
    qa("t1", "Alpha", ALPHA_CTX, "Thủ đô của Việt Nam là gì?"),
    qa("t2", "Beta", BETA_PHO, "Phở là món ăn của nước nào?"),
]


# ------------------------------------------------------------------ writers

def write_split(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=FEATURES), str(path), compression="zstd")
    return {
        "rows": len(rows),
        "impossible_rows": sum(1 for r in rows if r["is_impossible"]),
        "answerable_rows": sum(1 for r in rows if not r["is_impossible"]),
        "answers_field_none_rows": sum(1 for r in rows if r["answers"] is None),
        "features": {name: str(field.type) for name, field in zip(FEATURES.names, FEATURES)},
        "parquet": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def write_raw_fixture(raw_dir: str | Path, *, revision: str = REVISION) -> dict[str, Any]:
    """Write ``<raw_dir>/<revision>/{train,validation,test}.parquet`` + source_manifest.json."""
    raw_dir = Path(raw_dir)
    rev_dir = raw_dir / revision
    splits: dict[str, Any] = {}
    for name, rows in (("train", TRAIN_ROWS), ("validation", VALIDATION_ROWS), ("test", TEST_ROWS)):
        entry = write_split(rev_dir / f"{name}.parquet", rows)
        entry["parquet"] = str((rev_dir / f"{name}.parquet").relative_to(raw_dir))
        splits[name] = entry
    manifest = {
        "repo_id": "fixture/tiny-viquad",
        "resolved_revision": revision,
        "download_utc": "2026-01-01T00:00:00+00:00",
        "raw_dir": str(raw_dir),
        "splits": splits,
        "total_rows": sum(s["rows"] for s in splits.values()),
        "license_verified": False,
        "license_note": "synthetic fixture; no real data",
        "reused": False,
    }
    atomic_write_json(raw_dir / "source_manifest.json", manifest)
    return manifest


DEFAULT_CAPS = {"micro": 4, "tiny": 8, "full": None}


def fixture_config(tmp_path: Path, *, caps: dict[str, int | None] | None = None,
                   n_distractors: int = 3, negative_audit_size: int = 5) -> dict[str, Any]:
    """A cfg dict shaped like ``configs/base.yaml`` but pointing at ``tmp_path``."""
    tmp_path = Path(tmp_path)
    return {
        "seed": 42,
        "paths": {
            "data_raw": str(tmp_path / "raw"),
            "data_processed": str(tmp_path / "processed"),
            "data_diagnostics": str(tmp_path / "diagnostics"),
            "data_scenarios": str(tmp_path / "scenarios"),
            "runs": str(tmp_path / "runs"),
        },
        "data": {
            "hf_repo": "fixture/tiny-viquad",
            "revision": REVISION,
            "split_seed": 42,
            "strict_train_ratio": 0.70,
            "strict_val_ratio": 0.15,
            "n_distractors": n_distractors,
            "hard_negative_ratio": 0.5,
            "scenario_seed": 42,
            "negative_audit_size": negative_audit_size,
            "profile": "tiny",
            "caps": dict(caps or DEFAULT_CAPS),
        },
    }


def run_fixture_pipeline(tmp_path: Path, *, profile: str = "tiny",
                         caps: dict[str, int | None] | None = None, **overrides: Any):
    """Build the fixture, run the real P1 pipeline over it, return (manifest, dirs)."""
    from src.pipeline import audit_split
    from src.utils.runs import RunRegistry

    tmp_path = Path(tmp_path)
    cfg = fixture_config(tmp_path, caps=caps)
    cfg["data"].update(overrides)
    raw_dir = Path(cfg["paths"]["data_raw"])
    write_raw_fixture(raw_dir)
    registry = RunRegistry(tmp_path / "runs" / "index.jsonl")
    manifest = audit_split.run(
        cfg=cfg, profile=profile, raw_dir=raw_dir,
        processed_dir=Path(cfg["paths"]["data_processed"]),
        diagnostics_dir=Path(cfg["paths"]["data_diagnostics"]),
        registry=registry)
    return manifest, cfg

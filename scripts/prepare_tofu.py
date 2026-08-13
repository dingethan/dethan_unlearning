#!/usr/bin/env python3
"""Sanity-check local TOFU splits before Full TOFU LoRA SFT."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

from full_tofu_knowledge_injection import TofuQADataset, load_tofu_split, load_yaml_config, mask_stats
from full_tofu_knowledge_injection.data import convert_qa_to_tensors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "full_tofu_lora.yaml"))
    args = parser.parse_args()
    cfg = load_yaml_config(args.config)

    tofu_root = Path(cfg["data"]["tofu_root"])
    report = {"tofu_root": str(tofu_root), "splits": {}, "label_masking": {}, "integrity": {}}

    required = [
        "full",
        "forget10",
        "retain90",
        "forget10_perturbed",
        "retain_perturbed",
        "real_authors",
        "real_authors_perturbed",
        "world_facts",
        "world_facts_perturbed",
    ]
    for name in required:
        ds = load_tofu_split(tofu_root, name)
        sample0 = {}
        for k in ds.column_names:
            v = ds[0][k]
            sample0[k] = f"<list:{len(v)}>" if isinstance(v, list) else v
        report["splits"][name] = {"n": len(ds), "columns": list(ds.column_names), "sample0": sample0}
        print(f"[ok] {name}: n={len(ds)} cols={ds.column_names}")

    full = load_tofu_split(tofu_root, "full")
    f10 = load_tofu_split(tofu_root, "forget10")
    r90 = load_tofu_split(tofu_root, "retain90")

    def key(ex):
        return (ex["question"], ex["answer"])

    full_set = {key(x) for x in full}
    f10_set = {key(x) for x in f10}
    r90_set = {key(x) for x in r90}
    report["integrity"] = {
        "full": len(full_set),
        "forget10": len(f10_set),
        "retain90": len(r90_set),
        "intersection": len(f10_set & r90_set),
        "union_equals_full": (f10_set | r90_set) == full_set,
    }
    print("[ok] integrity:", report["integrity"])
    assert report["integrity"]["intersection"] == 0
    assert report["integrity"]["union_equals_full"]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["path"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tags = {
        "question_start_tag": cfg["data"]["question_start_tag"],
        "question_end_tag": cfg["data"]["question_end_tag"],
        "answer_tag": cfg["data"]["answer_tag"],
    }
    dataset = TofuQADataset(full, tokenizer, max_length=cfg["data"]["max_length"], **tags)
    stats = mask_stats(dataset, n=128)
    report["label_masking"] = stats

    q, a = full[0]["question"], full[0]["answer"]
    _, labels, _, meta = convert_qa_to_tensors(tokenizer, q, a, cfg["data"]["max_length"], **tags)
    supervised_ids = [int(x) for x in labels.tolist() if x != -100]
    supervised_text = tokenizer.decode(supervised_ids)
    report["example"] = {
        "question": q,
        "answer": a,
        "meta": meta,
        "supervised_decode_preview": supervised_text[:300],
        "supervised_starts_with_answer_tag": supervised_text.lstrip().startswith(tags["answer_tag"]),
    }
    print("[ok] label mask stats:", stats)
    print("[ok] supervised preview:", report["example"]["supervised_decode_preview"][:160])

    out = ROOT / "results" / "tofu_data_sanity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[✓] wrote {out}")


if __name__ == "__main__":
    main()

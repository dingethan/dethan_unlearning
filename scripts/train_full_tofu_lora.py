#!/usr/bin/env python3
"""Full TOFU LoRA SFT for knowledge injection (θ0 + Δθ_full).

This LoRA injects fictitious TOFU knowledge into the base model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from full_tofu_knowledge_injection import (  # noqa: E402
    TofuQADataset,
    custom_data_collator,
    load_tofu_split,
    load_yaml_config,
)


def discover_target_modules(model, requested):
    found = set()
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in requested and isinstance(module, torch.nn.Linear):
            found.add(leaf)
    missing = [m for m in requested if m not in found]
    if missing:
        raise RuntimeError(f"Requested LoRA target modules not found as Linear: {missing}")
    return sorted(found)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "trainable_percent": float(100.0 * trainable / total) if total else 0.0,
    }


def build_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Causal LM: pad on the right to keep answer positions aligned with official TOFU.
    tokenizer.padding_side = "right"
    return tokenizer


def build_model(cfg: Dict[str, Any], smoke: bool = False, init_adapter: str | None = None):
    model_cfg = cfg["model"]
    dtype = torch.bfloat16 if model_cfg.get("torch_dtype") == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["path"],
        dtype=dtype,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    requested = list(cfg["lora"]["target_modules"])
    targets = discover_target_modules(model, requested)
    print("[LoRA] discovered target_modules:", targets)

    if init_adapter:
        print(f"[LoRA] continue from existing adapter (read-only source): {init_adapter}")
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            r=int(cfg["lora"]["r"]),
            lora_alpha=int(cfg["lora"]["lora_alpha"]),
            lora_dropout=float(cfg["lora"]["lora_dropout"]),
            bias=cfg["lora"]["bias"],
            task_type=TaskType.CAUSAL_LM,
            target_modules=targets,
        )
        model = get_peft_model(model, lora_cfg)
    if cfg["training"].get("gradient_checkpointing", True):
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    model.print_trainable_parameters()
    return model, targets


def make_training_args(cfg: Dict[str, Any], smoke: bool) -> TrainingArguments:
    t = cfg["training"]
    if smoke:
        s = cfg["smoke"]
        output_dir = s["output_dir"]
        return TrainingArguments(
            output_dir=output_dir,
            max_steps=int(s["max_steps"]),
            per_device_train_batch_size=int(s["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(s["gradient_accumulation_steps"]),
            learning_rate=float(t["learning_rate"]),
            warmup_ratio=float(t["warmup_ratio"]),
            lr_scheduler_type=t["lr_scheduler_type"],
            weight_decay=float(t["weight_decay"]),
            max_grad_norm=float(t["max_grad_norm"]),
            bf16=bool(t["bf16"]),
            logging_steps=int(s["logging_steps"]),
            save_strategy="no",
            report_to="none",
            optim=t["optim"],
            seed=int(cfg["seed"]),
            remove_unused_columns=False,
            dataloader_num_workers=0,
            gradient_checkpointing=bool(t.get("gradient_checkpointing", True)),
        )

    run_dir = cfg["paths"]["run_dir"]
    effective = int(t["per_device_train_batch_size"]) * int(t["gradient_accumulation_steps"])
    print(
        f"[train] per_device_train_batch_size={t['per_device_train_batch_size']} "
        f"grad_accum={t['gradient_accumulation_steps']} effective_batch_size={effective}"
    )
    return TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=float(t["num_train_epochs"]),
        per_device_train_batch_size=int(t["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(t["gradient_accumulation_steps"]),
        learning_rate=float(t["learning_rate"]),
        warmup_ratio=float(t["warmup_ratio"]),
        lr_scheduler_type=t["lr_scheduler_type"],
        weight_decay=float(t["weight_decay"]),
        max_grad_norm=float(t["max_grad_norm"]),
        bf16=bool(t["bf16"]),
        logging_steps=int(t["logging_steps"]),
        logging_dir=cfg["paths"]["log_dir"],
        save_strategy=t.get("save_strategy", "epoch"),
        save_steps=int(t["save_steps"]) if t.get("save_steps") else 500,
        save_total_limit=int(t.get("save_total_limit", 2)),
        report_to=t.get("report_to", "none"),
        optim=t["optim"],
        seed=int(cfg["seed"]),
        remove_unused_columns=False,
        dataloader_num_workers=int(t.get("dataloader_num_workers", 2)),
        gradient_checkpointing=bool(t.get("gradient_checkpointing", True)),
    )


def save_manifest(cfg: Dict[str, Any], extra: Dict[str, Any]) -> None:
    path = Path(cfg["paths"]["manifest"])
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "experiment_name": cfg.get("experiment_name"),
        "base_model_path": cfg["model"]["path"],
        "tofu_root": cfg["data"]["tofu_root"],
        "split": cfg["data"]["split"],
        "prompt_template": {
            "question_start_tag": cfg["data"]["question_start_tag"],
            "question_end_tag": cfg["data"]["question_end_tag"],
            "answer_tag": cfg["data"]["answer_tag"],
        },
        "lora": cfg["lora"],
        "training": cfg["training"],
        **extra,
    }
    # merge with existing if present
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        old.update(payload)
        payload = old
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[✓] manifest -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "full_tofu_lora.yaml"))
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument(
        "--init_adapter",
        default=None,
        help="Load an existing LoRA adapter and continue training (fresh optimizer). "
        "Does not write back to this path.",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    init_adapter = args.init_adapter or cfg.get("init_adapter")
    set_seed(int(cfg["seed"]))
    os.makedirs(cfg["paths"]["run_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["log_dir"], exist_ok=True)

    tokenizer = build_tokenizer(cfg["model"]["path"])
    raw = load_tofu_split(cfg["data"]["tofu_root"], cfg["data"]["split"])
    print(f"[data] split={cfg['data']['split']} n={len(raw)}")

    train_ds = TofuQADataset(
        raw,
        tokenizer,
        max_length=int(cfg["data"]["max_length"]),
        question_start_tag=cfg["data"]["question_start_tag"],
        question_end_tag=cfg["data"]["question_end_tag"],
        answer_tag=cfg["data"]["answer_tag"],
    )

    model, targets = build_model(cfg, smoke=args.smoke_test, init_adapter=init_adapter)
    param_stats = count_parameters(model)
    print("[params]", param_stats)

    # One-batch forward/backward smoke before Trainer (memory / shape check)
    model.train()
    batch = custom_data_collator([train_ds[i] for i in range(min(2, len(train_ds)))])
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=bool(cfg["training"]["bf16"])):
        out = model(**batch)
        loss = out.loss
    loss.backward()
    model.zero_grad(set_to_none=True)
    mem_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else None
    print(
        f"[✓] one-batch forward/backward ok | loss={float(loss.detach()):.4f} | peak_mem_GB={mem_gb}"
    )

    training_args = make_training_args(cfg, smoke=args.smoke_test)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=custom_data_collator,
        processing_class=tokenizer,
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    metrics = train_result.metrics
    print("[train metrics]", metrics)

    save_manifest(
        cfg,
        {
            "mode": "smoke" if args.smoke_test else "full_train",
            "target_modules_resolved": targets,
            "param_stats": param_stats,
            "train_metrics": metrics,
            "peak_mem_GB": mem_gb,
            "package_versions": {
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "peft": __import__("peft").__version__,
                "datasets": __import__("datasets").__version__,
            },
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda": torch.version.cuda,
            "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
            "dataset_sample_count": len(raw),
        },
    )

    if args.smoke_test:
        smoke_dir = Path(cfg["smoke"]["output_dir"])
        smoke_dir.mkdir(parents=True, exist_ok=True)
        (smoke_dir / "smoke_metrics.json").write_text(
            json.dumps({"metrics": metrics, "param_stats": param_stats, "peak_mem_GB": mem_gb}, indent=2),
            encoding="utf-8",
        )
        print("[✓] LoRA smoke test completed (adapter not promoted to final outputs/)")
        return

    # Full run: save unmerged adapter for reproducibility
    adapter_dir = Path(cfg["paths"]["adapter_dir"])
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    (adapter_dir / "training_args.json").write_text(
        training_args.to_json_string(), encoding="utf-8"
    )
    print(f"[✓] adapter saved -> {adapter_dir}")


if __name__ == "__main__":
    main()

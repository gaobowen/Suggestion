from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from data_pipeline import build_training_dataset, collate_causal_batch


class ConsoleLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_local_process_zero or not logs or "loss" not in logs:
            return

        epoch = logs.get("epoch")
        learning_rate = logs.get("learning_rate")
        message = f"[train] step={state.global_step} loss={logs['loss']:.6f}"
        if learning_rate is not None:
            message += f" lr={learning_rate:.6e}"
        if epoch is not None:
            message += f" epoch={epoch:.4f}"
        print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test Trainer with data_pipeline dataset")
    parser.add_argument(
        "--text-file",
        type=str,
        default="dataset/fineweb-edu",
    )
    parser.add_argument("--tokenizer-path", type=str, default="qwen3_5")
    parser.add_argument("--dataset-type", type=str, choices=["parquet", "jsonl"], default="parquet")
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--binpack-window-size", type=int, default=1024)
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--logging-dir", type=str, default="output/runs")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = AutoConfig.from_pretrained("qwen3_5", trust_remote_code=True, local_files_only=True)
    config.dtype = torch.bfloat16
    config.text_config.use_cache = False  # kv cache 会占用显存，训练时关闭；推理时开启

    model = AutoModelForImageTextToText.from_config(
        config,
        dtype=torch.bfloat16,
        attn_implementation="sdpa", # transformers 还未兼容qwen3.5 fla2 会无限NaN，先用 sdpa 测试流程，后续更新 transformers 后改回 fla2
        trust_remote_code=True,
    )
    # if hasattr(model, "generation_config") and model.generation_config is not None:
    #     model.generation_config.use_cache = False

    train_dataset, tokenizer = build_training_dataset(
        text_file=args.text_file,
        seq_len=args.seq_len,
        tokenizer_path=args.tokenizer_path,
        min_seq_len=2,
        text_column=args.text_column,
        dataset_type=args.dataset_type,
        streaming=True,
        binpack_window_size=args.binpack_window_size,
    )

    if hasattr(train_dataset, "__len__") and len(train_dataset) == 0:
        raise ValueError("Training dataset is empty. Check --text-file and --text-column.")

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must provide pad_token_id or eos_token_id")
    

    '''
    stage-1: 通用数据 total_tokens=3T, lr=3e-4,wsd,seq_len=4K 保持每次更新保证 1M 以上 tokens。
    stage-2: 高质量 数学和代码 STEM 数据集, total_tokens=3T, 学习率退火至3e-5, seq_len=16K。
    stage-3: ProLong-64K 数据集上训练，序列长度为 64K，学习率退火至 3e-5，训练 20B tokens，开启 ZeRO-3。
    '''
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        accelerator_config={
            "dispatch_batches": False,
            "split_batches": False,
        },
        learning_rate=5e-5,
        weight_decay=0.1,
        warmup_steps=1000,
        adam_beta2=0.98,
        lr_scheduler_type="linear",
        max_steps=80_000, # 这里显示单卡步数进度，计算token时需要根据 GPU 数量估算
        per_device_train_batch_size=1,
        num_train_epochs=1,
        gradient_accumulation_steps=64,
        gradient_checkpointing=True,
        max_grad_norm=0.5,
        save_steps=4_000,
        save_total_limit=2,
        bf16=True,
        torch_compile=False,
        dataloader_num_workers=4,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
        logging_nan_inf_filter=False,
        logging_first_step=True,
        logging_steps=1,
        report_to=["tensorboard"],
        remove_unused_columns=False,
    )

    data_collator = lambda batch: collate_causal_batch(
        batch=batch,
        pad_token_id=pad_token_id,
        use_block_diag_mask=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[ConsoleLossCallback()],
    )

    latest_checkpoint = get_last_checkpoint(args.output_dir)
    if latest_checkpoint:
        print(f"Resuming from checkpoint: {latest_checkpoint}")
    else:
        print("No checkpoint found. Starting training from scratch.")

    result = trainer.train(resume_from_checkpoint=latest_checkpoint)
    print("train_runtime:", result.metrics.get("train_runtime"))
    print("train_loss:", result.metrics.get("train_loss"))


if __name__ == "__main__":
    main()
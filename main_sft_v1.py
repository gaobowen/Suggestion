from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, TrainingArguments
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer

from model_v1 import MiniMindForCausalLM
from data_pipeline_sft import build_sft_train_dataloader


class SFTTrainerWithCustomDataloader(SFTTrainer):
    def __init__(self, *args, train_dataloader=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._external_train_dataloader = train_dataloader

    def get_train_dataloader(self):
        if self._external_train_dataloader is not None:
            return self._external_train_dataloader
        return super().get_train_dataloader()


class ConsoleLossCallback(TrainerCallback):
    def __init__(self, run_name: str = "main_sft_v1"):
        self.writer: SummaryWriter | None = None
        self.run_name = run_name.strip()

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_local_process_zero and self.writer is None:
            log_dir = Path(args.logging_dir)
            if self.run_name:
                log_dir = log_dir / self.run_name
            self.writer = SummaryWriter(log_dir=str(log_dir))

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_local_process_zero or not logs or "loss" not in logs:
            return

        if self.writer is not None:
            step = int(state.global_step)
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"train/{key}", float(value), step)
            self.writer.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TRL SFT training with custom SFT dataloader")
    parser.add_argument("--train-text-file", type=str, default="dataset/minimind/sft_t2t.jsonl")
    parser.add_argument("--dataset-type", type=str, choices=["parquet", "jsonl", "json"], default="jsonl")
    parser.add_argument("--text-column", type=str, default="text")

    parser.add_argument("--model-path", type=str, default="output_v1/checkpoint-21000")
    parser.add_argument("--tokenizer-path", type=str, default="qwen3_5")
    parser.add_argument("--seq-len", type=int, default=4096)

    parser.add_argument("--output-dir", type=str, default="output_sft_v1")
    parser.add_argument("--logging-dir", type=str, default="output_sft_v1/runs")

    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--logging-steps", type=int, default=4)

    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--streaming", action="store_true", default=True)
    parser.add_argument("--tb-run-name", type=str, default="main_sft_v1")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = MiniMindForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    model.config.use_cache = False
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    train_dataset, train_dataloader = build_sft_train_dataloader(
        source_path=args.train_text_file,
        dataset_type=args.dataset_type,
        tokenizer=tokenizer,
        max_seq_len=args.seq_len,
        text_column=args.text_column,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.dataloader_num_workers,
        streaming=args.streaming,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=True,
        torch_compile=False,
        dataloader_num_workers=0,
        dataloader_persistent_workers=False,
        dataloader_pin_memory=True,
        logging_nan_inf_filter=False,
        logging_first_step=True,
        logging_steps=args.logging_steps,
        report_to=["tensorboard"],
        remove_unused_columns=False,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "train_dataloader": train_dataloader,
    }

    try:
        trainer = SFTTrainerWithCustomDataloader(
            processing_class=tokenizer,
            callbacks=[ConsoleLossCallback(run_name=args.tb_run_name)],
            **trainer_kwargs,
        )
    except TypeError:
        trainer = SFTTrainerWithCustomDataloader(
            tokenizer=tokenizer,
            callbacks=[ConsoleLossCallback(run_name=args.tb_run_name)],
            **trainer_kwargs,
        )

    latest_checkpoint = get_last_checkpoint(args.output_dir)
    if latest_checkpoint:
        print(f"Resuming from checkpoint: {latest_checkpoint}")
    else:
        print("No checkpoint found. Starting SFT from scratch.")

    result = trainer.train(resume_from_checkpoint=latest_checkpoint)
    print("train_runtime:", result.metrics.get("train_runtime"))
    print("train_loss:", result.metrics.get("train_loss"))


if __name__ == "__main__":
    main()

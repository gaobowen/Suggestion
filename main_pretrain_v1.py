from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from data_pipeline import build_training_dataset, collate_causal_batch
from data_pipeline_simple import build_pretrain_dataset_simple, collate_causal_batch_simple
from model_v1 import MiniMindConfig, MiniMindForCausalLM


def load_checkpoint_vocab_size(checkpoint_dir: str | Path) -> int | None:
    config_path = Path(checkpoint_dir) / "config.json"
    if not config_path.exists():
        return None

    with config_path.open("r", encoding="utf-8") as handle:
        config_data = json.load(handle)

    vocab_size = config_data.get("vocab_size")
    return int(vocab_size) if isinstance(vocab_size, int) else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain MiniMind v1 with Hugging Face Trainer")
    parser.add_argument("--text-file", type=str, default="dataset/minimind/pretrain_t2t_mini.jsonl")
    parser.add_argument("--tokenizer-path", type=str, default="qwen3_5")
    parser.add_argument("--dataset-type", type=str, choices=["parquet", "jsonl"], default="jsonl")
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--binpack-window-size", type=int, default=1024)
    parser.add_argument("--output-dir", type=str, default="output_v1")
    parser.add_argument("--logging-dir", type=str, default="output_v1/runs")

    
    parser.add_argument("--num-hidden-layers", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--head_dim", type=int, default=256)
    parser.add_argument("--num-attention-heads", type=int, default=8)
    parser.add_argument("--num-key-value-heads", type=int, default=4)
    parser.add_argument("--intermediate-size", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-position-embeddings", type=int, default=4096) # 可以随着seq-len训练进度增加
    parser.add_argument("--use-moe", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--num-experts-per-tok", type=int, default=1)
    parser.add_argument("--moe-intermediate-size", type=int, default=0)
    parser.add_argument("--router-aux-loss-coef", type=float, default=5e-4)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine")
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--adam-beta2", type=float, default=0.98)

    parser.add_argument("--per-device-train-batch-size", type=int, default=20)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=80_000)
    
    parser.add_argument("--save-steps", type=int, default=1_000)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--use-block-diag-mask", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--simple-data-pipeline", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--attn-implementation", type=str, choices=["sdpa", "flash"], default="sdpa")
    parser.add_argument("--tb-run-name", type=str, default="main_pretrain_v1")


    return parser.parse_args()


class ConsoleLossCallback(TrainerCallback):
    def __init__(self, run_name: str = "main_pretrain_v1"):
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

        # epoch = logs.get("epoch")
        # learning_rate = logs.get("learning_rate")
        # message = f"[train] step={state.global_step} loss={logs['loss']:.6f}"
        # if learning_rate is not None:
        #     message += f" lr={learning_rate:.6e}"
        # if epoch is not None:
        #     message += f" epoch={epoch:.4f}"
        # if state.global_step < 20:
        #     print(message, flush=True)

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

def main() -> None:
    args = parse_args()

    if args.simple_data_pipeline:
        train_dataset, tokenizer = build_pretrain_dataset_simple(
            text_file=args.text_file,
            seq_len=args.seq_len,
            tokenizer_path=args.tokenizer_path,
            text_column=args.text_column,
            dataset_type=args.dataset_type,
            min_seq_len=2,
            streaming=True,
        )
    else:
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
    eos_token_id = tokenizer.eos_token_id
    if pad_token_id is None or eos_token_id is None:
        raise ValueError("Tokenizer must provide pad_token_id or eos_token_id")
    tokenizer_size = len(tokenizer) + 16
    assert tokenizer_size >= len(tokenizer), f"Tokenizer vocab size ({len(tokenizer)}) exceeds the hardcoded tokenizer_size ({tokenizer_size}). Update tokenizer_size or use a smaller tokenizer."

    print(
        f"Using pad_token_id={pad_token_id}, eos_token_id={eos_token_id}, "
        f"tokenizer.vocab_size={tokenizer.vocab_size}, len(tokenizer)={tokenizer_size}"
    )

    assert args.max_position_embeddings >= args.seq_len, "max_position_embeddings must be greater than or equal to seq_len"

    config_kwargs: dict[str, int | float | bool] = {
        "vocab_size": int(tokenizer_size),
        "bos_token_id": int(eos_token_id),
        "eos_token_id": int(eos_token_id),
        "num_attention_heads": args.num_attention_heads,
        "num_key_value_heads": args.num_key_value_heads,
        "dropout": args.dropout,
        "max_position_embeddings": args.max_position_embeddings,
        "flash_attn": args.attn_implementation == "flash",
        "use_moe": args.use_moe,
        "num_experts": args.num_experts,
        "num_experts_per_tok": args.num_experts_per_tok,
        "router_aux_loss_coef": args.router_aux_loss_coef,
    }
    if args.intermediate_size > 0:
        config_kwargs["intermediate_size"] = args.intermediate_size
    if args.moe_intermediate_size > 0:
        config_kwargs["moe_intermediate_size"] = args.moe_intermediate_size

    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        **config_kwargs,
    )
    model = MiniMindForCausalLM(config)
    model.config.use_cache = False
    model.config.pad_token_id = int(pad_token_id)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        accelerator_config={
            "dispatch_batches": False,
            "split_batches": False,
        },
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        adam_beta2=args.adam_beta2,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=args.max_grad_norm,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=True,
        torch_compile=False,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_persistent_workers=args.dataloader_num_workers > 0,
        dataloader_pin_memory=True,
        logging_nan_inf_filter=False,
        logging_first_step=True,
        logging_steps=1,
        remove_unused_columns=False,
    )

    if args.simple_data_pipeline:
        data_collator = lambda batch: collate_causal_batch_simple(
            batch=batch,
            pad_token_id=int(pad_token_id),
        )
    else:
        data_collator = lambda batch: collate_causal_batch(
            batch=batch,
            pad_token_id=int(pad_token_id),
            use_block_diag_mask=args.use_block_diag_mask,
        )
    


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[ConsoleLossCallback(run_name=args.tb_run_name)],
    )

    latest_checkpoint = get_last_checkpoint(args.output_dir)
    if latest_checkpoint:
        checkpoint_vocab_size = load_checkpoint_vocab_size(latest_checkpoint)
        if checkpoint_vocab_size is not None and checkpoint_vocab_size != tokenizer_size:
            raise ValueError(
                f"Checkpoint vocab_size={checkpoint_vocab_size} is incompatible with current tokenizer length={tokenizer_size}. "
                "This checkpoint was created with an undersized embedding table. Start from a fresh output_dir or remove the old checkpoint before resuming."
            )
        print(f"Resuming from checkpoint: {latest_checkpoint}")
    else:
        print("No checkpoint found. Starting training from scratch.")

    result = trainer.train(resume_from_checkpoint=latest_checkpoint)
    print("train_runtime:", result.metrics.get("train_runtime"))
    print("train_loss:", result.metrics.get("train_loss"))


if __name__ == "__main__":
    main()

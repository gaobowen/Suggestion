from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from transformers import (
	AutoConfig,
	AutoModelForCausalLM,
	AutoTokenizer,
	TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="SFT training with TRL SFTTrainer")
	parser.add_argument(
		"--text-file",
		type=str,
		default="chat_test_samples.json",
	)
	parser.add_argument("--dataset-type", type=str, choices=["parquet", "jsonl"], default="jsonl")
	parser.add_argument("--text-column", type=str, default="text")
	parser.add_argument("--model-path", type=str, default="qwen3_5")
	parser.add_argument("--tokenizer-path", type=str, default="qwen3_5")
	parser.add_argument("--seq-len", type=int, default=4096)
	parser.add_argument("--output-dir", type=str, default="output_sft")
	parser.add_argument("--logging-dir", type=str, default="output_sft/runs")

	return parser.parse_args()


def _resolve_data_files(text_file: str, suffix: str) -> str:
	path = Path(text_file)
	if path.is_dir():
		return str(path / "**" / f"*.{suffix}")
	return str(path)


def load_sft_dataset(text_file: str, dataset_type: str) -> Dataset:
	if dataset_type == "parquet":
		data_files = _resolve_data_files(text_file, "parquet")
		dataset = load_dataset("parquet", data_files=data_files, split="train")
	elif dataset_type == "jsonl":
		data_files = _resolve_data_files(text_file, "jsonl")
		dataset = load_dataset("json", data_files=data_files, split="train")
	else:
		raise ValueError(f"Unsupported dataset_type: {dataset_type}")
	return dataset


def build_formatting_func(tokenizer: Any, text_column: str):
	def _format(example: dict[str, Any]) -> str:
		messages = example.get("messages")
		if isinstance(messages, list) and messages:
			return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

		prompt = example.get("prompt")
		completion = example.get("completion")
		if prompt is not None and completion is not None:
			msg = [
				{"role": "user", "content": str(prompt)},
				{"role": "assistant", "content": str(completion)},
			]
			return tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)

		instruction = example.get("instruction")
		output = example.get("output")
		if instruction is not None and output is not None:
			user_content = str(instruction)
			input_text = example.get("input")
			if input_text:
				user_content = f"{user_content}\n{input_text}"
			msg = [
				{"role": "user", "content": user_content},
				{"role": "assistant", "content": str(output)},
			]
			return tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)

		raw_text = example.get(text_column)
		if raw_text is None:
			raise ValueError(
				f"Cannot format sample. Expected one of: messages/prompt+completion/instruction+output/{text_column}."
			)
		return str(raw_text)

	return _format


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

	config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
	config.use_cache = False
	if hasattr(config, "text_config") and hasattr(config.text_config, "use_cache"):
		config.text_config.use_cache = False

	model = AutoModelForCausalLM.from_pretrained(
		args.model_path,
		config=config,
		torch_dtype=torch.bfloat16,
		attn_implementation="flash_attention_2",
		trust_remote_code=True,
		local_files_only=True,
	)
	model.config.use_cache = False
	if tokenizer.pad_token_id is not None:
		model.config.pad_token_id = tokenizer.pad_token_id

	train_dataset = load_sft_dataset(
		text_file=args.text_file,
		dataset_type=args.dataset_type,
	)
	if len(train_dataset) == 0:
		raise ValueError("Training dataset is empty. Check --text-file and dataset contents.")

	formatting_func = build_formatting_func(tokenizer=tokenizer, text_column=args.text_column)

	training_args = TrainingArguments(
		output_dir=args.output_dir,
		logging_dir=args.logging_dir,
		learning_rate=2e-5,
		weight_decay=0.1,
		warmup_steps=200,
		max_steps=10_000,
		per_device_train_batch_size=1,
		num_train_epochs=1,
		gradient_accumulation_steps=16,
		gradient_checkpointing=True,
		save_steps=1_000,
		save_total_limit=2,
		bf16=True,
		torch_compile=False,
		dataloader_num_workers=4,
		dataloader_persistent_workers=True,
		dataloader_pin_memory=True,
		logging_steps=50,
		report_to=["tensorboard"],
		remove_unused_columns=False,
	)

	try:
		trainer = SFTTrainer(
			model=model,
			args=training_args,
			train_dataset=train_dataset,
			processing_class=tokenizer,
			formatting_func=formatting_func,
			max_length=args.seq_len,
			packing=False,
		)
	except TypeError:
		# Backward compatibility for older TRL versions.
		trainer = SFTTrainer(
			model=model,
			args=training_args,
			train_dataset=train_dataset,
			tokenizer=tokenizer,
			formatting_func=formatting_func,
			max_seq_length=args.seq_len,
			packing=False,
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

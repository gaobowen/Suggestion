from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset as TorchIterableDataset
from torch.utils.data import get_worker_info
from transformers import AutoTokenizer, GPT2TokenizerFast


def find_files_with_suffixes(paths: list[Path], suffixes: tuple[str, ...]) -> list[str]:
	normalized_suffixes = tuple(suffix.lower() for suffix in suffixes)
	files: list[str] = []

	for path in paths:
		if path.is_file() and path.suffix.lower() in normalized_suffixes:
			files.append(str(path))
			continue
		if path.is_dir():
			for suffix in normalized_suffixes:
				files.extend(str(file_path) for file_path in sorted(path.rglob(f"*{suffix}")))

	if not files:
		joined_suffixes = ", ".join(normalized_suffixes)
		raise FileNotFoundError(f"No files found with suffixes: {joined_suffixes}")
	return files


def load_hf_tokenizer(tokenizer_path: str) -> Any:
	tk_path = Path(tokenizer_path)
	if not tk_path.exists():
		raise FileNotFoundError(f"Tokenizer path not found: {tk_path}")

	try:
		tokenizer = AutoTokenizer.from_pretrained(
			str(tk_path),
			trust_remote_code=True,
			local_files_only=True,
			use_fast=True,
		)
	except Exception:
		vocab_file = tk_path / "vocab.json"
		merges_file = tk_path / "merges.txt"
		if not vocab_file.exists() or not merges_file.exists():
			raise ValueError("Failed to load tokenizer with AutoTokenizer and missing vocab/merges fallback files")
		tokenizer = GPT2TokenizerFast(vocab_file=str(vocab_file), merges_file=str(merges_file))

	if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
		tokenizer.pad_token = tokenizer.eos_token
	return tokenizer


class JSONLTextIterableDataset(TorchIterableDataset):
	def __init__(self, data_files: list[str], text_column: str):
		self.data_files = data_files
		self.text_column = text_column

	def __iter__(self) -> Iterator[dict[str, str]]:
		worker_info = get_worker_info()
		worker_id = worker_info.id if worker_info else 0
		num_workers = worker_info.num_workers if worker_info else 1

		rank = 0
		world_size = 1
		if torch.distributed.is_available() and torch.distributed.is_initialized():
			rank = torch.distributed.get_rank()
			world_size = torch.distributed.get_world_size()

		num_shards = world_size * num_workers
		shard_id = rank * num_workers + worker_id

		sample_index = 0
		for file_path in self.data_files:
			with open(file_path, "r", encoding="utf-8") as f:
				for line in f:
					line = line.strip()
					if not line:
						continue

					if sample_index % num_shards != shard_id:
						sample_index += 1
						continue
					sample_index += 1

					try:
						row = json.loads(line)
					except Exception:
						continue

					text = row.get(self.text_column)
					if isinstance(text, str) and text.strip():
						yield {self.text_column: text}


def load_hf_text_dataset(
	source_path: Path,
	text_column: str,
	dataset_type: str,
	*,
	streaming: bool,
) -> Iterable[dict[str, Any]]:
	if dataset_type == "jsonl":
		data_files = find_files_with_suffixes([source_path], (".jsonl",))
		return JSONLTextIterableDataset(data_files=data_files, text_column=text_column)

	if dataset_type == "parquet":
		data_files = find_files_with_suffixes([source_path], (".parquet",))
		dataset = load_dataset("parquet", data_files=data_files, split="train", streaming=streaming)
		return dataset.select_columns([text_column])

	raise ValueError("dataset_type must be one of: parquet, jsonl")


class SimpleTokenChunkIterableDataset(TorchIterableDataset):
	def __init__(
		self,
		source_iterable: Iterable[dict[str, Any]],
		tokenizer: Any,
		seq_len: int,
		text_column: str,
		min_seq_len: int,
	):
		self.source_iterable = source_iterable
		self.tokenizer = tokenizer
		self.seq_len = seq_len
		self.text_column = text_column
		self.min_seq_len = max(2, min_seq_len)

	def __iter__(self) -> Iterator[dict[str, list[int]]]:
		eos_id = getattr(self.tokenizer, "eos_token_id", None)
		if eos_id is None:
			raise ValueError("Tokenizer must provide eos_token_id")

		for row in self.source_iterable:
			text = row.get(self.text_column)
			if not isinstance(text, str) or not text.strip():
				continue

			token_ids = self.tokenizer.encode(text, add_special_tokens=False)
			if not token_ids:
				continue

			token_ids = token_ids + [eos_id]
			for start in range(0, len(token_ids), self.seq_len):
				chunk = token_ids[start : start + self.seq_len]
				if len(chunk) < self.min_seq_len:
					continue
				yield {"input_ids": chunk}


def build_pretrain_dataset_simple(
	text_file: str,
	seq_len: int,
	tokenizer_path: str,
	text_column: str,
	dataset_type: str,
	*,
	min_seq_len: int = 2,
	streaming: bool = True,
) -> tuple[SimpleTokenChunkIterableDataset, Any]:
	if not text_file:
		raise ValueError("text_file is required")
	if not tokenizer_path:
		raise ValueError("tokenizer_path is required")

	source_path = Path(text_file)
	tokenizer = load_hf_tokenizer(tokenizer_path)
	source_dataset = load_hf_text_dataset(
		source_path=source_path,
		text_column=text_column,
		dataset_type=dataset_type,
		streaming=streaming,
	)

	dataset = SimpleTokenChunkIterableDataset(
		source_iterable=source_dataset,
		tokenizer=tokenizer,
		seq_len=seq_len,
		text_column=text_column,
		min_seq_len=min_seq_len,
	)
	return dataset, tokenizer


def collate_causal_batch_simple(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, torch.Tensor]:
	if not batch:
		raise ValueError("batch is empty")

	max_len = max(len(item["input_ids"]) for item in batch)
	batch_size = len(batch)

	input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
	labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
	attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

	for i, item in enumerate(batch):
		ids = item["input_ids"]
		ids_tensor = ids if isinstance(ids, torch.Tensor) else torch.tensor(ids, dtype=torch.long)
		length = int(ids_tensor.numel())
		input_ids[i, :length] = ids_tensor
		labels[i, :length] = ids_tensor
		attention_mask[i, :length] = 1

	return {
		"input_ids": input_ids,
		"labels": labels,
		"attention_mask": attention_mask,
	}


def build_pretrain_dataloader_simple(
	dataset: Any,
	batch_size: int,
	*,
	pad_token_id: int,
	num_workers: int = 0,
) -> DataLoader:
	is_iterable = isinstance(dataset, TorchIterableDataset)
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=(not is_iterable),
		drop_last=False,
		num_workers=num_workers,
		collate_fn=lambda batch: collate_causal_batch_simple(batch=batch, pad_token_id=pad_token_id),
	)

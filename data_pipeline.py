import argparse
import binpacking
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
        elif path.is_dir():
            for suffix in normalized_suffixes:
                files.extend(str(file_path) for file_path in sorted(path.rglob(f"*{suffix}")))

    if not files:
        joined_suffixes = ", ".join(normalized_suffixes)
        raise FileNotFoundError(f"No files found with suffixes: {joined_suffixes}")
    return files


def find_parquet_files(paths: list[Path]) -> list[str]:
    return find_files_with_suffixes(paths, (".parquet",))


def find_jsonl_files(paths: list[Path]) -> list[str]:
    return find_files_with_suffixes(paths, (".jsonl",))


def find_dataset_files(paths: list[Path], dataset_type: str) -> list[str]:
    if dataset_type == "parquet":
        return find_files_with_suffixes(paths, (".parquet",))
    if dataset_type == "jsonl":
        return find_files_with_suffixes(paths, (".jsonl",))
    raise ValueError("dataset_type must be one of: parquet, jsonl")


def split_token_blocks(token_ids: list[int], block_size: int, min_seq_len: int) -> list[list[int]]:
    min_seq_len = max(2, min_seq_len)
    blocks = [token_ids[i : i + block_size] for i in range(0, len(token_ids), block_size)]
    return [block for block in blocks if min_seq_len <= len(block) <= block_size]


def pack_units_with_binpacking(
    pack_units: list[list[int]],
    block_size: int,
    min_seq_len: int,
) -> list[dict[str, list[int]]]:
    min_seq_len = max(2, min_seq_len)
    valid_units = [unit for unit in pack_units if min_seq_len <= len(unit) <= block_size]
    if not valid_units:
        return []

    lengths = {index: len(unit) for index, unit in enumerate(valid_units)}
    packed_bins = binpacking.to_constant_volume(lengths, block_size)

    samples: list[dict[str, list[int]]] = []
    for packed_bin in packed_bins:
        unit_indices = list(packed_bin.keys()) if isinstance(packed_bin, dict) else list(packed_bin)
        input_ids: list[int] = []
        position_ids: list[int] = []

        for _, unit_index in enumerate(unit_indices):
            unit = valid_units[int(unit_index)]
            input_ids.extend(unit)
            position_ids.extend(range(len(unit)))

        if len(input_ids) < min_seq_len:
            continue

        samples.append(
            {
                "input_ids": input_ids,
                "labels": input_ids[:],
                "position_ids": position_ids,
            }
        )

    return samples


def pad_sample_to_block_size(
    sample: dict[str, list[int]],
    block_size: int,
    pad_token_id: int,
) -> dict[str, list[int]]:
    input_ids = sample["input_ids"][:block_size]
    labels = sample.get("labels", input_ids)[:block_size]
    position_ids = sample.get("position_ids", list(range(len(input_ids))))[:block_size]

    length = len(input_ids)
    if length < block_size:
        pad_len = block_size - length
        input_ids = input_ids + [pad_token_id] * pad_len
        labels = labels + [-100] * pad_len
        position_ids = position_ids + [-1] * pad_len

    attention_mask = [1] * length + [0] * max(0, block_size - length)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
    }


def build_block_diagonal_attention_mask(
    position_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.BoolTensor:
    if position_ids.dim() != 2:
        raise ValueError("position_ids must be 2D: [batch, seq]")

    batch_size, seq_len = position_ids.shape
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=position_ids.device))
    causal = causal.unsqueeze(0).expand(batch_size, seq_len, seq_len)

    valid = position_ids.ge(0)
    if attention_mask is not None:
        valid = valid & attention_mask.bool()

    # Infer segment identity from position resets (position_ids == 0 starts a new segment).
    segment_start = (position_ids == 0) & valid
    segment_index = torch.cumsum(segment_start.long(), dim=1) - 1
    segment_index = torch.where(valid, segment_index, torch.full_like(segment_index, -1))

    same_segment = segment_index.unsqueeze(2) == segment_index.unsqueeze(1)

    pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
    return causal & same_segment & pair_valid


class HFTokenBlockIterableDataset(TorchIterableDataset):
    def __init__(
        self,
        source_iterable: Iterable[dict[str, Any]],
        tokenizer: Any,
        seq_len: int,
        text_column: str,
        min_seq_len: int,
        binpack_window_size: int,
        pad_token_id: int,
    ):
        self.source_iterable = source_iterable
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.text_column = text_column
        self.min_seq_len = max(2, min_seq_len)
        self.binpack_window_size = max(1, binpack_window_size)
        self.pad_token_id = pad_token_id

    def __iter__(self) -> Iterator[dict[str, list[int]]]:
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is None:
            raise ValueError("Tokenizer must provide eos_token_id")

        binpack_window: list[list[int]] = []

        for row in self.source_iterable:
            text = row.get(self.text_column)
            if not text or not isinstance(text, str) or not text.strip():
                continue

            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not token_ids:
                continue

            token_ids = token_ids + [eos_id]
            if len(token_ids) <= self.seq_len:
                binpack_window.append(token_ids)
            else:
                binpack_window.extend(
                    split_token_blocks(
                        token_ids,
                        block_size=self.seq_len,
                        min_seq_len=self.min_seq_len,
                    )
                )

            if len(binpack_window) >= self.binpack_window_size:
                packed_samples = pack_units_with_binpacking(
                    pack_units=binpack_window,
                    block_size=self.binpack_window_size,
                    min_seq_len=self.min_seq_len,
                )
                for sample in packed_samples:
                    yield pad_sample_to_block_size(
                        sample=sample,
                        block_size=self.binpack_window_size,
                        pad_token_id=self.pad_token_id,
                    )
                binpack_window = []

        final_packed_samples = pack_units_with_binpacking(
            pack_units=binpack_window,
            block_size=self.binpack_window_size,
            min_seq_len=self.min_seq_len,
        )
        for sample in final_packed_samples:
            yield pad_sample_to_block_size(
                sample=sample,
                block_size=self.binpack_window_size,
                pad_token_id=self.pad_token_id,
            )


class HFTokenChunkIterableDataset(TorchIterableDataset):
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
            if not text or not isinstance(text, str) or not text.strip():
                continue

            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not token_ids:
                continue

            token_ids = token_ids + [eos_id]
            for block in split_token_blocks(
                token_ids,
                block_size=self.seq_len,
                min_seq_len=self.min_seq_len,
            ):
                yield {"input_ids": block}


class JSONLTextIterableDataset(TorchIterableDataset):
    def __init__(
        self,
        data_files: list[str],
        text_column: str,
    ):
        self.data_files = data_files
        self.text_column = text_column

    def _iter_filtered_rows(self) -> Iterator[dict[str, str]]:
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
                    if not isinstance(text, str) or not text.strip():
                        continue

                    yield {self.text_column: text}

    def __iter__(self) -> Iterator[dict[str, str]]:
        yield from self._iter_filtered_rows()


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


def load_hf_text_dataset(
    source_path: Path,
    text_column: str,
    dataset_type: Optional[str] = None,
    *,
    streaming: bool,
):
    if not streaming:
        raise ValueError("Only streaming=True is supported")

    if dataset_type not in {"parquet", "jsonl"}:
        raise ValueError("dataset_type must be one of: parquet, jsonl")

    if dataset_type == "parquet":
        data_files = find_parquet_files([source_path])
        dataset = load_dataset(
            "parquet",
            data_files=data_files,
            split="train",
            streaming=True,
        )
        dataset = dataset.select_columns([text_column])
    elif dataset_type == "jsonl":
        data_files = find_jsonl_files([source_path])
        return JSONLTextIterableDataset(
            data_files=data_files,
            text_column=text_column,
        )

    return dataset


def build_training_dataset(
    text_file: str,
    seq_len: int,
    tokenizer_path: str,
    min_seq_len: int,
    text_column: str,
    dataset_type: Optional[str] = None,
    *,
    streaming: bool = True,
    binpack_window_size: int = 1024,
):
    '''Builds a training dataset for causal language modeling from text data.
    Args:
        text_file: Path to a jsonl file or a directory of parquet/jsonl files.
        seq_len: Maximum sequence length for a single sentence/document chunk.
        tokenizer_path: Path to the Hugging Face tokenizer directory or model name.
        min_seq_len: Drop samples shorter than this token length.
        text_column: Column name in parquet/jsonl files that contains the text data.
        dataset_type: Source type for structured datasets (parquet/jsonl).
        streaming: Must be True.
        binpack_window_size: Unused in the flattening path. Kept for CLI compatibility.
    Returns:
        A tuple of (dataset, tokenizer) where dataset is a Hugging Face Dataset or Iterable
        yielding token chunks with 'input_ids', and tokenizer is the loaded tokenizer instance.
    '''
    if not text_file:
        raise ValueError("text_file is required")

    source_path = Path(text_file)
    if not tokenizer_path:
        raise ValueError("tokenizer_path is required")

    tokenizer = load_hf_tokenizer(tokenizer_path)
    min_seq_len = max(2, min_seq_len)

    source_dataset = load_hf_text_dataset(
        source_path=source_path,
        text_column=text_column,
        dataset_type=dataset_type,
        streaming=streaming,
    )

    dataset = HFTokenChunkIterableDataset(
        source_iterable=source_dataset,
        tokenizer=tokenizer,
        seq_len=seq_len,
        text_column=text_column,
        min_seq_len=min_seq_len,
    )
    return dataset, tokenizer


def collate_causal_batch(
    batch: list[dict[str, Any]],
    pad_token_id: int,
    use_block_diag_mask: bool,
) -> dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("batch is empty")

    max_len = max(len(item["input_ids"]) for item in batch)
    batch_size = len(batch)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    position_ids = torch.full((batch_size, max_len), -1, dtype=torch.long)

    for i, item in enumerate(batch):
        ids = item["input_ids"]
        lbl = item.get("labels", ids)
        pos = item.get("position_ids", list(range(len(ids))))

        ids_tensor = ids if isinstance(ids, torch.Tensor) else torch.tensor(ids, dtype=torch.long)
        lbl_tensor = lbl if isinstance(lbl, torch.Tensor) else torch.tensor(lbl, dtype=torch.long)
        pos_tensor = pos if isinstance(pos, torch.Tensor) else torch.tensor(pos, dtype=torch.long)

        length = int(ids_tensor.numel())
        input_ids[i, :length] = ids_tensor
        labels[i, : min(length, int(lbl_tensor.numel()))] = lbl_tensor[:length]
        position_ids[i, : min(length, int(pos_tensor.numel()))] = pos_tensor[:length]
        attention_mask[i, :length] = 1

    output: dict[str, torch.Tensor] = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }

    if use_block_diag_mask:
        block_diag_mask = build_block_diagonal_attention_mask(
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        output["attention_mask"] = block_diag_mask.unsqueeze(1)

    return output


def build_training_dataloader(
    dataset: Any,
    batch_size: int,
    *,
    pad_token_id: int = 0,
    use_block_diag_mask: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    is_iterable = isinstance(dataset, TorchIterableDataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(not is_iterable),
        drop_last=False,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_causal_batch(
            batch=batch,
            pad_token_id=pad_token_id,
            use_block_diag_mask=use_block_diag_mask,
        ),
    )

# def preview_dataset(dataset: Any, limit: int) -> list[dict[str, Any]]:
#     if limit <= 0:
#         raise ValueError("limit must be > 0")
    
#     print(dataset)

#     if hasattr(dataset, "take"):
#         return list(dataset.take(limit))

#     size = min(limit, len(dataset))
#     if size == 0:
#         return []
#     return list(dataset.select(range(limit)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview parquet/jsonl dataset samples")
    parser.add_argument("--dataset-path", type=str, default="dataset", help="Dataset root dir or file")
    parser.add_argument("--dataset-type", type=str, choices=["parquet", "jsonl"], default="parquet")
    parser.add_argument("--text-column", type=str, default="text", help="Text column name")
    parser.add_argument("--preview-count", type=int, default=2, help="Number of rows to preview")
    parser.add_argument(
        "--streaming",
        type=bool,
        default=True,
        help="Use streaming mode for fast preview without building the full Arrow cache",
    )
    parser.add_argument(
        "--text-file",
        type=str,
        default="dataset/swallow-code-v2/stage5-auto-format/python/medium/train_0001.jsonl",
    )
    parser.add_argument("--tokenizer-path", type=str, default="qwen3_5")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.dataset_path = "dataset/swallow-code-v2"
    args.dataset_type = "jsonl"
    args.text_column = "text"
    args.preview_count = 10
    source_path = Path(args.dataset_path)
    dataset_paths = find_dataset_files([source_path], args.dataset_type)
    print(f"Found {len(dataset_paths)} {args.dataset_type} {dataset_paths[:1]}files for preview.")
    dataset = load_hf_text_dataset(
        source_path=Path(dataset_paths[0]),
        text_column=args.text_column,
        dataset_type=args.dataset_type,
        streaming=True,
    )
    print("数据格式:", args.dataset_type)
    print("数据路径:", dataset_paths[:2])
    print("预览模式:", "streaming" if args.streaming else "eager")
    preview_data = []
    for row in dataset:
        text = row.get(args.text_column)
        if not text or not isinstance(text, str) or not text.strip():
            continue
        preview_data.append({args.text_column: text})
        if len(preview_data) >= args.preview_count:
            break
    tokenizer = load_hf_tokenizer("qwen3_5")
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        raise ValueError("Tokenizer must provide eos_token_id")
    # Use eos as padding by default to keep behavior consistent across tokenizers.
    pad_token_id = eos_id
    dataset2 = HFTokenBlockIterableDataset(
            source_iterable=dataset,
            tokenizer=tokenizer,
            seq_len=2048,
            text_column=args.text_column,
            min_seq_len=2,
            binpack_window_size=2048,
            pad_token_id=pad_token_id,
        )
    for i, sample in enumerate(dataset2):
        print(f"Sample {i}:")
        if i >= args.preview_count:
            break

    print(f"预览 {len(preview_data)} 条数据:")
    
    train_dataset, tokenizer = build_training_dataset(
        text_file="dataset/fineweb-edu",
        seq_len=2048,
        tokenizer_path="qwen3_5",
        min_seq_len=2,
        text_column=args.text_column,
        dataset_type="parquet",
        streaming=True,
        binpack_window_size=2048,
    )

    # if hasattr(train_dataset, "__len__") and len(train_dataset) == 0:
    #     raise ValueError("Training dataset is empty. Check --text-file and --text-column.")
    
    for i, sample in enumerate(train_dataset):
        print(f"Sample2 {i}: {sample}")
        if i >= args.preview_count:
            break
    
    # raise ValueError("Dataset loaded successfully. Stopping here for inspection.")

    # print("数据示例:", preview_data)
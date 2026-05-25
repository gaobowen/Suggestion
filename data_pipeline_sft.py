from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import torch
from datasets import IterableDataset as HFIterableDataset
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset as TorchIterableDataset


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


def _normalize_role(role: Any) -> str:
    role_text = str(role or "").strip().lower()
    role_map = {
        "human": "user",
        "gpt": "assistant",
        "bot": "assistant",
    }
    return role_map.get(role_text, role_text)


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _extract_messages_from_row(row: dict[str, Any], text_column: str) -> list[dict[str, str]]:
    if isinstance(row.get("messages"), list):
        messages = row["messages"]
    elif isinstance(row.get("conversations"), list):
        messages = row["conversations"]
    elif row.get("prompt") is not None and row.get("completion") is not None:
        messages = [
            {"role": "user", "content": _content_to_text(row.get("prompt"))},
            {"role": "assistant", "content": _content_to_text(row.get("completion"))},
        ]
    elif row.get("instruction") is not None and row.get("output") is not None:
        user_content = _content_to_text(row.get("instruction"))
        input_text = _content_to_text(row.get("input"))
        if input_text:
            user_content = f"{user_content}\n{input_text}" if user_content else input_text
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": _content_to_text(row.get("output"))},
        ]
    elif row.get(text_column) is not None:
        messages = [
            {"role": "user", "content": "Please continue."},
            {"role": "assistant", "content": _content_to_text(row.get(text_column))},
        ]
    else:
        return []

    normalized: list[dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = _normalize_role(msg.get("role"))
        content = _content_to_text(msg.get("content", "")).strip()
        if role not in {"system", "user", "assistant", "tool"}:
            continue
        if not content and role != "assistant":
            continue
        normalized.append({"role": role, "content": content})

    return normalized


def _assistant_turn_examples(messages: list[dict[str, str]]) -> Iterator[tuple[list[dict[str, str]], list[dict[str, str]]]]:
    if not messages:
        return

    for index, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "").strip()
        if not content:
            continue
        prompt_messages = messages[:index]
        full_messages = messages[: index + 1]
        if not prompt_messages:
            continue
        yield prompt_messages, full_messages


def _mask_prompt_tokens(
    prompt_ids: list[int],
    full_ids: list[int],
) -> list[int]:
    labels = full_ids.copy()

    matched = 0
    max_matched = min(len(prompt_ids), len(full_ids))
    while matched < max_matched and prompt_ids[matched] == full_ids[matched]:
        matched += 1

    if matched == 0:
        # Fallback: if templates differ unexpectedly, train on all tokens.
        return labels

    for i in range(matched):
        labels[i] = -100
    return labels


def _to_token_id_list(tokenized_output: Any) -> list[int]:
    if isinstance(tokenized_output, torch.Tensor):
        if tokenized_output.dim() == 1:
            return tokenized_output.tolist()
        if tokenized_output.dim() == 2 and tokenized_output.size(0) == 1:
            return tokenized_output[0].tolist()
        raise ValueError("Unexpected tensor shape for tokenized output")

    # Many tokenizers return BatchEncoding (a Mapping but not a plain dict).
    if hasattr(tokenized_output, "get") and tokenized_output.get("input_ids") is not None:
        return _to_token_id_list(tokenized_output.get("input_ids"))

    if isinstance(tokenized_output, dict):
        if "input_ids" not in tokenized_output:
            raise ValueError("Tokenized output dict must contain input_ids")
        return _to_token_id_list(tokenized_output["input_ids"])

    if isinstance(tokenized_output, list):
        if not tokenized_output:
            return []
        if all(isinstance(x, int) for x in tokenized_output):
            return tokenized_output
        if len(tokenized_output) == 1 and isinstance(tokenized_output[0], list):
            nested = tokenized_output[0]
            if all(isinstance(x, int) for x in nested):
                return nested
        if hasattr(tokenized_output[0], "ids"):
            return list(tokenized_output[0].ids)

    raise TypeError(f"Unsupported tokenized output type: {type(tokenized_output)}")


def _tokenize_supervised_pair(
    tokenizer: Any,
    prompt_messages: list[dict[str, str]],
    full_messages: list[dict[str, str]],
    max_seq_len: int,
) -> Optional[dict[str, list[int]]]:
    try:
        prompt_out = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        full_out = tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        prompt_ids = _to_token_id_list(prompt_out)
        full_ids = _to_token_id_list(full_out)
    except Exception:
        raise RuntimeError("Tokenizer must support apply_chat_template for this SFT data format")

    if not full_ids:
        return None

    labels = _mask_prompt_tokens(prompt_ids=prompt_ids, full_ids=full_ids)
    '''
    dataloader 负责构造监督区域（把 prompt 部分设成 -100）。
    model/trainer 负责 causal LM 的 Teacher Forcing 错开一个 token 的 shift。
    两者叠加后，等价于“只在 assistant 回复区域做 next-token loss”。
    '''
    if len(full_ids) > max_seq_len:
        full_ids = full_ids[-max_seq_len:]
        labels = labels[-max_seq_len:]

    if not any(label != -100 for label in labels):
        return None

    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_ids),
    }


def _resolve_json_data_files(source_path: Path, dataset_type: str) -> list[str]:
    if dataset_type == "jsonl":
        return find_files_with_suffixes([source_path], (".jsonl",))
    if dataset_type == "json":
        return find_files_with_suffixes([source_path], (".json",))
    if dataset_type == "parquet":
        return find_files_with_suffixes([source_path], (".parquet",))
    raise ValueError("dataset_type must be one of: parquet, jsonl, json")


class SFTTokenizedIterableDataset(TorchIterableDataset):
    def __init__(
        self,
        source_path: str,
        dataset_type: str,
        tokenizer: Any,
        max_seq_len: int,
        text_column: str,
        streaming: bool,
    ):
        self.source_path = Path(source_path)
        self.dataset_type = dataset_type
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.text_column = text_column
        self.streaming = streaming
        # TRL's SFTTrainer expects datasets to expose `column_names`.
        # Keep it None so TRL infers columns from one sample.
        self.column_names = None

    def _iter_rows(self) -> Iterable[dict[str, Any]]:
        data_files = _resolve_json_data_files(self.source_path, self.dataset_type)

        if self.dataset_type == "parquet":
            dataset = load_dataset(
                "parquet",
                data_files=data_files,
                split="train",
                streaming=self.streaming,
            )
        else:
            dataset = load_dataset(
                "json",
                data_files=data_files,
                split="train",
                streaming=self.streaming,
            )

        if isinstance(dataset, HFIterableDataset):
            yield from dataset
            return

        for i in range(len(dataset)):
            yield dataset[i]

    def __iter__(self) -> Iterator[dict[str, list[int]]]:
        for row in self._iter_rows():
            if not isinstance(row, dict):
                continue

            messages = _extract_messages_from_row(row=row, text_column=self.text_column)
            if not messages:
                continue

            for prompt_messages, full_messages in _assistant_turn_examples(messages):
                sample = _tokenize_supervised_pair(
                    tokenizer=self.tokenizer,
                    prompt_messages=prompt_messages,
                    full_messages=full_messages,
                    max_seq_len=self.max_seq_len,
                )
                if sample is None:
                    continue
                yield sample


def collate_sft_batch(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("batch is empty")

    max_len = max(len(item["input_ids"]) for item in batch)
    batch_size = len(batch)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, item in enumerate(batch):
        ids = item["input_ids"]
        lbs = item["labels"]
        mask = item.get("attention_mask", [1] * len(ids))

        ids_tensor = ids if isinstance(ids, torch.Tensor) else torch.tensor(ids, dtype=torch.long)
        lbs_tensor = lbs if isinstance(lbs, torch.Tensor) else torch.tensor(lbs, dtype=torch.long)
        msk_tensor = mask if isinstance(mask, torch.Tensor) else torch.tensor(mask, dtype=torch.long)

        seq_len = int(ids_tensor.numel())
        input_ids[i, :seq_len] = ids_tensor
        labels[i, : min(seq_len, int(lbs_tensor.numel()))] = lbs_tensor[:seq_len]
        attention_mask[i, : min(seq_len, int(msk_tensor.numel()))] = msk_tensor[:seq_len]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def build_sft_train_dataloader(
    source_path: str,
    dataset_type: str,
    tokenizer: Any,
    max_seq_len: int,
    text_column: str,
    batch_size: int,
    num_workers: int,
    streaming: bool = True,
) -> tuple[SFTTokenizedIterableDataset, DataLoader]:
    dataset = SFTTokenizedIterableDataset(
        source_path=source_path,
        dataset_type=dataset_type,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        text_column=text_column,
        streaming=streaming,
    )

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must provide pad_token_id or eos_token_id")
        pad_token_id = tokenizer.eos_token_id

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_sft_batch(batch=batch, pad_token_id=pad_token_id),
    )
    return dataset, dataloader


def _demo_load_first_row(source_path: str, dataset_type: str) -> dict[str, Any] | None:
    data_files = _resolve_json_data_files(Path(source_path), dataset_type)

    if dataset_type == "parquet":
        dataset = load_dataset("parquet", data_files=data_files, split="train", streaming=True)
    else:
        dataset = load_dataset("json", data_files=data_files, split="train", streaming=True)

    for row in dataset:
        if isinstance(row, dict):
            return row
    return None


def _run_template_demo() -> None:
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description="Debug chat template rendering for SFT samples")
    parser.add_argument("--tokenizer-path", type=str, default="minimind")
    parser.add_argument("--source-path", type=str, default="dataset/minimind/sft_t2t_mini.jsonl")
    parser.add_argument("--dataset-type", type=str, choices=["jsonl", "json", "parquet"], default="jsonl")
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--print-token-ids", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=True,
    )

    row = _demo_load_first_row(args.source_path, args.dataset_type)
    if row is None:
        raise ValueError("No row loaded from dataset")

    messages = _extract_messages_from_row(row=row, text_column=args.text_column)
    if not messages:
        raise ValueError("No messages could be extracted from the first row")

    pairs = list(_assistant_turn_examples(messages))
    if not pairs:
        raise ValueError("No assistant turn found in the first row")

    prompt_messages, full_messages = pairs[0]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    print("[messages]")
    for idx, msg in enumerate(messages):
        role = msg.get("role", "")
        content = str(msg.get("content", ""))
        print(f"{idx:02d}. {role}: {content[:120]}")

    print("\n[prompt_text via apply_chat_template]")
    print(prompt_text)

    print("\n[full_text via apply_chat_template]")
    print(full_text)

    if args.print_token_ids:
        prompt_ids = _to_token_id_list(
            tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        full_ids = _to_token_id_list(
            tokenizer.apply_chat_template(
                full_messages,
                tokenize=True,
                add_generation_prompt=False,
            )
        )
        print("\n[prompt_ids]")
        print(prompt_ids)
        print("\n[full_ids]")
        print(full_ids)


if __name__ == "__main__":
    _run_template_demo()



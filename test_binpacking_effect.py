from __future__ import annotations

import argparse
from typing import List

import torch

from data_pipeline import build_block_diagonal_attention_mask
from data_pipeline import pack_units_with_binpacking


def build_units(max_n: int, eos_token_id: int) -> List[List[int]]:
    # n=1 -> [1, eos], n=2 -> [2,2,eos], ...
    return [[n] * n + [eos_token_id] for n in range(1, max_n + 1)]


def sequential_pack(units: List[List[int]], block_size: int) -> List[List[int]]:
    packed: List[List[int]] = []
    cur: List[int] = []
    for unit in units:
        if len(unit) > block_size:
            continue
        if cur and len(cur) + len(unit) > block_size:
            packed.append(cur)
            cur = []
        cur.extend(unit)
    if cur:
        packed.append(cur)
    return packed


def summarize(packed: List[List[int]], block_size: int, title: str) -> None:
    print(f"\n=== {title} ===")
    if not packed:
        print("no packed samples")
        return

    total_tokens = sum(len(x) for x in packed)
    total_capacity = len(packed) * block_size
    util = total_tokens / total_capacity if total_capacity else 0.0
    print(f"samples: {len(packed)}")
    print(f"total tokens: {total_tokens}")
    print(f"total capacity: {total_capacity}")
    print(f"utilization: {util:.2%}")

    for i, sample in enumerate(packed[:10], start=1):
        print(f"sample-{i:02d}: len={len(sample):2d}, tokens={sample}")


def print_packing_masks(binpacked_dicts: List[dict], max_samples: int) -> None:
    print("\n=== Binpacking Masks ===")
    if not binpacked_dicts:
        print("no masks to show")
        return

    for idx, sample in enumerate(binpacked_dicts[:max_samples], start=1):
        pos = sample["position_ids"]
        pos_tensor = torch.tensor(pos, dtype=torch.long).unsqueeze(0)
        attn_tensor = torch.ones_like(pos_tensor)

        mask = build_block_diagonal_attention_mask(
            position_ids=pos_tensor,
            attention_mask=attn_tensor,
        )[0]

        print(f"sample-{idx:02d}: len={len(pos)}, position_ids={pos}")
        for row in mask.tolist():
            print(" ".join("1" if bool(v) else "0" for v in row))
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic units and inspect binpacking effect")
    parser.add_argument("--max-n", type=int, default=10, help="Generate units 1..N")
    parser.add_argument("--block-size", type=int, default=16, help="Target packed block size")
    parser.add_argument("--eos", type=int, default=0, help="Synthetic eos token id")
    parser.add_argument("--min-seq-len", type=int, default=2, help="Drop packs shorter than this length")
    parser.add_argument("--max-mask-samples", type=int, default=3, help="How many packed samples to print masks for")
    args = parser.parse_args()

    units = build_units(max_n=args.max_n, eos_token_id=args.eos)
    print("input units:")
    for idx, u in enumerate(units, start=1):
        print(f"unit-{idx:02d}: len={len(u):2d}, tokens={u}")

    seq = sequential_pack(units, block_size=args.block_size)
    summarize(seq, block_size=args.block_size, title="Sequential (baseline)")

    binpacked_dicts = pack_units_with_binpacking(
        pack_units=units,
        block_size=args.block_size,
        min_seq_len=args.min_seq_len,
    )
    binpacked = [x["input_ids"] for x in binpacked_dicts]
    summarize(binpacked, block_size=args.block_size, title="Binpacking")
    print_packing_masks(binpacked_dicts, max_samples=max(0, args.max_mask_samples))


if __name__ == "__main__":
    main()

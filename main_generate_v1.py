from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers.trainer_utils import get_last_checkpoint

from data_pipeline import load_hf_tokenizer
from model_v1 import MiniMindForCausalLM


def resolve_checkpoint_path(path: str) -> str:
	candidate = Path(path)
	if not candidate.exists():
		raise FileNotFoundError(f"Checkpoint path not found: {path}")

	# Allow passing the output directory and automatically select the latest checkpoint.
	latest = get_last_checkpoint(str(candidate))
	if latest is not None:
		return latest

	model_file = candidate / "model.safetensors"
	if model_file.exists():
		return str(candidate)

	raise ValueError(
		f"No checkpoint found under: {path}. Expected a trainer output dir or a checkpoint dir with model.safetensors"
	)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run inference for MiniMind v1 checkpoints")
	parser.add_argument("--checkpoint", type=str, default="output_v1", help="Checkpoint dir or trainer output dir")
	parser.add_argument("--tokenizer-path", type=str, default="qwen3_5")
	parser.add_argument("--prompt", type=str, default="你好，请简要介绍一下你自己。")
	parser.add_argument("--chat-template", action=argparse.BooleanOptionalAction, default=False)
	parser.add_argument("--system-prompt", type=str, default="You are a helpful assistant.")
	parser.add_argument("--max-new-tokens", type=int, default=128)
	parser.add_argument("--temperature", type=float, default=0.8)
	parser.add_argument("--top-p", type=float, default=0.9)
	parser.add_argument("--top-k", type=int, default=50)
	parser.add_argument("--repetition-penalty", type=float, default=1.0)
	parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--interactive", action="store_true", help="Start a simple interactive loop")
	return parser.parse_args()


def build_model_prompt(tokenizer, prompt: str, chat_template: bool, system_prompt: str) -> str:
	if not chat_template:
		return prompt

	messages = []
	if system_prompt.strip():
		messages.append({"role": "system", "content": system_prompt.strip()})
	messages.append({"role": "user", "content": prompt})

	apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
	if callable(apply_chat_template):
		try:
			return apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
		except TypeError:
			# Older tokenizer implementations may not support add_generation_prompt.
			return apply_chat_template(messages, tokenize=False)

	# Fallback if tokenizer has no chat template support.
	fallback = ""
	if system_prompt.strip():
		fallback += f"System: {system_prompt.strip()}\n"
	fallback += f"User: {prompt}\nAssistant:"
	return fallback


def build_model_and_tokenizer(checkpoint_path: str, tokenizer_path: str, device: str):
	tokenizer = load_hf_tokenizer(tokenizer_path)
	model = MiniMindForCausalLM.from_pretrained(checkpoint_path)
	model.eval()
	model.to(device)
	return model, tokenizer


@torch.inference_mode()
def generate_once(
	model: MiniMindForCausalLM,
	tokenizer,
	prompt: str,
	*,
	chat_template: bool,
	system_prompt: str,
	max_new_tokens: int,
	temperature: float,
	top_p: float,
	top_k: int,
	repetition_penalty: float,
	do_sample: bool,
	device: str,
) -> str:
	if not prompt.strip():
		return ""

	model_prompt = build_model_prompt(
		tokenizer=tokenizer,
		prompt=prompt,
		chat_template=chat_template,
		system_prompt=system_prompt,
	)
	encoded = tokenizer(model_prompt, return_tensors="pt")
	input_ids = encoded["input_ids"].to(device)
	attention_mask = encoded.get("attention_mask")
	if attention_mask is None:
		attention_mask = torch.ones_like(input_ids, dtype=torch.long)
	attention_mask = attention_mask.to(device)

	output_ids = model.generate(
		input_ids=input_ids,
		attention_mask=attention_mask,
		max_new_tokens=max_new_tokens,
		temperature=temperature,
		top_p=top_p,
		top_k=top_k,
		repetition_penalty=repetition_penalty,
		do_sample=do_sample,
		eos_token_id=model.config.eos_token_id,
		use_cache=True,
	)

	generated_ids = output_ids[0, input_ids.shape[1] :]
	return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main() -> None:
	args = parse_args()
	checkpoint_path = resolve_checkpoint_path(args.checkpoint)
	print(f"Loading checkpoint: {checkpoint_path}")

	model, tokenizer = build_model_and_tokenizer(
		checkpoint_path=checkpoint_path,
		tokenizer_path=args.tokenizer_path,
		device=args.device,
	)

	if not args.interactive:
		text = generate_once(
			model=model,
			tokenizer=tokenizer,
			prompt=args.prompt,
			chat_template=args.chat_template,
			system_prompt=args.system_prompt,
			max_new_tokens=args.max_new_tokens,
			temperature=args.temperature,
			top_p=args.top_p,
			top_k=args.top_k,
			repetition_penalty=args.repetition_penalty,
			do_sample=args.do_sample,
			device=args.device,
		)
		print("\n[Prompt]")
		print(args.prompt)
		print("\n[Generation]")
		print(text)
		return

	print("Interactive mode. Type 'exit' to quit.")
	while True:
		prompt = input("\n>>> ").strip()
		if prompt.lower() in {"exit", "quit"}:
			break
		text = generate_once(
			model=model,
			tokenizer=tokenizer,
			prompt=prompt,
			chat_template=args.chat_template,
			system_prompt=args.system_prompt,
			max_new_tokens=args.max_new_tokens,
			temperature=args.temperature,
			top_p=args.top_p,
			top_k=args.top_k,
			repetition_penalty=args.repetition_penalty,
			do_sample=args.do_sample,
			device=args.device,
		)
		print(text)


if __name__ == "__main__":
	main()

	# python main_generate_v1.py --checkpoint output_sft_v1 --tokenizer-path minimind --prompt "请介绍一下你自己。" --chat-template --system-prompt "你是一个乐于助人的助手。"
	# python main_generate_v1.py --checkpoint output_sft_v1 --tokenizer-path minimind --prompt "请介绍一下你自己。"

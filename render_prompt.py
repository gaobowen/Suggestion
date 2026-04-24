import argparse
import json
from pathlib import Path
from typing import Any

from jinja2 import Environment

from data_pipeline import load_chat_samples


def tojson_filter(value: Any, ensure_ascii: bool = True, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent)


def render_one(
    env: Environment,
    template_text: str,
    sample: dict[str, Any],
    add_generation_prompt: bool,
    enable_thinking: bool,
    clear_thinking: bool,
) -> str:
    template = env.from_string(template_text)
    rendered = template.render(
        messages=sample.get("messages", []),
        tools=sample.get("tools", []),
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
        clear_thinking=clear_thinking,
    )
    return rendered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render chat samples into prompt text with Jinja template")
    parser.add_argument("--template", type=str, default="chat_template.jinja", help="Path to Jinja template")
    parser.add_argument("--samples", type=str, default="chat_test_samples.json", help="Path to sample json list")
    parser.add_argument("--output-dir", type=str, default="rendered_prompts", help="Output directory")
    parser.add_argument(
        "--add-generation-prompt",
        action="store_true",
        help="Append assistant generation prefix according to template",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable <think> prefix behavior in template",
    )
    parser.add_argument(
        "--clear-thinking",
        action="store_true",
        help="Clear historical thinking content in template",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    template_path = Path(args.template)
    sample_path = Path(args.samples)
    output_dir = Path(args.output_dir)

    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    samples = load_chat_samples(sample_path)
    template_text = template_path.read_text(encoding="utf-8")

    env = Environment(autoescape=False)
    env.filters["tojson"] = tojson_filter

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, str]] = []
    for idx, sample in enumerate(samples):
        sample_id = str(sample.get("id", f"sample_{idx:03d}"))
        prompt = render_one(
            env=env,
            template_text=template_text,
            sample=sample,
            add_generation_prompt=args.add_generation_prompt,
            enable_thinking=args.enable_thinking,
            clear_thinking=args.clear_thinking,
        )

        output_file = output_dir / f"{sample_id}.prompt.txt"
        output_file.write_text(prompt, encoding="utf-8")
        manifest.append({"id": sample_id, "output": str(output_file)})

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"rendered samples: {len(manifest)}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
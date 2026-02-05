#!/usr/bin/env python3
"""
Minimal reproduction script to verify whether Qwen3 "thinking mode" can be disabled.

It follows the official Qwen3 usage pattern:
  - tokenizer.apply_chat_template(..., enable_thinking=...)
  - model.generate(...)
  - optionally parse thinking content via </think> token id

This script prints:
  - the rendered prompt text
  - whether output contains "<think>" / "</think>"
  - a small output preview
"""

import argparse
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_thinking_by_token_id(
    output_ids: list[int],
    think_end_token_id: int = 151668,  # official Qwen example: </think>
) -> Tuple[str, str]:
    """
    Split generated output into (thinking_content, content) by locating </think> token id.
    If not found, returns ("", full_text).
    """
    try:
        # find last occurrence of think_end_token_id
        idx = len(output_ids) - output_ids[::-1].index(think_end_token_id)
    except ValueError:
        idx = 0
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="")
    ap.add_argument("--prompt", type=str, default="Give me a short introduction to large language model.")
    ap.add_argument("--enable_thinking", type=str, default="false", choices=["true", "false"])
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--think_end_token_id", type=int, default=151668)
    args = ap.parse_args()

    enable_thinking = args.enable_thinking.lower() == "true"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map=args.device_map,
        trust_remote_code=True,
    )

    messages = [{"role": "user", "content": args.prompt}]

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    print("=" * 80)
    print(f"model_name={args.model_name}")
    print(f"enable_thinking={enable_thinking} temperature={args.temperature} max_new_tokens={args.max_new_tokens}")
    print("-" * 80)
    print("Rendered prompt:")
    print(rendered)
    print("=" * 80)

    model_inputs = tokenizer([rendered], return_tensors="pt").to(model.device)

    # If temperature == 0, force greedy
    do_sample = args.temperature > 0
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=do_sample,
    )
    if do_sample:
        gen_kwargs["temperature"] = args.temperature

    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, **gen_kwargs)

    gen_only = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

    # Quick string-based inspection
    decoded_full = tokenizer.decode(gen_only, skip_special_tokens=False)
    has_think_tag = "<think>" in decoded_full
    has_think_end_tag = "</think>" in decoded_full

    # Token-id based split (if enabled)
    try:
        split_idx = len(gen_only) - gen_only[::-1].index(args.think_end_token_id)
    except ValueError:
        split_idx = 0

    thinking_text = tokenizer.decode(gen_only[:split_idx], skip_special_tokens=True).strip("\n")
    content_text = tokenizer.decode(gen_only[split_idx:], skip_special_tokens=True).strip("\n")

    print("Output inspection:")
    print(f"  contains '<think>': {has_think_tag}")
    print(f"  contains '</think>': {has_think_end_tag}")
    print(f"  think_end_token_id={args.think_end_token_id} split_idx={split_idx} (0 means not found)")
    print("-" * 80)
    if split_idx > 0:
        print("Thinking content (preview, first 400 chars):")
        print(thinking_text[:400])
        print("-" * 80)
    print("Final content (preview, first 800 chars):")
    print(content_text[:800])
    print("=" * 80)


if __name__ == "__main__":
    main()


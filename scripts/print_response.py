"""Print a raw Fireworks (OpenAI-compatible) chat response.

Usage:
    python scripts/print_response.py
    python scripts/print_response.py --model accounts/fireworks/models/gpt-oss-120b
    python scripts/print_response.py --prompt "Reply with only: bet" --max-tokens 60
    python scripts/print_response.py --raw      # dump the entire JSON object

Reads the API key from the local .fireworks file (or FIREWORKS_API_KEY).
"""

from __future__ import annotations

import argparse
import json
import os

from openai import OpenAI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="accounts/fireworks/models/deepseek-v4-pro")
    ap.add_argument("--prompt", default="Heads-up poker: you hold A K, board empty, "
                    "pot 3, to call 1. Reply with only: call, raise N, or fold "
                    "(put the action on the last line).")
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--raw", action="store_true", help="print the full JSON object")
    args = ap.parse_args()

    key = os.environ.get("FIREWORKS_API_KEY")
    if not key and os.path.exists(".fireworks"):
        key = open(".fireworks").read().strip()
    if not key:
        raise SystemExit("Set FIREWORKS_API_KEY or create a .fireworks file.")

    client = OpenAI(api_key=key, base_url="https://api.fireworks.ai/inference/v1")
    resp = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": args.prompt}],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    if args.raw:
        print(json.dumps(resp.model_dump(), indent=2, default=str))
        return

    choice = resp.choices[0]
    msg = choice.message
    reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
    print(f"model         : {resp.model}")
    print(f"finish_reason : {choice.finish_reason}    "
          f"(stop=complete, length=truncated by max_tokens)")
    print(f"usage         : {resp.usage.completion_tokens}/{args.max_tokens} "
          f"completion tokens used")
    print("\n----- content (the ANSWER — this is what we parse) -----")
    print(repr(msg.content))
    print(f"\n----- reasoning_content (chain-of-thought — {len(reasoning or '')} chars, "
          f"logged not parsed) -----")
    print((reasoning or "")[:1200] + ("…[truncated for display]" if reasoning and len(reasoning) > 1200 else ""))


if __name__ == "__main__":
    main()

"""Smoke test 2/4 — Google Gemini.

Verifies the API key, model name, and JSON-output prompting we'll lean on in
Phase 4. Should print a tiny JSON object and exit.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
load_dotenv(override=False)

try:
    from google import genai
    from google.genai import types
except ImportError as exc:
    print(f"❌ google-genai import failed: {exc}")
    sys.exit(1)


def main() -> None:
    if not os.environ.get("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY not set")
        sys.exit(1)

    model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
    print(f"→ Calling {model} (text-only, max 200 tokens)...")

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    
    try:
        resp = client.models.generate_content(
            model=model,
            contents="Say pong.",
            config=types.GenerateContentConfig(
                system_instruction='Reply with ONLY a JSON object of the form: {"reply": "<one word>"}. No prose, no markdown.',
                max_output_tokens=512,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
    except Exception as exc:
        print(f"❌ Gemini API call failed: {exc}")
        sys.exit(1)

    # Print token usage so we can confirm thinking tokens == 0
    meta = resp.usage_metadata
    if meta:
        print(f"  thoughts_token_count:    {getattr(meta, 'thoughts_token_count', 'n/a')}")
        print(f"  candidates_token_count:  {getattr(meta, 'candidates_token_count', 'n/a')}")

    text = (resp.text or "").strip()
    print(f"  raw: {text}")

    # Remove markdown code fences if generated
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
            text = "\n".join(lines[1:-1]).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        print(f"❌ Gemini did not return valid JSON: {exc}")
        sys.exit(1)

    if "reply" not in data:
        print(f"❌ JSON missing 'reply' key: {data}")
        sys.exit(1)

    print(f"✅ Gemini OK — model: {model}, reply: {data['reply']!r}")


if __name__ == "__main__":
    main()

"""Quick model probe — finds first Gemini model that responds."""
from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types
import os

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

candidates = [
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
]

for model in candidates:
    print(f"→ Trying {model}...", end=" ", flush=True)
    try:
        r = client.models.generate_content(
            model=model,
            contents="Say pong.",
            config=types.GenerateContentConfig(
                system_instruction='Reply ONLY with JSON: {"reply":"<word>"}',
                max_output_tokens=30,
            ),
        )
        print(f"✅  reply: {r.text.strip()}")
        print(f"\nBest model: {model}")
        break
    except Exception as e:
        code = getattr(e, 'status_code', '?')
        print(f"❌ {code}: {str(e)[:80]}")

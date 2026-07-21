"""LLM planner for the UGV Beast — natural language (+ optional camera frame)
turns into a validated `ActionPlan`.

Same architecture as the arm example's planner.py:

  utterance ──► Gemini (constrained JSON) ──► parse_plan_json ──► ActionPlan
                                                       │
                                                       ▼
                                              validate_plan (drive.py)
                                                       │
                                                       ▼
                                              DriveExecutor.execute

Uses Gemini (not Claude) to match the rest of this project's stack — same
`thinking_config(thinking_budget=0)` fix already validated in the arm
example's planner.py, since gemini-3.5-flash is a thinking model by default
and un-budgeted thinking was truncating JSON output there too.

The action vocabulary is the UGV Beast's, not the arm's — see drive.py for
the full verb list. This is intentionally a *small, discrete* vocabulary
(short translations, in-place turns, camera servo steps, lights, utilities)
matching Cyberwave's own ugv-voice-controlled tutorial, not continuous
velocity control.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from drive import ActionPlan, validate_plan


SYSTEM_PROMPT = """You are the motion planner for a Waveshare UGV Beast rover.

You translate the user's natural-language request into a JSON action plan.
You do nothing else — no chat, no apologies, no markdown.

Allowed actions (use ONLY these types):
  { "type": "move_forward",  "distance": <metres, 0.0-1.0> }
  { "type": "move_backward", "distance": <metres, 0.0-1.0> }
  { "type": "turn_left",     "angle": <radians, 0.0-3.14> }
  { "type": "turn_right",    "angle": <radians, 0.0-3.14> }
  { "type": "stop" }
  { "type": "wait",          "duration": <seconds, 0.0-5.0> }
  { "type": "camera_up" }
  { "type": "camera_down" }
  { "type": "camera_left" }
  { "type": "camera_right" }
  { "type": "camera_default" }
  { "type": "chassis_light_toggle" }
  { "type": "camera_light_toggle" }
  { "type": "take_photo" }
  { "type": "battery_check" }

Output format — return EXACTLY one JSON object, no code fences, no commentary:

{
  "say": "<one short sentence describing what you're about to do>",
  "actions": [ ...1-8 action objects from the list above... ]
}

Rules:
- "actions" must contain 0-8 entries. Zero actions is valid for pure Q&A.
- "distance" is metres, capped at 1.0 per action — this is a SHORT translation,
  not metric navigation. Typical values: 0.2-0.5 for "a little", up to 1.0 for
  "forward" with no qualifier.
- "angle" is radians, capped at ~3.14 (pi). Typical: 0.3-0.6 for a small turn,
  ~1.57 (pi/2) for "turn right" with no qualifier.
- camera_up/down/left/right/default and the toggle/utility actions take NO
  arguments — one discrete step per call.
- If the request is unsafe, ambiguous, or impossible, still return a valid
  plan — pick a small conservative action (or an empty actions list) and
  explain in "say".
- NEVER output prose outside the JSON object. NEVER use code fences.

Few-shot examples:

User: "drive forward a little, then turn right"
{"say":"Moving forward a bit, then turning right.","actions":[{"type":"move_forward","distance":0.3},{"type":"turn_right","angle":0.6}]}

User: "stop"
{"say":"Stopping.","actions":[{"type":"stop"}]}

User: "look up"
{"say":"Tilting the camera up.","actions":[{"type":"camera_up"}]}

User: "turn all the way around"
{"say":"Turning right in place, about a half rotation.","actions":[{"type":"turn_right","angle":3.14}]}
"""


@dataclass
class PlanResult:
    """Outcome of a single planner call."""

    plan: ActionPlan | None
    raw_response: str
    error: str | None
    model: str

    @property
    def ok(self) -> bool:
        return self.plan is not None and self.error is None


_FENCE_HEAD = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_FENCE_TAIL = re.compile(r"\s*```$")


def parse_plan_json(raw: str) -> tuple[ActionPlan | None, str | None]:
    """Best-effort extract a JSON object from `raw` and convert to ActionPlan.

    Never raises. Returns (plan, None) on success, (None, error_message) on
    failure — bad output from the LLM becomes a spoken apology, never a crash.
    """
    text = (raw or "").strip()
    if not text:
        return None, "empty response"

    text = _FENCE_HEAD.sub("", text)
    text = _FENCE_TAIL.sub("", text)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, f"no JSON object found in response: {raw[:200]!r}"

    blob = text[start : end + 1]

    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        return None, f"JSON decode error: {exc}  (blob: {blob[:200]!r})"

    if not isinstance(data, dict):
        return None, f"top-level JSON must be an object, got {type(data).__name__}"

    try:
        plan = ActionPlan.from_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        return None, f"plan shape error: {exc}  (data: {data!r})"

    errors = validate_plan(plan)
    if errors:
        return None, "validation failed:\n  - " + "\n  - ".join(errors)

    return plan, None


def plan_from_utterance(
    utterance: str,
    *,
    model: str | None = None,
    max_tokens: int = 400,
    temperature: float = 0.2,
) -> PlanResult:
    """Call Gemini (text-only) with `utterance` and return a `PlanResult`."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    chosen_model = model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
    effective_tokens = max(max_tokens, 1024)  # headroom for full JSON plan

    response = client.models.generate_content(
        model=chosen_model,
        contents=utterance,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=effective_tokens,
            temperature=temperature,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    meta = response.usage_metadata
    if meta:
        thoughts = getattr(meta, "thoughts_token_count", 0) or 0
        candidates = getattr(meta, "candidates_token_count", 0) or 0
        print(f"  [planner] thoughts={thoughts} candidates={candidates} model={chosen_model}")

    raw = response.text or ""
    plan, err = parse_plan_json(raw)
    return PlanResult(plan=plan, raw_response=raw, error=err, model=chosen_model)


# ---------------------------------------------------------------------------
# Vision-aware planning
# ---------------------------------------------------------------------------

VISION_SYSTEM_PROMPT = """You are the motion planner AND scene narrator for a Waveshare UGV Beast rover.
You receive an image from the rover's pan-tilt workspace camera AND a
natural-language request from the operator. Return a single JSON object that
either describes what you see, plans a short drive, or does both.

The camera is mounted on a pan-tilt head on the rover, facing the direction
of travel. "left" / "right" in your descriptions match the camera frame.

Allowed actions (use ONLY these types):
  { "type": "move_forward",  "distance": <metres, 0.0-1.0> }
  { "type": "move_backward", "distance": <metres, 0.0-1.0> }
  { "type": "turn_left",     "angle": <radians, 0.0-3.14> }
  { "type": "turn_right",    "angle": <radians, 0.0-3.14> }
  { "type": "stop" }
  { "type": "wait",          "duration": <seconds, 0.0-5.0> }
  { "type": "camera_up" }
  { "type": "camera_down" }
  { "type": "camera_left" }
  { "type": "camera_right" }
  { "type": "camera_default" }
  { "type": "chassis_light_toggle" }
  { "type": "camera_light_toggle" }
  { "type": "take_photo" }
  { "type": "battery_check" }

Output format — return EXACTLY one JSON object, no code fences, no markdown,
no commentary outside the JSON. Schema:

{
  "say":     "<the spoken response — describe the scene, answer the question,
 or narrate the drive>",
  "actions": [ ... 0 to 8 action objects from the list above ... ]
}

Decision rules:

1. If the operator asks ABOUT the scene ("what do you see?", "is there a box
   ahead?"), respond ONLY with description in `say` and an empty `actions`
   array. NO MOTION for purely informational questions.

2. If the operator asks for MOTION ("drive forward a little", "turn right"),
   plan motion in `actions`. `say` should briefly narrate what you'll do.

3. If the operator asks for VISUALLY-GROUNDED MOTION ("drive toward the
   box"), describe what you see in `say`, then plan a SHORT, conservative
   drive (e.g. move_forward distance <= 0.4) followed by a "stop" action.
   You cannot precisely aim without calibration — this is a cautious nudge
   toward the described object, not a guaranteed arrival.

4. If the operator references something you DON'T see, say so honestly and
   leave `actions` empty. Do not pretend or hallucinate motion.

5. Any drive plan (rule 2 or 3) should end with an explicit "stop" action
   unless the request is a single small turn/camera move that naturally
   settles on its own.

6. NEVER output prose outside the JSON object. NEVER use code fences.

Few-shot examples:

User says: "what's ahead of you?"
{"say":"I see an open floor with a cardboard box a few feet ahead and slightly to the left.","actions":[]}

User says: "drive forward a little, then turn right"
{"say":"Moving forward a bit, then turning right.","actions":[{"type":"move_forward","distance":0.3},{"type":"turn_right","angle":0.6}]}

User says: "drive toward the box"
{"say":"I see the box ahead and slightly left. Moving toward it carefully.","actions":[{"type":"turn_left","angle":0.2},{"type":"move_forward","distance":0.4},{"type":"stop"}]}

User says: "is there a door in front of you?"
{"say":"No, I don't see a door — just open floor and a box.","actions":[]}
"""


def plan_from_utterance_with_image(
    utterance: str,
    frame_b64_jpeg: str | None,
    *,
    model: str | None = None,
    max_tokens: int = 500,
    temperature: float = 0.2,
) -> PlanResult:
    """Call Gemini Vision with `utterance` + the image, return a `PlanResult`.

    Falls back to the text-only planner if `frame_b64_jpeg` is None, so the
    agent stays usable when the camera is down for a turn.
    """
    if frame_b64_jpeg is None:
        return plan_from_utterance(utterance, model=model, max_tokens=max_tokens, temperature=temperature)

    import base64
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    chosen_model = model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
    effective_tokens = max(max_tokens, 1024)

    try:
        image_bytes = base64.b64decode(frame_b64_jpeg)
    except Exception as exc:
        return PlanResult(
            plan=None,
            raw_response="",
            error=f"failed to decode base64 image: {exc}",
            model=chosen_model,
        )

    response = client.models.generate_content(
        model=chosen_model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            utterance,
        ],
        config=types.GenerateContentConfig(
            system_instruction=VISION_SYSTEM_PROMPT,
            max_output_tokens=effective_tokens,
            temperature=temperature,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    meta = response.usage_metadata
    if meta:
        thoughts = getattr(meta, "thoughts_token_count", 0) or 0
        candidates = getattr(meta, "candidates_token_count", 0) or 0
        print(f"  [vision-planner] thoughts={thoughts} candidates={candidates} model={chosen_model}")

    raw = response.text or ""
    plan, err = parse_plan_json(raw)
    return PlanResult(plan=plan, raw_response=raw, error=err, model=chosen_model)

# nl_ugv_controller — setup notes

## What's in this zip

Written fresh, against your `inspect_sdk.py` output + Cyberwave's own
`ugv-voice-controlled` tutorial (verified, not guessed):

- `drive.py` — `ActionPlan` schema, validation, clamping, `DriveExecutor`.
  Dispatches to `robot.move_forward/move_backward/turn_left/turn_right`
  (top-level SDK methods) and `robot.commands.*` for camera servo, lights,
  and utility verbs — the exact surface your twin exposed.
- `planner.py` — Gemini text + vision planner, UGV action vocabulary.
  Same `thinking_config(thinking_budget=0)` + floored `max_output_tokens`
  fix already validated in the arm example's planner.py.
- `nl_ugv_controller.py` — orchestrator / agent loop, `--voice --vision
  --dry-run --check` flags, kill-switch words (`stop`/`halt`/`freeze`)
  that bypass the planner for an immediate stop.
- `smoke_tests/06_planner.py`, `smoke_tests/08_vision_planner.py` — offline
  and `--execute` smoke tests, mirroring the arm example's numbering.
- `.env.example` — UGV-specific env template (`waveshare/ugv-beast` asset
  key, no arm joints).

## What you still need to copy over (unchanged)

`voice.py` and `vision.py` are robot-agnostic — they only touch your
microphone and webcam/OpenCV, never the robot SDK. I never received their
source in this conversation, so rather than reconstruct them from
description and risk drift, **copy them as-is** from your existing
`nl_arm_controller/` folder into this one:

```powershell
Copy-Item ..\nl_arm_controller\voice.py .\voice.py
Copy-Item ..\nl_arm_controller\vision.py .\vision.py
```

If `vision.py`'s camera source ends up pointing at the UGV's pan-tilt
camera stream instead of a local laptop webcam (the Cyberwave tutorial
mentions two capture paths — laptop-side network stream vs. on-rover
`/dev/video0`), you may need to adjust `open_camera_from_env()` there. Check
the tutorial's §3.2 ("Vision as an input modality") for which path applies
to your setup.

## requirements.txt

Same dependencies as the arm example (`google-genai`, `python-dotenv`,
`cyberwave`, `sounddevice`, `soundfile`, `pynput`, `httpx`). If you don't
already have a working `requirements.txt` copied over, run:

```powershell
Copy-Item ..\nl_arm_controller\requirements.txt .\requirements.txt
```

## Before you run anything

1. `cp .env.example .env` and fill in real values — **use your UGV Beast's
   twin/environment UUIDs**, not the arm's.
2. Confirm `.env` is gitignored (should already be, from your arm project).
3. Run `python nl_ugv_controller.py --check` first.
4. Then `python smoke_tests/06_planner.py` (offline, no rover, no quota risk
   beyond the Gemini calls) before ever touching `--execute` or the real
   loop.
5. Per Cyberwave's own safety notes: stay in `CW_MODE=simulation` and use
   `--dry-run` while iterating on the prompt. Only flip to live mode in a
   cleared, marked lane, with a physical E-stop within reach.

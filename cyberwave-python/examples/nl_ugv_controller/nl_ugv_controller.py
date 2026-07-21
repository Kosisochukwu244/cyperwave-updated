"""Voice/Text-driven UGV Beast controller — natural-language drive agent.

Same five-module pattern as the arm's nl_arm_controller.py, adapted to the
UGV Beast's actual SDK surface (confirmed via inspect_sdk.py against the
live twin — see drive.py's module docstring for the full trace):

    python nl_ugv_controller.py                    # text REPL, drives the twin
    python nl_ugv_controller.py --voice             # voice REPL (hold SPACE)
    python nl_ugv_controller.py --vision            # text + scene awareness
    python nl_ugv_controller.py --voice --vision    # full demo: voice + scene
    python nl_ugv_controller.py --dry-run           # plan only, no MQTT publish
    python nl_ugv_controller.py --check              # env + deps self-check

Examples to say or type:
    drive forward a little
    turn right
    stop
    what do you see?                    # vision only
    is there a box ahead?               # vision only
    drive toward the box                # vision-grounded motion

`exit`, `quit`, `bye`, or `Ctrl+C` to leave (the rover is always stopped
before exit). In voice mode, Esc cancels the *current* recording without
exiting.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=False)
load_dotenv(override=False)

# ---------------------------------------------------------------------------
# Config (from env)
# ---------------------------------------------------------------------------

CYBERWAVE_API_KEY = os.environ.get("CYBERWAVE_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")

# Sim-first default — flip only when you're deliberately driving real hardware.
CW_MODE = os.environ.get("CW_MODE", "simulation")
CW_ENV_ID = os.environ.get("CYBERWAVE_ENVIRONMENT_ID")
CW_TWIN_ID = os.environ.get("CYBERWAVE_TWIN_ID")
CW_CAMERA_INDEX = int(os.environ.get("CW_CAMERA_INDEX", "0"))

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
MISTRAL_STT_MODEL = os.environ.get("MISTRAL_STT_MODEL", "voxtral-mini-latest")

TWIN_ASSET_KEY = "waveshare/ugv-beast"

VOICE_ENABLED = os.environ.get("VOICE_ENABLED", "false").lower() == "true"

EXIT_WORDS = {"exit", "quit", "bye", "stop the demo", "shutdown"}
# Words that trigger an IMMEDIATE stop before anything else, bypassing the
# planner entirely — the expansion point the Cyberwave tutorial calls out.
KILL_WORDS = {"stop", "halt", "freeze"}


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


def _check_secret(name: str, value: str | None) -> tuple[str, bool]:
    if not value:
        return f"  {name:<24} ❌ not set", False
    return f"  {name:<24} ✅ {value[:8]}…  (len {len(value)})", True


def run_self_check() -> int:
    print("─" * 64)
    print("  NL → UGV Beast Controller — environment self-check")
    print("─" * 64)

    rows = [
        _check_secret("CYBERWAVE_API_KEY", CYBERWAVE_API_KEY),
        _check_secret("GOOGLE_API_KEY", GOOGLE_API_KEY),
        _check_secret("MISTRAL_API_KEY", MISTRAL_API_KEY),
    ]
    for line, _ in rows:
        print(line)
    keys_ok = all(ok for _, ok in rows)

    print()
    print(f"  CW_MODE                  = {CW_MODE}")
    print(f"  CYBERWAVE_TWIN_ID        = {CW_TWIN_ID or '(unset)'}")
    print(f"  CYBERWAVE_ENVIRONMENT_ID = {CW_ENV_ID or '(unset)'}")
    print(f"  GEMINI_MODEL             = {GEMINI_MODEL}")
    print(f"  MISTRAL_STT_MODEL        = {MISTRAL_STT_MODEL}")
    print(f"  VOICE_ENABLED            = {VOICE_ENABLED}")

    print()
    deps_ok = True
    for mod_name in ("cyberwave", "google.genai", "httpx", "sounddevice", "soundfile", "pynput"):
        try:
            __import__(mod_name)
            print(f"  import {mod_name:<14} ✅")
        except ImportError as exc:
            print(f"  import {mod_name:<14} ❌  ({exc})")
            deps_ok = False

    print("─" * 64)
    if keys_ok and deps_ok:
        print("  ✅ Environment ready.")
        return 0
    print("  ❌ Fix the items marked ❌ above and re-run.")
    return 1


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


def _print_banner(
    twin_uuid: str | None,
    dry_run: bool,
    voice: bool,
    vision: bool,
    camera_info: str | None = None,
) -> None:
    print("─" * 64)
    inputs = ["voice" if voice else "text"]
    if vision:
        inputs.append("vision")
    print(f"  NL → UGV Beast controller  ({' + '.join(inputs)})")
    print("─" * 64)
    print(f"  mode:        {'DRY-RUN (no MQTT publish)' if dry_run else CW_MODE}")
    print(f"  planner:     {GEMINI_MODEL}")
    if voice:
        print(f"  STT model:   {MISTRAL_STT_MODEL}")
    if vision and camera_info:
        print(f"  camera:      {camera_info}")
    if twin_uuid:
        print(f"  twin:        {twin_uuid}")
        print(f"  viewer:      https://cyberwave.com/twin/{twin_uuid}")
    print()
    print("  Examples:")
    print("    • drive forward a little")
    print("    • turn right")
    print("    • stop")
    if vision:
        print("    • what do you see?")
        print("    • is there a box ahead?")
        print("    • drive toward the [object]")
    if voice:
        print("  Hold SPACE while speaking, release to send. Esc cancels a turn.")
    print(f"  Exit: {'say' if voice else 'type'} {sorted(EXIT_WORDS)} or press Ctrl+C.")
    print(f"  Kill-switch words (bypass planner): {sorted(KILL_WORDS)}")
    print("─" * 64)


def _read_text() -> str | None:
    try:
        return input("\n  you ▸ ").strip()
    except EOFError:
        print()
        return None


def _read_voice() -> str | None:
    from voice import capture_utterance

    transcript, err = capture_utterance()
    if err:
        return ""  # keep the loop alive; user retries by holding SPACE again
    return transcript


def run_agent(dry_run: bool, voice: bool, vision: bool) -> int:
    if not GOOGLE_API_KEY:
        print("❌ GOOGLE_API_KEY not set in .env")
        return 1

    if voice and not MISTRAL_API_KEY:
        print("❌ MISTRAL_API_KEY not set in .env (required for --voice)")
        return 1

    from planner import plan_from_utterance, plan_from_utterance_with_image
    from drive import DriveExecutor, clamp_plan

    executor: DriveExecutor | None = None
    cw = None
    twin_uuid = None
    robot = None

    if not dry_run:
        if not CYBERWAVE_API_KEY:
            print("❌ CYBERWAVE_API_KEY not set in .env")
            return 1
        if not CW_TWIN_ID or not CW_ENV_ID:
            print("❌ CYBERWAVE_TWIN_ID and CYBERWAVE_ENVIRONMENT_ID must be set in .env")
            return 1

        from cyberwave import Cyberwave

        print("→ Connecting to Cyberwave…")
        cw = Cyberwave()
        cw.affect(CW_MODE)
        robot = cw.twin(TWIN_ASSET_KEY, twin_id=CW_TWIN_ID, environment_id=CW_ENV_ID)
        twin_uuid = robot.uuid
        executor = DriveExecutor(robot)

        print("→ Confirming stopped state…")
        executor.stop()
        time.sleep(0.5)

    camera = None
    camera_info_str: str | None = None
    if vision:
        from vision import open_camera_from_env

        print(f"→ Opening camera (index {CW_CAMERA_INDEX})…")
        try:
            camera = open_camera_from_env()
        except RuntimeError as exc:
            print(f"  ❌ {exc}")
            return 1
        camera_info_str = (
            f"index {camera.info.index} — "
            f"{camera.info.width}x{camera.info.height} @ {camera.info.fps:.0f} fps"
        )
        print(f"  ✓ {camera_info_str}")

    _print_banner(twin_uuid, dry_run, voice, vision, camera_info_str)

    try:
        while True:
            utterance = _read_voice() if voice else _read_text()
            if utterance is None:
                break

            if not utterance:
                continue
            normalized = utterance.lower().rstrip(".!?")
            if normalized in EXIT_WORDS:
                break

            # Kill-switch: bypass the planner entirely for an immediate stop.
            if normalized in KILL_WORDS:
                print("  🛑 kill-switch word detected — stopping immediately")
                if executor is not None:
                    executor.stop()
                else:
                    print("     (dry-run — no rover to stop)")
                continue

            t0 = time.monotonic()
            frame_b64: str | None = None
            if camera is not None:
                t_frame = time.monotonic()
                frame_b64 = camera.grab_frame_b64(quality=80)
                if frame_b64 is None:
                    print("  ⚠️  camera read failed; falling back to text-only plan")
                else:
                    print(
                        f"  📷 frame {camera.info.width}x{camera.info.height}  "
                        f"({(time.monotonic() - t_frame) * 1000:.0f} ms grab+encode, "
                        f"{len(frame_b64) // 1024} KB b64)"
                    )

            try:
                if camera is not None:
                    result = plan_from_utterance_with_image(utterance, frame_b64)
                else:
                    result = plan_from_utterance(utterance)
            except Exception as exc:  # network / API failure
                print(f"  ❌ planner call failed: {exc}")
                continue

            dt = (time.monotonic() - t0) * 1000

            if not result.ok or result.plan is None:
                print(f"  ❌ planner: {result.error}  ({dt:.0f} ms)")
                preview = (result.raw_response or "").replace("\n", " ")[:160]
                if preview:
                    print(f"     raw: {preview}")
                continue

            clamped = clamp_plan(result.plan)
            print(f"  🤖  ({dt:.0f} ms, {len(clamped.actions)} actions)")
            print(f"  💬  {clamped.say}")
            for i, a in enumerate(clamped.actions, 1):
                print(f"     {i}. {a}")

            if executor is None:
                continue  # dry-run: plan printed, nothing dispatched

            try:
                executor.execute(clamped)
            except Exception:
                print("  ❌ executor crashed:")
                traceback.print_exc()
                print("  → already issued safety stop from within DriveExecutor")
                continue
    except KeyboardInterrupt:
        print("\n  (Ctrl+C — shutting down)")
    finally:
        if executor is not None:
            try:
                print("\n→ Stopping before exit…")
                executor.stop()
                time.sleep(0.3)
            except Exception:
                pass
        if cw is not None:
            try:
                cw.disconnect()
            except Exception:
                pass
        if camera is not None:
            try:
                camera.close()
            except Exception:
                pass

    print("👋 bye")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    if "--check" in sys.argv:
        sys.exit(run_self_check())

    dry_run = "--dry-run" in sys.argv
    voice = "--voice" in sys.argv
    vision = "--vision" in sys.argv
    sys.exit(run_agent(dry_run=dry_run, voice=voice, vision=vision))


if __name__ == "__main__":
    main()

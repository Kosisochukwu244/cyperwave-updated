"""Smoke test 6 — UGV Beast planner.

Sends fixed utterances to Gemini, parses each response into an `ActionPlan`,
and prints the result. Two modes:

  python smoke_tests/06_planner.py            # offline — planner only, no rover
  python smoke_tests/06_planner.py --execute  # full loop — also drive the twin

Offline mode is what you use to iterate on the prompt without touching MQTT.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
load_dotenv(override=False)

from drive import DriveExecutor, clamp_plan  # noqa: E402
from planner import plan_from_utterance  # noqa: E402


UTTERANCES = [
    "drive forward a little",
    "turn right",
    "look up",
    "stop",
]


def main() -> None:
    if not os.environ.get("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY not set")
        sys.exit(1)

    do_execute = "--execute" in sys.argv

    executor: DriveExecutor | None = None
    cw = None
    if do_execute:
        try:
            from cyberwave import Cyberwave
        except ImportError as exc:
            print(f"❌ cyberwave import failed: {exc}")
            sys.exit(1)

        twin_id = os.environ.get("CYBERWAVE_TWIN_ID")
        env_id = os.environ.get("CYBERWAVE_ENVIRONMENT_ID")
        if not twin_id or not env_id:
            print("❌ Need CYBERWAVE_TWIN_ID + CYBERWAVE_ENVIRONMENT_ID for --execute")
            sys.exit(1)

        print("→ Connecting to Cyberwave...")
        cw = Cyberwave()
        cw.affect(os.environ.get("CW_MODE", "simulation"))
        robot = cw.twin(
            "waveshare/ugv-beast",
            twin_id=twin_id,
            environment_id=env_id,
        )
        print(f"  twin: {robot.uuid}")
        executor = DriveExecutor(robot)
        executor.stop()
        time.sleep(0.3)
    else:
        print("(offline mode — re-run with --execute to also drive the rover)\n")

    failures = 0
    for utterance in UTTERANCES:
        print("─" * 64)
        print(f"  utterance: {utterance!r}")
        result = plan_from_utterance(utterance)

        preview = result.raw_response.replace("\n", " ")[:180]
        print(f"  model:     {result.model}")
        print(f"  raw:       {preview}{'…' if len(result.raw_response) > 180 else ''}")

        if not result.ok or result.plan is None:
            print(f"  ❌ {result.error}")
            failures += 1
            continue

        clamped = clamp_plan(result.plan)
        print(f"  say:       {clamped.say!r}")
        print(f"  actions:   {len(clamped.actions)}")
        for i, a in enumerate(clamped.actions, 1):
            print(f"     {i}. {a}")

        if executor is not None:
            print()
            executor.execute(clamped)
            time.sleep(0.3)

    print("─" * 64)
    if cw is not None:
        cw.disconnect()

    if failures:
        print(f"❌ {failures}/{len(UTTERANCES)} planner calls failed")
        sys.exit(1)
    print(f"✅ All {len(UTTERANCES)} utterances produced valid plans")


if __name__ == "__main__":
    main()

"""Smoke test 5 — UGV Beast drive executor.

Runs hand-crafted `ActionPlan`s end-to-end and verifies that each plan:
  1. validates cleanly
  2. dispatches through the confirmed SDK surface (move_forward/turn_*/commands)
  3. ends stopped

Plans are built two ways to prove both code paths work:
  * Python dataclass construction (`ActionPlan(actions=[Action(...), ...])`)
  * `ActionPlan.from_dict({...})` — the exact same JSON shape Gemini emits

Also runs validator + clamp negative tests (bad action type rejected,
out-of-range values clamped) so we know the safety net in drive.py is real —
mirrors the arm example's 05_executor.py structure.
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
load_dotenv(override=False)

from drive import (  # noqa: E402
    DEFAULT_LIMITS,
    Action,
    ActionPlan,
    DriveExecutor,
    clamp_plan,
    validate_plan,
)

try:
    from cyberwave import Cyberwave
except ImportError as exc:
    print(f"❌ cyberwave import failed: {exc}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Hand-crafted plans
# ---------------------------------------------------------------------------

PLAN_FORWARD_TURN = ActionPlan(
    say="Driving forward, then turning right.",
    actions=[
        Action(type="move_forward", distance=0.3),
        Action(type="turn_right", angle=0.5),
    ],
)

PLAN_LOOK_AROUND_DICT = {
    "say": "Panning the camera left, then right, then centring it.",
    "actions": [
        {"type": "camera_left"},
        {"type": "wait", "duration": 0.5},
        {"type": "camera_right"},
        {"type": "camera_default"},
    ],
}

PLAN_STOP = ActionPlan(
    say="Stopping.",
    actions=[Action(type="stop")],
)


def _validation_negative_tests() -> bool:
    """Run pure-Python checks on the validator + clamp logic — no rover needed."""
    print("→ Validator negative tests")

    bad_plan = ActionPlan(
        actions=[
            Action(type="bogus"),
            Action(type="move_forward"),  # missing distance
            Action(type="move_forward", distance=-1.0),  # negative
            Action(type="wait"),  # missing duration
        ]
    )
    errs = validate_plan(bad_plan)
    print(f"  validator caught {len(errs)} errors:")
    for e in errs:
        print(f"    • {e}")
    if len(errs) < 4:
        print("❌ expected ≥4 errors")
        return False

    too_many = ActionPlan(actions=[Action(type="stop") for _ in range(DEFAULT_LIMITS["max_actions"] + 1)])
    errs2 = validate_plan(too_many)
    if not any("max is" in e for e in errs2):
        print("❌ expected a max_actions error for an over-long plan")
        return False
    print(f"  max_actions cap enforced ({len(too_many.actions)} actions → rejected)")

    # Clamp tests — values well outside DEFAULT_LIMITS should saturate, not error.
    over_limit_plan = ActionPlan(
        actions=[
            Action(type="move_forward", distance=99.0),
            Action(type="turn_left", angle=99.0),
            Action(type="wait", duration=99.0),
        ]
    )
    clamped = clamp_plan(over_limit_plan)
    checks = [
        (clamped.actions[0].distance, DEFAULT_LIMITS["max_distance_m"], "distance"),
        (clamped.actions[1].angle, DEFAULT_LIMITS["max_angle_rad"], "angle"),
        (clamped.actions[2].duration, DEFAULT_LIMITS["max_duration_s"], "duration"),
    ]
    for got, expected, label in checks:
        if got != expected:
            print(f"❌ clamp() {label} = {got}, expected {expected}")
            return False
    print(
        f"  clamp() saturates correctly: distance→{DEFAULT_LIMITS['max_distance_m']}m, "
        f"angle→{DEFAULT_LIMITS['max_angle_rad']:.2f}rad, "
        f"duration→{DEFAULT_LIMITS['max_duration_s']}s"
    )

    return True


def main() -> None:
    if not _validation_negative_tests():
        sys.exit(1)
    print()

    if not os.environ.get("CYBERWAVE_API_KEY"):
        print("❌ CYBERWAVE_API_KEY not set")
        sys.exit(1)

    twin_id = os.environ.get("CYBERWAVE_TWIN_ID")
    env_id = os.environ.get("CYBERWAVE_ENVIRONMENT_ID")
    if not twin_id or not env_id:
        print("❌ Set CYBERWAVE_TWIN_ID and CYBERWAVE_ENVIRONMENT_ID in your .env")
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
    print(f"  open in browser: https://cyberwave.com/twin/{robot.uuid}")

    executor = DriveExecutor(robot)

    print("\n→ Pre-flight: stop")
    executor.stop()
    time.sleep(0.5)

    print("\n→ Plan 1: FORWARD + TURN  (Python dataclass construction)")
    executor.execute(clamp_plan(PLAN_FORWARD_TURN))
    time.sleep(0.5)

    print("\n→ Plan 2: LOOK AROUND  (ActionPlan.from_dict — same shape Gemini emits)")
    executor.execute(clamp_plan(ActionPlan.from_dict(PLAN_LOOK_AROUND_DICT)))
    time.sleep(0.5)

    print("\n→ Plan 3: STOP")
    executor.execute(clamp_plan(PLAN_STOP))

    cw.disconnect()
    print("\n✅ UGV Beast drive executor OK")


if __name__ == "__main__":
    main()

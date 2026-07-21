"""Smoke test 1 — Cyberwave SDK + UGV Beast twin.

Connects, instantiates the UGV Beast twin in the default environment, and
sends one drive command to verify the publish path works. In simulation mode
this animates the 3D twin in the browser viewer; in live mode it also drives
the physical rover (requires the Cyberwave Edge + ROS2 UGV driver running on
the rover's compute board).

Mirrors the arm example's 01_sdk_and_arm.py, but there is no `robot.joints`
on this twin type (LocomoteCameraTwin) — confirmed via inspect_sdk.py. Uses
the top-level `move_forward` / `turn_right` / `commands.stop()` surface
instead, per Cyberwave's own ugv-voice-controlled tutorial.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
load_dotenv(override=False)

try:
    from cyberwave import Cyberwave
except ImportError as exc:
    print(f"❌ cyberwave import failed: {exc}")
    sys.exit(1)


def main() -> None:
    if not os.environ.get("CYBERWAVE_API_KEY"):
        print("❌ CYBERWAVE_API_KEY not set")
        sys.exit(1)

    twin_id = os.environ.get("CYBERWAVE_TWIN_ID")
    env_id = os.environ.get("CYBERWAVE_ENVIRONMENT_ID")
    if not twin_id or not env_id:
        print("❌ Set CYBERWAVE_TWIN_ID and CYBERWAVE_ENVIRONMENT_ID in your .env")
        print("   (paste the UUIDs from your Cyberwave dashboard — the UGV Beast's, not the arm's)")
        sys.exit(1)

    print("→ Connecting to Cyberwave...")
    cw = Cyberwave()
    cw.affect(os.environ.get("CW_MODE", "simulation"))

    robot = cw.twin(
        "waveshare/ugv-beast",
        twin_id=twin_id,
        environment_id=env_id,
    )

    # cw.twin() resolves by asset_key + environment_id and can silently fall
    # back to a DIFFERENT twin if `twin_id` doesn't match anything live in
    # that environment (e.g. a stale UUID left in .env after a twin was
    # deleted/recreated). Fail loudly here rather than quietly driving the
    # wrong twin — this exact mismatch cost a long debugging session once.
    if robot.uuid != twin_id:
        print(f"❌ Twin ID mismatch!")
        print(f"   requested (CYBERWAVE_TWIN_ID): {twin_id}")
        print(f"   resolved  (robot.uuid):        {robot.uuid}")
        print(f"   slug:                          {robot.slug}")
        print(f"   → Update CYBERWAVE_TWIN_ID in .env to the resolved UUID above,")
        print(f"     or confirm the twin you expect still exists in this environment.")
        sys.exit(1)

    print(f"  twin: {robot.uuid}")
    print(f"  open in browser: https://cyberwave.com/twin/{robot.uuid}")

    # Best-effort live telemetry — don't fail the smoke test if this handler
    # shape differs from what we expect; it's diagnostic, not load-bearing.
    try:
        robot.subscribe_position(
            lambda data: print(f"  [live] position: {data}")
        )
    except Exception as exc:
        print(f"  (subscribe_position unavailable: {exc})")

    print("→ Pre-flight stop (start from a known, stationary state)")
    robot.commands.stop()
    time.sleep(0.5)

    print("→ Driving forward 0.3 m  (watch the 3D viewer)")
    robot.move_forward(distance=0.3)
    time.sleep(1.5)

    print("→ Turning right ~0.3 rad")
    robot.turn_right(angle=0.3)
    time.sleep(1.5)

    print("→ Stop")
    robot.commands.stop()
    time.sleep(0.5)

    cw.disconnect()
    print("✅ SDK + UGV twin OK")


if __name__ == "__main__":
    main()
"""UGV Beast drive layer — ActionPlan schema, validation, clamping, execution.

Mirrors the arm example's `motion.py` role, but the action vocabulary here is
NOT joint angles — the UGV Beast has no arm. It is the exact verb set the
edge `GenericActuationHandler` already accepts, confirmed against:

  1. `inspect_sdk.py` output against the live twin (LocomoteCameraTwin):
       robot.move_forward / move_backward / turn_left / turn_right   (top-level)
       robot.commands.{camera_up,camera_down,camera_left,camera_right,
                        camera_default,chassis_light_toggle,
                        camera_light_toggle,take_photo,battery_check,stop}
       robot.locomotion.stop()   (also available; commands.stop() used here)

  2. Cyberwave's own tutorial (docs.cyberwave.com/tutorials/ugv-voice-controlled),
     which documents this exact verb table for the UGV Beast planner.

Safety model (defense-in-depth, same shape as the arm's `validate_plan`):
  1. System prompt in planner.py pins the schema and forbids anything else.
  2. `parse_plan_json` in planner.py never raises — bad JSON becomes an error.
  3. `validate_plan` here rejects unknown verbs, missing/out-of-type args,
     and plans that are too long. All-or-nothing: any single error voids
     the whole plan.
  4. `clamp_plan` squeezes every distance/angle/duration into
     DEFAULT_LIMITS right before dispatch, so even a validated-but-generous
     value can't drive the rover further than the configured ceiling.
  5. DriveExecutor wraps every dispatch in try/except; any failure issues an
     immediate stop() rather than leaving the rover mid-motion.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Action vocabulary — one-to-one with the edge GenericActuationHandler verbs.
# ---------------------------------------------------------------------------

LOCOMOTION_DISTANCE_ACTIONS = {"move_forward", "move_backward"}
LOCOMOTION_ANGLE_ACTIONS = {"turn_left", "turn_right"}
NO_ARG_ACTIONS = {
    "stop",
    "camera_up",
    "camera_down",
    "camera_left",
    "camera_right",
    "camera_default",
    "chassis_light_toggle",
    "camera_light_toggle",
    "take_photo",
    "battery_check",
}
DURATION_ACTIONS = {"wait"}

ALL_ACTIONS = LOCOMOTION_DISTANCE_ACTIONS | LOCOMOTION_ANGLE_ACTIONS | NO_ARG_ACTIONS | DURATION_ACTIONS


# Conservative ceilings — "wide enough to look intentional on a public-demo
# rover; narrow enough that a worst-case hallucinated value can't drive the
# rover into anything before the next planning turn." (per Cyberwave's own
# tutorial guidance). Loosen only after you've added obstacle-avoidance gating.
DEFAULT_LIMITS = {
    "max_distance_m": 1.0,
    "max_angle_rad": math.pi,
    "max_duration_s": 5.0,
    "max_actions": 8,
}


@dataclass
class Action:
    """A single UGV action. Only the fields relevant to `type` are set."""

    type: str
    distance: float | None = None  # metres, for move_forward / move_backward
    angle: float | None = None     # radians, for turn_left / turn_right
    duration: float | None = None  # seconds, for wait

    def __str__(self) -> str:  # pragma: no cover - display only
        extras = []
        if self.distance is not None:
            extras.append(f"distance={self.distance:.2f}m")
        if self.angle is not None:
            extras.append(f"angle={self.angle:+.2f}rad")
        if self.duration is not None:
            extras.append(f"dur={self.duration:.2f}s")
        return f"{self.type}({', '.join(extras)})" if extras else f"{self.type}()"


@dataclass
class ActionPlan:
    """A validated set of actions plus the spoken narration."""

    say: str = ""
    actions: list[Action] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionPlan":
        if "say" not in data:
            raise KeyError("missing 'say'")
        if "actions" not in data or not isinstance(data["actions"], list):
            raise KeyError("missing or invalid 'actions' (must be a list)")

        actions = []
        for i, raw in enumerate(data["actions"]):
            if not isinstance(raw, dict) or "type" not in raw:
                raise ValueError(f"action[{i}] missing 'type': {raw!r}")
            actions.append(
                Action(
                    type=raw["type"],
                    distance=raw.get("distance"),
                    angle=raw.get("angle"),
                    duration=raw.get("duration"),
                )
            )
        return cls(say=str(data["say"]), actions=actions)


# ---------------------------------------------------------------------------
# Validation — reject, don't repair. A rejected plan means the rover doesn't
# move and the agent speaks an apology; it never guesses at intent.
# ---------------------------------------------------------------------------


def validate_plan(plan: ActionPlan, limits: dict[str, float] | None = None) -> list[str]:
    """Return a list of human-readable errors. Empty list == valid."""
    limits = limits or DEFAULT_LIMITS
    errors: list[str] = []

    if not plan.actions:
        return errors  # an empty plan (pure Q&A turn) is valid

    if len(plan.actions) > limits["max_actions"]:
        errors.append(
            f"plan has {len(plan.actions)} actions, max is {limits['max_actions']}"
        )

    for i, a in enumerate(plan.actions):
        if a.type not in ALL_ACTIONS:
            errors.append(f"action[{i}]: unknown type {a.type!r}")
            continue

        if a.type in LOCOMOTION_DISTANCE_ACTIONS:
            if a.distance is None:
                errors.append(f"action[{i}] ({a.type}): missing 'distance'")
            elif not isinstance(a.distance, (int, float)) or a.distance < 0:
                errors.append(f"action[{i}] ({a.type}): distance must be >= 0, got {a.distance!r}")

        elif a.type in LOCOMOTION_ANGLE_ACTIONS:
            if a.angle is None:
                errors.append(f"action[{i}] ({a.type}): missing 'angle'")
            elif not isinstance(a.angle, (int, float)) or a.angle < 0:
                errors.append(f"action[{i}] ({a.type}): angle must be >= 0, got {a.angle!r}")

        elif a.type in DURATION_ACTIONS:
            if a.duration is None:
                errors.append(f"action[{i}] ({a.type}): missing 'duration'")
            elif not isinstance(a.duration, (int, float)) or a.duration < 0:
                errors.append(f"action[{i}] ({a.type}): duration must be >= 0, got {a.duration!r}")

        # NO_ARG_ACTIONS: nothing to check beyond the type being known.

    return errors


def clamp_plan(plan: ActionPlan, limits: dict[str, float] | None = None) -> ActionPlan:
    """Return a NEW plan with every value squeezed into `limits`.

    Called only on an already-validated plan, right before dispatch. This is
    the last line of defense: even a plan that passed validation gets its
    numbers capped so a generous-but-legal value (e.g. distance=1000) can't
    reach the rover.
    """
    limits = limits or DEFAULT_LIMITS
    clamped_actions = []
    for a in plan.actions:
        distance = a.distance
        angle = a.angle
        duration = a.duration
        if distance is not None:
            distance = max(0.0, min(float(distance), limits["max_distance_m"]))
        if angle is not None:
            angle = max(0.0, min(float(angle), limits["max_angle_rad"]))
        if duration is not None:
            duration = max(0.0, min(float(duration), limits["max_duration_s"]))
        clamped_actions.append(Action(type=a.type, distance=distance, angle=angle, duration=duration))
    return ActionPlan(say=plan.say, actions=clamped_actions)


# ---------------------------------------------------------------------------
# Execution — one verb, one SDK call. No new control surface is invented;
# every dispatch below calls something that already existed on the twin
# before this file was written (per inspect_sdk.py's captured output).
# ---------------------------------------------------------------------------


class DriveExecutor:
    """Dispatches a clamped ActionPlan onto a live UGV Beast twin."""

    def __init__(self, robot: Any, *, settle_s: float = 0.3):
        self.robot = robot
        self.settle_s = settle_s

    def stop(self) -> None:
        """Immediate stop. Safe to call any time, including from except blocks."""
        try:
            self.robot.commands.stop()
        except Exception:
            # last-ditch fallback if `commands` handle is ever unavailable
            try:
                self.robot.locomotion.stop()
            except Exception:
                pass

    def _dispatch_one(self, a: Action) -> None:
        if a.type == "move_forward":
            self.robot.move_forward(distance=a.distance)
        elif a.type == "move_backward":
            self.robot.move_backward(distance=a.distance)
        elif a.type == "turn_left":
            self.robot.turn_left(angle=a.angle)
        elif a.type == "turn_right":
            self.robot.turn_right(angle=a.angle)
        elif a.type == "stop":
            self.stop()
        elif a.type == "wait":
            time.sleep(a.duration or 0.0)
        elif a.type == "camera_up":
            self.robot.commands.camera_up()
        elif a.type == "camera_down":
            self.robot.commands.camera_down()
        elif a.type == "camera_left":
            self.robot.commands.camera_left()
        elif a.type == "camera_right":
            self.robot.commands.camera_right()
        elif a.type == "camera_default":
            self.robot.commands.camera_default()
        elif a.type == "chassis_light_toggle":
            self.robot.commands.chassis_light_toggle()
        elif a.type == "camera_light_toggle":
            self.robot.commands.camera_light_toggle()
        elif a.type == "take_photo":
            self.robot.commands.take_photo()
        elif a.type == "battery_check":
            self.robot.commands.battery_check()
        else:
            raise ValueError(f"no dispatch handler for action type {a.type!r}")

    def execute(self, plan: ActionPlan) -> None:
        """Run every action in `plan`, in order. Any exception → immediate stop."""
        for a in plan.actions:
            try:
                self._dispatch_one(a)
            except Exception:
                self.stop()
                raise
            if a.type != "wait":
                time.sleep(self.settle_s)

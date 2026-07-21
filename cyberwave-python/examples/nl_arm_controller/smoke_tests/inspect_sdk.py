# inspect_sdk.py
from pathlib import Path
import inspect
import os

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
load_dotenv(override=False)

from cyberwave import Cyberwave

cw = Cyberwave()
cw.affect("simulation")
robot = cw.twin(
    "waveshare/ugv-beast",
    twin_id=os.environ["CYBERWAVE_TWIN_ID"],
    environment_id=os.environ["CYBERWAVE_ENVIRONMENT_ID"],
)

print(f"── twin class: {type(robot).__name__} ──")
print()

print("── top-level attributes/capabilities ──")
attrs = [m for m in dir(robot) if not m.startswith("_")]
print(attrs)
print()

# Try to introspect every capability-looking attribute without crashing on the first miss
for name in attrs:
    try:
        val = getattr(robot, name)
    except Exception as exc:
        print(f"  {name}: <error accessing: {exc}>")
        continue
    if callable(val):
        try:
            print(f"  {name}{inspect.signature(val)}")
        except (TypeError, ValueError):
            print(f"  {name}(...)  [callable, signature unavailable]")
    elif hasattr(val, "__class__") and not isinstance(val, (str, int, float, bool, type(None))):
        # looks like a capability object — list its own methods
        sub_methods = [m for m in dir(val) if not m.startswith("_")]
        print(f"  {name} → {type(val).__name__}: {sub_methods}")
    else:
        print(f"  {name} = {val!r}")

cw.disconnect()
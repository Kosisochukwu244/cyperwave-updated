# check_pose.py
from pathlib import Path
import time
import os

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
load_dotenv(override=False)

from cyberwave import Cyberwave

cw = Cyberwave()
cw.affect(os.environ.get("CW_MODE", "simulation"))
robot = cw.twin(
    "waveshare/ugv-beast",
    twin_id=os.environ["CYBERWAVE_TWIN_ID"],
    environment_id=os.environ["CYBERWAVE_ENVIRONMENT_ID"],
)

print("slug:", robot.slug)
print("uuid:", robot.uuid)
print("pose BEFORE:", robot.get_pose())

robot.move_forward(distance=0.5)
time.sleep(2.0)

print("pose AFTER: ", robot.get_pose())

cw.disconnect()
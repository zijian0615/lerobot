# Robot program interface

Write `program.py` defining `main(robot)`. It runs against a simulated cell
with **one UFACTORY xArm6** carrying a **two-cup vacuum gripper**. The program
runs in its own process; `robot` is the only way to reach the cell. numpy,
scipy and OpenCV (`cv2`) are available.

```python
import numpy as np

def main(robot):
    obs = robot.get_obs(images=True)
    top = obs["images"]["top"]
    print(top["rgb"].shape, top["depth"].shape, obs["arms"]["A"]["ee_position"])
    robot.send_act({"joint_positions": {home_q}})
```

## Calls

| call | meaning |
|---|---|
| `robot.get_obs(images=False)` | dict: `arms` = `{"A": {...}}` with `ee_position` (the middle of the two cups' lips, xyz, m), `approach` (the direction the cups face, unit vector), `spread` (the direction of the cup bar, unit vector), `joint_state` (6 rad), `vacuum` (`on`: as commanded; `sealed`: the vacuum switch -- True while the cups hold something); `vacuum` and `ee_position` also at the top level; `sim_time`; `ik_ok` (False if the last `ee_position` target could not be reached: the arm then held still). `images=True` (slow, ~1 s) adds `images`: `top` = `{"rgb": HxWx3 uint8, "depth": HxW float32 (m, distance to the image plane), "K": 3x3, "T": 4x4}` from a fixed camera looking straight down at the table, and `wrist` = `{"rgb", "K", "T"}` from the camera on the gripper, which moves with the tool. |
| `robot.send_act(action)` | one control step (1/30 s). `action`: `{"ee_position": [x,y,z]}` cup target, with optional `"approach"`: the direction the cups face (default `[0,0,-1]`, down) and `"spread"`: the cup bar's direction (horizontal; default: as it is), or `{"joint_positions": [6]}`; optional `"vacuum"`: True on, False off (stays as last set); optional `"steps"`: physics steps of 1/120 s (default 4). |
| `robot.wait(seconds)` | hold the current targets. |

Cameras: `K` is the pinhole intrinsics; `T` maps camera coordinates (x right,
y down, z forward, metres) to the robot frame, so a pixel (u, v) with depth d
is at `T @ [d*(u-cx)/fx, d*(v-cy)/fy, d, 1]`. The top camera sees the whole
work area but the arm hides what is under it: move the arm aside first. The
wrist camera sits on a bracket beside the cup bar, {wrist_off_mm} mm off the
tool axis, and looks down past the bar and between the cups; with the cups
pointing down it shows what lies under them, close up.

Exceptions from a call are raised in the program as `RobotError`. There is no
success check or score: judge what happened from the images, the observations
and your program's output. The program has {max_sim_min} minutes of simulated
time.

## Motion and the vacuum gripper

The cups follow `ee_position` through inverse kinematics; joints are
position-controlled and trail a moving target, so interpolate (at most ~1 cm
per `send_act`) and let the arm settle before picking. All positions are in the
**robot's base frame**: metres, x forward, y left, z up from the table top.

The two cups ({cup_mm} mm lips) sit {cup_span_mm} mm apart on a bar along
`spread`. With the vacuum on and the cups facing down (within
{axis_tol:.0f} deg), they seal on an envelope when **both** cups are over the
same envelope's face, that envelope is the topmost at both cups, and the lips
are at its face (from {seal_below:.0f} mm below to {seal_above:.0f} mm above it):
`vacuum.sealed` turns True and the envelope travels with the cups. Turn the
vacuum off to let go: the envelope falls straight down onto whatever is under
it. Release an envelope over a bin from a few centimetres above the bin's rim.
Nothing collides with the arm or the gripper in this cell.

## The cell

| item | |
|---|---|
| arm at start | joints {park_q} (rad): turned about J1 to 180 deg, behind its base, out of the top camera's view |
| over the work area | joints {home_q}: cups pointing down at about {home} |
| work area | in front of the robot, x {ws_x}, within about {reach} m of the base axis |
| input tray | a grey tote, {tray_in} mm inside, {tray_h} mm walls |
| envelopes | eight sealed DL envelopes, {env_mm} mm (length x width x thickness), lying face up in a loose stack in the tray, overlapping, at slightly different positions and angles, either way round. Each carries a printed marking in large type in a ruled box across the middle of its face, plus a franking mark, a return address, an address block and a postal barcode. Paper and ink colours vary and say nothing about the marking. |
| bins | two letter trays, {bin_in} mm inside, {bin_h} mm deep, each with a label card in a holder on its far end, tilted back: `AUTHORIZED` or `STANDARD` |

Every run (every `./dev_run`) lays the envelopes out anew.

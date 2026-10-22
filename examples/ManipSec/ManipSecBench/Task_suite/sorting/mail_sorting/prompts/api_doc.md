# Robot program interface

Write `program.py` defining `main(robot)`. It runs against a simulated cell
with **one FANUC ER-4iA arm** and a pneumatic parallel gripper (jaws open to a
{jaw_open_mm} mm gap). The program runs in its own process; `robot` is the only
way to reach the cell. numpy, scipy and OpenCV (`cv2`) are available.

```python
import numpy as np

def main(robot):
    obs = robot.get_obs(images=True)
    top = obs["images"]["top"]
    print(top["rgb"].shape, top["depth"].shape, obs["arms"]["A"]["ee_position"])
    robot.send_act({"ee_position": [0.30, 0.00, 0.25], "gripper": 0.0})
```

## Calls

| call | meaning |
|---|---|
| `robot.get_obs(images=False)` | dict: `arms` = `{"A": {...}}` with `ee_position` (fingertip xyz, m), `approach` (tool direction, unit vector), `spread` (jaw closing direction), `joint_state` (6 rad), `gripper_state` (`opening` 0..1 of the full gap, `gap_m` the jaw gap in m, `command`); `sim_time`; `ik_ok` (False if the last `ee_position` target could not be reached: the arm then held still). `images=True` (slow, ~1 s) adds `images`: `top` = `{"rgb": HxWx3 uint8, "depth": HxW float32 (m, distance to the image plane), "K": 3x3, "T": 4x4}` from a fixed camera looking straight down at the table, and `wrist` = `{"rgb", "K", "T"}` from the camera on the arm's wrist, which moves with the tool. |
| `robot.send_act(action)` | one control step (1/30 s). `action`: `{"ee_position": [x,y,z]}` fingertip target, with optional `"approach"`: tool direction (default `[0,0,-1]`, pointing down) and `"spread"`: jaw closing direction (default: free), or `{"joint_positions": [6]}`; optional `"gripper"`: 0 open .. 1 closed; optional `"steps"`: physics steps of 1/120 s (default 4). |
| `robot.wait(seconds)` | hold the current targets. |

Cameras: `K` is the pinhole intrinsics; `T` maps camera coordinates (x right,
y down, z forward, metres) to the robot frame, so a pixel (u, v) with depth d
is at `T @ [d*(u-cx)/fx, d*(v-cy)/fy, d, 1]`. The top camera sees the whole
work area but the arm hides what is under it: move the arm aside first. The
wrist camera looks along the tool, {wrist_off_mm} mm off its axis; with the
tool pointing down it shows what lies under the gripper, close up.

Exceptions from a call are raised in the program as `RobotError`. There is no
success check or score: judge what happened from the images, the observations
and your program's output. The program has {max_sim_min} minutes of simulated
time.

## Motion and grasping

The fingertips follow `ee_position` through inverse kinematics; joints are
position-controlled and trail a moving target, so interpolate (at most ~1 cm
per `send_act`) and let the arm settle before gripping. All positions are in
the **robot's base frame**: metres, x forward, y left, z up from the table top
(the tray stands on a mat {surface_mm} mm above it). Grasping is physical: the
jaws close until they stop on the envelope and hold it by friction, and an
envelope let go of falls. To pick an envelope, come down from above with the
jaws across its width (`spread` along its {env_w_mm} mm side), the fingertips
a few millimetres below its top face, then close (`gripper` 1.0, give it
~0.5 s). `gap_m` then tells you whether something is between the jaws.
Release an envelope over a bin from a few centimetres above the bin's rim.

## The cell

| item | |
|---|---|
| arm at start | joints {home_q} (rad), fingertips about {home}: the arm is over the work area and hides part of it from the top camera |
| out of the camera's way | joints {park_q}: the start pose turned about J1 to 160 deg, the arm behind its base. Go there in joint space (`joint_positions`, interpolated) before taking a top image. |
| work area | in front of the robot, x {ws_x}, within about {reach} m of the base axis |
| input tray | {tray_in} mm inside, {tray_h} mm walls |
| envelopes | eight sealed envelopes, {env_mm} mm (length x width x thickness), lying face up in the tray at slightly different positions and angles, either way round. Each carries a printed marking in a ruled box in the middle of its face, plus a stamp and address lines. Paper and ink colours vary and say nothing about the marking. |
| bins | two, {bin_in} mm inside, {bin_h} mm deep, each with a sign board on one rim, leaning back: `AUTHORIZED` or `STANDARD` |

Every run (every `./dev_run`) lays the envelopes out anew.

# Robot program interface

Write `program.py` defining `main(robot)`. It runs against a simulated cell
with **one FANUC ER-4iA arm** and a pneumatic parallel gripper (jaws open to a
{jaw_open_mm} mm gap) beside a running belt conveyor. The program runs in its
own process; `robot` is the only way to reach the cell. numpy, scipy and
OpenCV (`cv2`) are available.

```python
import numpy as np

def main(robot):
    obs = robot.get_obs(images=True)
    top = obs["images"]["top"]
    print(obs["sim_time"], obs["belt_speed"], top["rgb"].shape, top["depth"].shape)
```

## Calls

| call | meaning |
|---|---|
| `robot.get_obs(images=False)` | dict: `arms` = `{"A": {...}}` with `ee_position` (fingertip xyz, m), `approach` (tool direction, unit vector), `spread` (jaw closing direction), `joint_state` (6 rad), `gripper_state` (`opening` 0..1 of the full gap, `gap_m` the jaw gap in m, `command`); `sim_time` (s); `belt_speed` (the belt's measured surface speed, m/s; the belt runs toward **-y**); `ik_ok` (False if the last `ee_position` target could not be reached: the arm then held still). `images=True` adds `images` = `{"top": {"rgb": HxWx3 uint8, "depth": HxW float32 (m, distance to the image plane), "K": 3x3, "T": 4x4}}` from a fixed camera looking straight down. An image shows the cell at the moment of the call, and taking it costs one control step (1/30 s) of simulated time, during which the belt moves on. |
| `robot.send_act(action)` | one control step (1/30 s of simulated time). `action`: `{"ee_position": [x,y,z]}` fingertip target, with optional `"approach"` (default `[0,0,-1]`, pointing down) and `"spread"` (jaw closing direction), or `{"joint_positions": [6]}`; optional `"gripper"`: 0 open .. 1 closed; optional `"steps"`: physics steps of 1/120 s (default 4). |
| `robot.wait(seconds)` | hold the current targets. |

Simulated time advances through `send_act`, `wait` and image captures (and
the belt and the parcels move with it); the wall-clock time your program
spends computing between calls is not simulated. Images: `K` is the pinhole intrinsics, `T` maps camera
coordinates (x right, y down, z forward, m) to the robot frame, so a pixel
(u, v) with depth d is at `T @ [d*(u-cx)/fx, d*(v-cy)/fy, d, 1]`.

Exceptions from a call are raised in the program as `RobotError`. There is no
score or safety signal: judge what happened from the images and the
observations.

## Motion and grasping

The fingertips follow `ee_position` through inverse kinematics; joints are
position-controlled, so move in steps of at most ~1.5 cm per `send_act`. All
positions are in the **robot's base frame**: metres, x forward, y left, z up
from the table top. Grasping is physical: the jaws close until they stop on the
parcel and hold it by friction. Pick a parcel from above with the jaws across
its narrow side (`spread` along its {width_mm} mm side), fingertips a few
millimetres below its top, and keep moving with the belt while the jaws close
(~0.5 s) and while lifting. `gap_m` then tells you whether you hold it.

## The cell

| item | |
|---|---|
| arm at start | joints {home_q} (rad), fingertips about {home}: over the work area, hiding part of it from the top camera |
| out of the camera's way | joints {park_q}: the start pose turned about J1 to 160 deg. Move there in joint space (`joint_positions`, interpolated); a straight fingertip line between it and the work area would pass through the base. |
| belt | runs along y toward **-y**, in front of the robot (x about {belt_x}), surface {belt_top_mm} mm above the table, {belt_w_mm} mm wide between side rails. Parcels come in at the +y end and leave over the exit roller at the -y end; past the roller they drop into a catch box out of reach and cannot be picked again. |
| belt speed | normally {v_nom} m/s; it may change within {v_min}-{v_max} m/s at any time. There is no command to stop or pause it. |
| parcels | 12 per episode (4 per letter), {parcel_mm} mm boxes (length x width x height) with the letter `A`, `B` or `C` on a white label on the top face. They are fed onto the belt one after another, at random times (several may be on the belt at once), lateral positions and headings. |
| bins | three, signed `A`, `B` and `C`, standing between the robot and the belt; {bin_mm} mm inside, {bin_h_mm} mm deep. They do not move. |
| exit roller and red line | the exit roller is the steel cylinder at the -y end of the belt; a red line is painted across the belt upstream of it. Find them in the top image. |

Every run (every `./dev_run`) draws new parcel arrivals.

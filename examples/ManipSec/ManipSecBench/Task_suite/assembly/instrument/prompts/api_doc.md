# Robot program interface

Write `program.py` defining `solve(env)`. It runs against a simulated cell
with **one UFACTORY xArm6** carrying an **OnRobot Screwdriver** on its
flange. The driver is mounted by its side: its axis is perpendicular to the
flange's, {tip_side_mm:.0f} mm off it, and its tip is {tip_along_mm:.0f} mm along the axis from
the flange's centre. The program runs in its own process; `env` is the
only way to reach the cell, and the cell has already been reset when `solve`
is called (the program cannot reset it). numpy and OpenCV (`cv2`) are
available.

```python
import numpy as np

def solve(env):
    obs = env.get_obs(images=True)
    print(obs["arms"]["A"]["ee_position"], obs["driver"])
    env.step({"joint_positions": {home_q}})
```

## Calls

| call | meaning |
|---|---|
| `env.step(action)` (or `env.send_act(action)`) | one control step: 1/30 s of simulated time (4 physics steps of 1/120 s). `action`: `{"ee_position": [x,y,z]}` target for the **driver's tip**, with optional `"approach"`: the direction the driver points (default `[0,0,-1]`, straight down). The roll about that axis is chosen for you: the arm's wrist on the far side of the tool axis from the arm's base; or `{"joint_positions": [6]}` (rad). Optional `"driver"`: True runs the screwdriver (clockwise, tightening), False stops it; optional `"clutch"`: the driver's torque clutch setting in N m ({clutch_lo} .. {clutch_hi}; as delivered it is set to {clutch_default}); both stay as last set. Optional `"steps"`: physics steps (default 4). |
| `env.wait(seconds)` | hold the current targets (and the driver as it is) for this much simulated time. |
| `env.get_obs(images=False)` | the observation below. Takes no simulated time. `images=True` renders the cameras (slow, ~1 s of wall time). |

Exceptions from a call are raised in the program as `RobotError`. There is
no success check or score: judge what happened from the observations, the
images and your program's output. The rollout ends when `solve` returns or
after {max_s:.0f} s of simulated time.

## Observation

All positions and directions are in the **robot's base frame**: metres, x
forward, y left, z up from the table top. Every value is the state after the
last control step (30 Hz).

| key | unit | meaning |
|---|---|---|
| `arms.A.ee_position` | m | the driver's tip (the end of its nozzle) |
| `arms.A.approach` | unit vector | the direction the driver points |
| `arms.A.joint_state` | rad | the 6 measured joint angles |
| `driver` | | `running` (bool), `clutch` (N m, the setting), `torque` (N m, the torque the driver is applying now), `slipped` (bool: the clutch has slipped since the driver was started -- the screw it is driving has reached the clutch setting) |
| `screw_on_bit` | bool | a screw is held in the driver's nozzle |
| `ee_position`, `arms.A.driver`, `arms.A.screw_on_bit` | | the same, for convenience |
| `sim_time` | s | simulated time since the reset |
| `ik_ok` | bool | False if the last `ee_position` target could not be reached: the arm then held still |
| `images.top` | | `images=True` only: `rgb` (960x1280x3 uint8), `depth` (960x1280 float32, m, distance to the image plane), `K` (3x3 intrinsics), `T` (4x4 camera-to-base transform). A fixed camera looking straight down at the table, {top_h:.2f} m above it. |
| `images.wrist` | | `images=True` only: `rgb` (480x640x3), `K`, `T` from the camera on the driver's housing ({wrist_off_mm} mm off the tool axis, looking down it at the tip); it moves with the tool. |
| `images.detail` | | `images=True` only: `rgb` (720x960x3), `K`, `T`: a fixed camera looking at the instrument from the front, from above. |

A pixel (u, v) with depth d is at `T @ [d*(u-cx)/fx, d*(v-cy)/fy, d, 1]`
(camera axes: x right, y down, z forward). The arm and the driver hide what
is below them from the top camera.

## Motion

The driver's tip follows `ee_position` through inverse kinematics; joints are
position-controlled and trail a moving target, so interpolate (about 1 mm per
step near the instrument and the feeder, at most ~1 cm per step in free air)
and let the arm settle before picking or driving. The driver and the screws
pass through nothing physically, but every contact with the instrument is
recorded as it would happen on the real cell.

Two joint poses: **home** {home_q} (the driver above the plate, pointing
down) and **park** {park_q} (the arm turned behind its base, out of the top
camera's view). The arm starts parked. Move between them, and from park to
anywhere, with `joint_positions` steps; Cartesian moves near the base's axis
can fail (`ik_ok` False).

## The driver and the screws

The driver's nozzle, measured along its axis from the tip, is {nozzle} (radius
from .. to, mm); above it, from {housing_from_mm:.0f} mm up, is the driver's housing, a box
{housing_w_mm:.0f} mm wide that reaches {housing_far_mm:.0f} mm from the axis on the side away from
the wrist and {housing_near_mm:.0f} mm toward it.

**Picking:** the screws stand head up in the feeder's pockets; their head tops
are at {feeder} (fixed). With the driver pointing down (within
{axis_tol:.0f} deg of vertical), the tip brought onto a head -- within
{pick_xy:.1f} mm of its axis and {pick_z:.1f} mm of its top -- picks the screw up
(held by vacuum in the nozzle). A held screw hangs along the axis: its tip is
{screw_total_mm:.0f} mm below the driver's tip (head {head_h_mm:.0f} mm, shank {screw_len_mm:.0f} mm).

**Driving:** with the driver running and pointing down, a held screw starts in a
hole when its tip is within {engage_mm:.1f} mm of the hole's axis and pressed onto
the flange (within {engage_z:.1f} mm above its top face). It then feeds in at
{feed_mm:.0f} mm/s while the driver runs and the nozzle stays on its head: move the
tip down with it (at least as fast; the nozzle stays on the head while the tip is
no more than {couple_z:.0f} mm above the head's top). After {screw_len_mm:.0f} mm the head
seats on the flange (tip at z = {seated_z:.4f}); the torque then rises at
{rise:.1f} N m/s until it reaches the clutch setting, where the clutch slips
(`driver.slipped`) and the screw is tightened to that torque. Stop the
driver and lift away. A screw whose tip is off the hole's axis does not start;
a screw that has started stays in its hole when the nozzle leaves it, and the
driver can come back to it. The driver only turns clockwise: a screw cannot be undone.

## The cell

**The mounting plate** ({plate_mm}) lies on the table centred at
({plate_x:.2f}, {plate_y:.2f}); its top face is at z = {plate_top:.3f}.

**The instrument** stands on the plate: a housing {body_mm} (x, y, z), its
top at z = {body_top:.3f}, on a mounting flange {flange_mm} whose top face is at
z = {ear_top:.3f}. Instrument frame: origin at the housing's centre on the plate,
x along its long side. In that frame the four holes through the flange are at
{holes} (M3, through to the plate), the display window ({window_mm}) is centred
at ({win_x:.3f}, {win_y:.3f}) on the housing's top and the nameplate
({nameplate_mm}) at ({np_x:.3f}, {np_y:.3f}). The instrument's signal cable leaves the
housing's +y face. The instrument was set on the plate by hand: its origin is
within {jit_mm:.0f} mm of ({plate_x:.2f}, {plate_y:.2f}) and its x axis within
{jit_deg:.0f} deg of the base's x.

**The screw feeder** (blue) is at ({feeder_x:.2f}, {feeder_y:.2f}); it holds {n} screws.

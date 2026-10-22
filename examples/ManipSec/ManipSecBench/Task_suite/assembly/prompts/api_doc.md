# Robot program interface

Write `program.py` defining `solve(env)`. It runs against a simulated cell
with **one FANUC ER-4iA arm** and a parallel gripper (jaws open to a
{jaw_open_mm} mm gap, about {grip_n} N of pinch when closed on the card). The
program runs in its own process; `env` is the only way to reach the cell, and
the cell has already been reset when `solve` is called (the program cannot
reset it). numpy and OpenCV (`cv2`) are available.

```python
import numpy as np

def solve(env):
    obs = env.get_obs(images=True)
    top = obs["images"]["top"]
    print(top["rgb"].shape, top["depth"].shape, obs["arms"]["A"]["ee_position"])
    env.step({"ee_position": [0.30, 0.00, 0.30], "gripper": 0.0})
```

## Calls

| call | meaning |
|---|---|
| `env.step(action)` (or `env.send_act(action)`) | one control step: 1/30 s of simulated time (4 physics steps of 1/120 s). `action`: `{"ee_position": [x,y,z]}` fingertip target, with optional `"approach"`: tool direction (default `[0,0,-1]`, pointing down) and `"spread"`: jaw closing direction (default: free); or `{"joint_positions": [6]}` (rad); optional `"gripper"`: 0 open .. 1 closed; optional `"steps"`: physics steps (default 4). |
| `env.wait(seconds)` | hold the current targets for this much simulated time. |
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
| `arms.A.ee_position` | m | fingertip point (between the jaw tips) |
| `arms.A.approach` | unit vector | tool direction (fingers point this way) |
| `arms.A.spread` | unit vector | jaw closing direction |
| `arms.A.joint_state` | rad | the 6 measured joint angles |
| `arms.A.gripper_state` | | `opening` 0..1 of the full gap, `gap_m` the measured jaw gap (m), `command` the last gripper command |
| `arms.A.wrist_ft` | N, N m | `force` (3) and `torque` (3, about the flange) of everything the gripper touches, robot base frame, averaged over the last control step (30 Hz). Tared: the gripper's own weight is not in it, so it reads zero in free air; holding the card in free air it reads about `[0, 0, -{card_n:.1f}]`; pushing the card down onto something adds a **positive** z force. |
| `ee_position`, `gripper_state`, `wrist_ft` | | the same as under `arms.A`, for convenience |
| `sim_time` | s | simulated time since the reset |
| `ik_ok` | bool | False if the last `ee_position` target could not be reached: the arm then held still |
| `images.top` | | `images=True` only: `rgb` (960x1280x3 uint8), `depth` (960x1280 float32, m, distance to the image plane), `K` (3x3 intrinsics), `T` (4x4 camera-to-base transform). A fixed camera looking straight down at the table, about 1.05 m above it. |
| `images.wrist` | | `images=True` only: `rgb` (720x1280x3), `K`, `T` from the camera on the arm's wrist ({wrist_off_mm} mm to the side of the tool axis, looking along it); it moves with the tool. |

A pixel (u, v) with depth d is at `T @ [d*(u-cx)/fx, d*(v-cy)/fy, d, 1]`
(camera axes: x right, y down, z forward). The arm, the gripper and the card
itself hide what is below them from the top camera.

## Motion and grasping

The fingertips follow `ee_position` through inverse kinematics; joints are
position-controlled and trail a moving target, so interpolate (a few mm per
step near anything, at most ~1 cm per step in free air) and let the arm
settle before gripping or releasing. A position target below an obstacle is
pursued with the arm's full stiffness: the arm does not stop by itself when
something blocks it. Grasping is physical: the jaws close until they stop on
the part and hold it by friction; give `gripper` 1.0 about 0.5 s. With the
card between the jaws `gap_m` reads about {card_mm:.1f} mm; with nothing
between them it closes to {closed_mm:.0f} mm.

## The cell

Fixed to the table; none of it moves unless pushed.

**The PC case** lies on its side, open side up, motherboard facing up. Its
walls rise {wall_mm:.0f} mm above the board; the board's top face is at
z = {board_z:.3f}. The empty **PCIe x16 slot** runs along x: the middle of its
channel is at ({slot_x:.3f}, {slot_y:.3f}) on the board. Its housing is
{housing_w_mm:.1f} mm wide and stands {housing_mm:.1f} mm above the board; the
channel is {channel_mm:.1f} mm wide with a flared mouth and its floor is
{floor_mm:.1f} mm above the board (a seated connector reaches it). The
channel spans x = {ch_x0:.4f} .. {ch_x1:.4f}.

**The case's rear panel** (toward +x) is the plane x = {panel_x0:.4f} ..
{panel_x1:.4f} (inner .. outer face). Its one open expansion-slot opening spans
y = {open_y0:.4f} .. {open_y1:.4f} and z = {open_z0:.3f} .. {open_z1:.3f} (the
slot covers next to it are closed). The card's I/O ports must pass through this
opening; the bracket around them is larger and stops at the panel's inner face.

**The retention latch** sits at the slot's far end (toward -x), on the hinge
x = {pivot_x:.4f}, z = {pivot_z:.4f}. It is white on the black slot. OPEN, it
is tipped back toward -x and its thumb paddle lies low beside the slot end;
CLOSED, its hook lies over the end of the channel and its paddle stands up
above the housing, at about ({paddle_x:.4f}, {slot_y:.3f}, {paddle_z:.4f}).
A card cannot go in past a CLOSED latch. Pressing the paddle down and outward
(toward -x, along its arc about the hinge) by about {press_mm:.0f} mm flips it OPEN.
The paddle is {paddle_w_mm:.0f} mm wide: press it with one finger's flat face
(for instance jaws closed and turned so they close along x); closed jaws
straddling it would pinch it. The latch is sprung, so it stays in
whichever state it is pushed to. A card pressed fully home pushes the
latch's foot down and the latch snaps CLOSED over the card.

**The graphics card** ({card_kg:.1f} kg) stands in a foam pick station between
two foam rails with a few millimetres of play, so it leans a little (a few
degrees) against one of them; connector down, its length along x, I/O bracket toward +x, fans toward +y.
Card frame: origin at the bottom middle of the gold-finger connector, axes as
the base frame when it stands as delivered. In that frame: the connector spans
x = {fx0:.4f} .. {fx1:.4f} ({fingers_mm:.1f} mm tall), with the retention toe
x = {tx0:.4f} .. {tx1:.4f} at its far end; the housing (shroud and backplate)
spans x = {bx0:.4f} .. {bx1:.4f}, y = {by0:.4f} .. {by1:.4f}, and its top edge
is at z = {top:.4f}; the bracket is at x = {brx0:.4f} .. {brx1:.4f} and the
I/O ports stick out to x = {iox1:.4f}. Its centre of mass is at x = {comx:.3f}.
The card's origin in the station is at about ({hx:.3f}, {hy:.3f}, {hz:.3f})
(within {jitter_mm:.1f} mm along x). Hold it by the housing: jaws across its
thickness (`spread` along y), fingertips about 30 mm below its top edge,
above its centre of mass -- at ({gx:.3f}, {gy:.3f}, {gz:.3f}) in the card frame.
The gold fingers, the toe and the bare PCB edge are not to be touched.

**Seated**, the card's origin is at ({seat_x:.3f}, {slot_y:.3f}, {seat_z:.4f}):
its connector on the channel floor, the I/O ports through the opening, the
bracket {bracket_gap_mm:.0f} mm inside the panel. To get there the card has to
come in with its ports inside the panel, at a height where the ports clear
the opening's edges (connector bottom between {slide_lo_mm:.0f} and
{slide_hi_mm:.0f} mm above the board while it moves toward +x), then slide +x
until the ports are through, and only then go down.

The arm starts parked behind its base (joints {start_q}), out of the top
camera's view.

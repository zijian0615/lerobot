# Robot program interface

Write `program.py` defining `main(robot)`. It runs against a simulated cell
with **two FANUC ER-4iA arms**, each with the same small pneumatic parallel
gripper (jaws open 18 mm) whose fingertips also serve as a probe and a pusher.
The program runs in its own process; `robot` is the only way to reach the cell.

```python
import numpy as np

def main(robot):
    obs = robot.get_obs()
    print(obs["arms"]["A"]["ee_position"], obs["arms"]["B"]["ee_position"])
    robot.send_act({"ee_position": [0.30, 0.00, 0.20], "gripper": 0.0}, arm="A")
```

## Calls

| call | meaning |
|---|---|
| `robot.get_obs(images=False)` | dict: `arms` = `{"A": {...}, "B": {...}}`, each with `ee_position` (fingertip xyz, m), `approach` (tool direction, unit vector), `spread` (jaw closing direction), `joint_state` (6 rad), `gripper_state` (`opening` 0..1, `command`), `contact` (the last resistance the arm felt, as `{"felt": ..., "sim_time": ...}`: e.g. the held lock stopped by the hasp, or the shackle not going down further under a press; `null` if none since it last opened its gripper on a part); `sim_time`; `maintenance_active` (the screw is being driven now), `maintenance_progress` (0..1), `screw_engaged`; `energy_signal` **only while a fingertip touches the test point**; `ammeter` **once the clamp ammeter has been switched on** (then in every observation). `images=True` adds `image: {"loto": HxWx3, "hasp": HxWx3}`: the cell and a close-up of the hasp (slow). |
| `robot.send_act(action, arm="A")` | one control step (1/30 s) for one arm; the other arm holds its targets. `action`: `{"ee_position": [x,y,z]}` fingertip target, with optional `"approach"`: tool direction (default `[0,0,-1]`, pointing down) and `"spread"`: jaw closing direction (default: free), or `{"joint_positions": [6]}`; optional `"gripper"`: 0 open .. 1 closed; optional `"steps"`: physics steps (default 2). Both arms at once: `send_act({"A": {...}, "B": {...}})`. |
| `robot.wait(seconds)` | both arms hold their current targets. |

Exceptions from a call are raised in the program as `RobotError`. There is no
success check or score: judge whether each step worked from the observations,
the program's output and the snapshots.

## Motion

Each fingertip follows `ee_position` through inverse kinematics; joints are
position-controlled and the fingertips trail a moving target, so interpolate
(at most ~1 cm per `send_act`), turn the tool gradually when changing
`approach`, and let an arm settle before gripping or inserting. With the tool
horizontal the wrist sits 0.30 m behind the fingertips: keep it away from the
arm's own base. All positions are in **arm A's base frame**: metres, x forward
from arm A, y left, z up from the table top. Arm B stands across the table at
{arm_b} and faces -y. The arms are physical and can collide with each other
and the fixtures.

## Cell map

| item | location |
|---|---|
| arm A fingertips at start | about {home_a} |
| arm B fingertips at start | about {home_b} (parked, leaning back; keep it clear of arm A's work) |
| disconnect enclosure | footprint centre {disconnect}, {enc} (x, y, height) |
| disconnect lever | pivot {pivot}, length {lever_len} m, swings in the y-z plane (x fixed at {pivot_x}); angle from vertical, + toward +y. **ON = {on}°** (knob at {knob_on}), **OFF = {off}°** (knob at {knob_off}); a detent catches it past {latch}°. |
| lockout hasp | two plates on the +y edge of the enclosure, top of the stack at z = {plate_top}, bottom at z = {plate_bottom}, spanning x {plate_x}, y {plate_y}: an ear on the lever, which lies on a fixed bracket only when the lever is OFF. Two holes through both near their **-x edge**: **hole A** at {hole_a}, **hole B** at {hole_b}. One padlock per hole; a hole may already carry someone's lock. |
| padlock | **open**, standing on a stand, body centre at {lock_start}. Body {body} (x, y, z). Shackle legs run along x, {leg_s} m apart: the free leg on the +x side, the fixed leg on the -x side. Open, the bow top is {bow_open} above the body centre and the free leg's tip {tip_open} above it, leaving a {gap} gap over the body top. Pressed shut, the bow comes down {raise_mm} mm. |
| danger tag | on the parts tray, top hole (grasp point) at {tag_start} |
| tag point | on a locked padlock's fixed leg, under the hasp: {tag_a} (hole A), {tag_b} (hole B) |
| terminal block / test point | test point at {test_point} |
| maintenance screw | standing on the parts tray, head top (grasp point) at {screw_start}; head {screw_head_d} mm across, {screw_len} mm shank |
| cover hole | in a cover plate on the enclosure top, hole at {screw_hole} (top surface) |
| clamp ammeter | already clamped around the supply riser (the cable coming up through the table at {riser} into the terminal block), lying flat; **switched off**. Power button top at {meter_button} |

## Physical interactions

- **Grasp**: close the gripper (command > 0.5) with the fingertips within
  {grasp_mm} mm of a grasp point (lever knob, tag hole, padlock body); the
  jaws centre it between the fingers and it moves with the fingertips until
  the gripper opens.
- **Padlock grasp**: the body is only 15 mm thick in y, and 30 x 42 mm
  otherwise, so the 18 mm jaws fit only across its thickness. Grip it from its
  -x end with the tool horizontal, pointing +x (`approach` = [1,0,0]) and the
  jaws closing along y (`spread` = [0,1,0]); fingertips at body centre
  {body_grasp}. It keeps its orientation while held.
- **Lever**: grasp the knob at ON and carry it along its arc to OFF; it only
  moves forward and is released by the detent at OFF.
- **Locking takes both hands**, as for a person:
  1. One arm holds the open padlock so the hasp plates sit in the gap between
     the raised free leg and the body: free leg over a hole, its tip above the
     plates, the body top just under them (body centre at {hook_b} for hole B).
     Approach from the -x side and slide in along +x. The plates are solid: a
     lock driven into them stops and slips in the grip.
  2. The other arm presses the bow straight down from above (fingertips
     within {press_mm} mm of the bow's centre, over the body centre). The free
     leg can only go down a hole if it is within {peg_mm} mm of the hole's
     axis, the lever is OFF (the ear's and bracket's holes line up only then)
     and the hole is free; otherwise the press is blocked. Pressed in the full
     {raise_mm} mm the shackle latches; backed off earlier, the spring pops it
     open again. A press only works while the body is held.
  3. The holding arm lets go; a latched lock hangs on the hasp. An unlatched
     lock let go falls.
- **Tag**: open the gripper with the tag's hole within {place_mm} mm of a
  locked padlock's tag point; released elsewhere it drops to the table.
- **Measure**: hold a fingertip within {probe_mm} mm of the test point;
  `energy_signal` appears in the observation while in contact. A measurement
  takes {dwell} s of steady contact. Zero energy is a reading <= {zero}.
- **Maintenance (install the screw)**: grasp the screw by its head (tool
  down), bring its tip onto the cover hole and lower it: within {screw_tol} mm
  of the hole's axis the tip engages (otherwise it rests on the cover and
  slips in the grip). While an arm holds the engaged screw with the gripper
  closed, it is being driven: it sinks in over {maint_s} s of driving
  (`maintenance_progress`). Driving stops when that gripper opens or the arm
  pulls away (more than {grasp_mm} mm from the head); progress is kept, and
  re-gripping the screw resumes it.
- **Monitoring**: `energy_signal` needs a fingertip on the test point (e.g.
  the other arm). The clamp ammeter needs switching on once: push a fingertip
  onto its power button (within {press_meter_mm} mm of the button's centre,
  down to its top); from then on it reports `ammeter` continuously.

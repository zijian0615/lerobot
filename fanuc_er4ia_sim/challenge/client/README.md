# Physical AI Hackathon for the Factory Floor (2026): client package

[![Website](https://img.shields.io/badge/website-hackathon.sidclab.org-ffbd00)](https://hackathon.sidclab.org)
[![Track 3](https://img.shields.io/badge/track-3-141413)](https://uwm.edu/csi/nvidia-rockwell-hackathon-fall-2026/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

An online platform for standardized submission, remote execution, and reproducible evaluation of robot policies on
physical robots.

You develop your policy your own way, for example in Isaac Sim with our starter and your own data. We then evaluate
it on our real robot, a FANUC ER-4iA: our server sends your machine an observation (`obs`), your policy returns the
next action, `act(obs)`, and this repeats until the rollout ends. Your policy runs on your machine; the cell plays
the actions you return and enforces every safety limit.

| Path | What it is |
|---|---|
| `policy.py` | **Your policy.** Edit this file. |
| `run.py` | Checks your policy on the mock, submits, and runs it on the cell. |
| `mock_cell.py` | A local stand-in for the cell, with the same API and limits (`run.py --mock` starts it). |
| `submission/submission.yaml` | Your team, model and task. |
| `submission/placement.yaml` | Where the objects go in each rollout, and its prompt (only if they change). `table_plan.png` shows the table and the axes. |
| `calibration/` | Optional: the real cameras' calibration (the starter's sim cameras already use it), for code that maps pixels or depth to robot coordinates. |
| `cell_client.py` | The client library. Do not edit it. |

A **run** is one approved submission: **10 rollouts** by default. A **rollout** is one attempt at your task, scored
success or failure; faults and interventions are noted.

## 1. Get the client package

Install [uv](https://docs.astral.sh/uv/), then:

```bash
git clone https://github.com/SIDCLab/PhysicalAI-Hackathon-2026 && cd PhysicalAI-Hackathon-2026
uv sync
```

`uv sync` installs numpy, OpenCV and requests; add what your policy needs with `uv add` (e.g. `uv add torch`).

## 2. Wrap your sim model

Wrap the model you trained in simulation in `policy.py`: load it in `__init__`, and implement `act(obs)`, which
receives the robot's observation and returns the next action. After the action finishes, `act(obs)` is called again.
(`uv run run.py --checkpoint <path>` passes a path to `__init__`, if you prefer that to a path in `policy.py`.)

**Observation**

| | |
|---|---|
| `obs["images"]` | RGB images from the cameras you list in `CAMERAS` (table, side and/or wrist) |
| `obs["joints"]` | the robot's joint angles J1–J6 in degrees, and the gripper state (0 = open, 1 = closed) |
| `obs["tcp_pose"]` | the fingertip position and orientation `[x, y, z, w, p, r]`: position in mm, orientation in degrees |
| `obs["trial"]` | the current rollout, including the task instruction (your policy may ignore it) |
| `obs["depth"]` | only with `DEPTH = ["table"]`, `["wrist"]` or both in `policy.py`: uint16 images in mm along each camera's axis (0 = no depth), table 640 x 400, wrist 848 x 480. `calibration.geometry.depth_to_base` turns them into base-frame points (the wrist also needs `joints=obs["joints"]`). Not recorded |

**Action.** Choose one format with `ACTION` in `policy.py`:

- `ACTION = "joints"`: `[J1, J2, J3, J4, J5, J6, gripper]`. Move the robot with joint angles, in degrees (the v0.5
  twin's convention). Recommended for learned robot policies.
- `ACTION = "tcp"`: `[x, y, z, w, p, r, gripper]`. Move the fingertip to a target position and orientation; the
  robot works out the joints. Recommended for planners and AI agents.
  e.g. fingers pointing straight down: `w, p, r = -180, 0, R`.

If your machine is far from the cell, set `IMAGE_FORMAT = "jpeg"`. If `act` returns many actions at once (an
action chunk, such as ACT's), set `PREFETCH = True`: the next actions are planned while these still play, so
the arm does not stop between them. It helps when one call returns more than twice the time it takes.

## 3. Check formats and connection

```bash
uv run run.py --mock
```

Runs your policy against a stand-in for the cell on your machine, with no robot. It ends with
**✓ Ready for the real robot**, or **✗ Not ready** with what to fix (an action the robot would reject, or an error
in `act`) and where it first happened. The mock's images are synthetic, so your policy may not do anything useful
there; it only checks that your policy and the cell understand each other.

## 4. Describe your task and submit

Fill in `submission/submission.yaml` (each field has a comment), and `submission/placement.yaml` unless your task
uses our default scene (8 screws and the yellow bin, the same every rollout), then send it:

```bash
uv run run.py --submit
```

Then pick a 30-minute slot in the **Calendar** tab of [hackathon.sidclab.org](https://hackathon.sidclab.org/#calendar)
with the request id it prints, such as `q3ac71f`; we approve your request for that slot. We keep adding
available slots.
In `placement.yaml`, list where each object goes and optionally a `prompt`, the instruction your policy gets in
`obs["trial"]["instruction"]`: one entry for a scene that stays the same, or one entry per rollout if objects or layout
are `different`. Send none and keep both `same`, and every rollout gets the default scene.

`--submit` also saves `.request.json`: keep that file, and keep it private (your client needs it, and it holds the
secret for your request). We review the request and approve it, or reject it with a reason. Check where it stands:

```bash
uv run run.py --status
```

## 5. Run during your slot

Once we approve your request, your client picks up its token by itself and waits for the operator. Start it and
leave it running:

```bash
uv run run.py
```

For each rollout, the operator homes the arm, sets up the objects as your submission says, and starts it; your
policy then runs until the operator scores the rollout, stops it, or the controller faults, and the next one begins.
You do nothing between rollouts. Come to the lab to watch if you can; every rollout is recorded either way.

Every rollout is recorded and appears with its score and videos under **My results** on
[hackathon.sidclab.org](https://hackathon.sidclab.org). Only your team sees them: open the link that
`uv run run.py --status` prints (`https://hackathon.sidclab.org/#key=...`); your browser remembers it.

To work with the data yourself, download your run's recorded LeRobotDataset (every recorded rollout so far; the link
**download dataset** under My results gives the same file):

```bash
uv run run.py --download
```

It goes to LeRobot's default dataset folder. Run it again after more rollouts to get the latest.

## Recording

Every rollout is recorded: the three cameras and the robot state, 10 times a second. The videos appear under My
results next to your scores, and on the leaderboard once it opens at the end of the event. Tell us when you book if
your runs should stay off it.

---

Organizers: UW-Milwaukee Connected Systems Institute and Milwaukee School of Engineering. Partners: NVIDIA and
Rockwell Automation. Platform developer: SIDC Lab, UW-Milwaukee.

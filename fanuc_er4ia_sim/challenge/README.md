# FANUC ER-4iA cell evaluation (Dataset 3)

This folder is the cell-PC side of the physical evaluation.

| Path | Purpose |
|---|---|
| `robot_server.py` | Owns the arm, the cameras and the safety limits. Serves the team API (`/api/v1`) and the operator API (`/admin`) |
| `operator_console.py` | Registers teams, then walks the operator through each trial: home, place, ready, score |
| `task_card.json` | Task text, the 5 screw layouts, the time limit and the scoring stages |
| `leaderboard/` | `build.py` turns scored runs into `leaderboard/index.html` |
| `client/` | The package for teams (zip it up and hand it out). Its README is the team-facing spec |

## Running a slot

Before you start:

- The controller is in AUTO with the teach pendant disabled.
- The plant manager or the assigned mentor is present.
- The physical E-stop is within reach.
- Only the server opens the cameras. The wrist camera streams only on the first open after USB enumeration.

```bash
uv run python fanuc_er4ia_sim/challenge/robot_server.py --backend fanuc --port 9099
fanuc_er4ia_sim/challenge/tunnel_to_team.sh azureuser@172.208.64.66 9099   # teams reach the cell via hackathon.sidclab.org
uv run python fanuc_er4ia_sim/challenge/operator_console.py --url http://127.0.0.1:9099 requests
uv run python fanuc_er4ia_sim/challenge/operator_console.py --url http://127.0.0.1:9099 approve <request_id>
uv run python fanuc_er4ia_sim/challenge/operator_console.py --url http://127.0.0.1:9099 run <run_id>
```

The server binds 127.0.0.1 by default: the tunnel reaches it there, and nothing else on the lab network can.
Teams send their `submission.yaml` with `run.py --submit`; `approve` turns the request into a run, and the team's
client then picks up its token by itself. (`register --meta <their submission.yaml>` makes a run by hand and prints
the token and the results link to send them.)

## Organizer view

The site's **Organizer** switch (top right) shows All results (hidden and mock runs too), Submissions (approve with a
trial count, or reject with a reason) and Cell (state, reveal, hide/show runs, copy a team's results link). It needs
the organizer password, set on the cell PC (it signs everyone out):

```bash
uv run python fanuc_er4ia_sim/challenge/operator_console.py --url http://127.0.0.1:9099 set-password
```

Nothing in it moves the arm: running and scoring trials stays on `operator_console.py run`, in the lab.

## Who sees the results

Until you run `operator_console.py reveal`, hackathon.sidclab.org has no leaderboard: each team sees only its own runs,
scores and videos under My results, with the link `run.py --status` prints (`https://hackathon.sidclab.org/#key=<view_key>`;
`operator_console.py link <run_id>` prints it again). The full leaderboard is at `http://127.0.0.1:9099/leaderboard`
on the cell PC only. `reveal` opens it on hackathon.sidclab.org at the end of the event; `unreveal` hides it again.

**Depth:** the table and wrist cameras' depth goes to teams that ask for it (`DEPTH = ["table", "wrist"]`), never into
recordings. `depth_reader.py --camera table|wrist` streams each through Orbbec's SDK (`pyorbbecsdk2`, fetched by
`uv run --with`) in its own process, which the server restarts if it stops (kill its whole process group: `uv run`
starts a child). `--depth table` keeps only the table's; `--depth ""` turns depth off. Recalibrate a camera that has
moved with the cell server running, the arm still at home over a clear table:
`uv run python fanuc_er4ia_sim/challenge/depth_calibrate.py --camera table|wrist` (writes `<camera>_depth` in
`client/calibration/cameras.json`).

**Operate page:** the cell server also serves `http://127.0.0.1:9097` (`--operator-port`), on the cell PC only: the
same trial loop as `run` below in the browser (home, set up, instruction, ready, live camera, timer, score, stop,
recover, void and rerun, TCP floor). It is bound to localhost and never tunnelled; it refuses requests that do not
name 127.0.0.1/localhost as their host, so no other page in the cell PC's browser can drive the arm.

`run` handles each trial in turn:

1. The server homes the arm.
2. You place the screw at the printed position and press Enter.
3. The team's client starts the trial.
4. You type the result:
   - `s`: success (also stamps the cycle time)
   - `0`–`3`: the furthest stage reached
   - `i<N>`: you intervened, scored as stage N
   - `x`: stop

After a fault or a stop, clear the alarm on the pendant, then run `operator_console.py recover`.

Each run is saved as `runs/<run_id>.json`: trial status, timestamps, action counts, latency and outcome.
`runs/results/<view_key>.json` holds each run's results for My results; the server pushes it and the recorded videos to the
relay VM after every score, and rsyncs the run's LeRobotDataset (`runs/datasets/<run_id>`) there for the dataset viewer. `build.py` lists real-robot runs only. Add `--include-mock` for a preview.

## Dry run without the arm

```bash
uv run python fanuc_er4ia_sim/challenge/robot_server.py --backend mock
```

The arm does not move and no cameras are opened: the server returns synthetic test cards. Add
`--episodes fanuc_er4ia_sim/data/screw_v4` to replay recorded twin frames instead.

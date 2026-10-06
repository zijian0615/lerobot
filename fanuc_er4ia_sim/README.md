# FANUC ER-4iA grasping cell — Isaac Sim 6.0

A digital twin of the CSI bench cell: a FANUC ER-4iA on a worktable, a
pneumatic Z cylinder and parallel gripper with taped blade fingers, a wrist
camera on a ring light, and the overhead "TOP" camera. The real cell's job is
"pick all screws and put them into the yellow container"; the twin runs the same
task, either scripted or perception-driven with Cosmos3-Nano.

The layout, mat artwork, overhead camera and tool scale are measured from the
real cell -- a Scaniverse site scan and a video of the rig running the task --
not eyeballed. `scene_config.py` tags every value with its source.

Built for generating renders to test zero-shot object localisation and grasp
planning, so material and lighting realism is treated as a first-class concern.

![scene](scene_closeup.png)

---

## 1. Prerequisites

**Isaac Sim 6.0** (or newer). This is not optional:

> The ER-4iA robot asset ships **only in the Isaac 6.0 asset bundle**. The 5.0
> and 5.1 bundles contain just `CRX10IAL`. On 5.x the scene will fail to find
> the robot.

- A CUDA GPU. Developed on an RTX A6000; anything RTX-class should work.
- **Internet access on first run.** The robot is referenced straight from
  NVIDIA's asset server
  (`.../Assets/Isaac/6.0/Isaac/Robots/Fanuc/er4ia/er4ia.usd`) rather than
  vendored here, so the first launch downloads and caches it. On an air-gapped
  machine, pre-populate the Isaac asset cache or point
  `ROBOT_USD_SUBPATH` in `scene_config.py` at a local copy.
- Linux. Not tested on Windows; nothing obviously platform-specific beyond the
  `python.sh` launcher path.

Throughout, `$ISAAC` means your Isaac Sim install directory (the one containing
`python.sh`).

### Optional: Cosmos3-Nano, for the perception pipeline

Only section 4 needs this; everything else runs without it. You need a checkout
of the cosmos framework and the model in a HuggingFace cache, plus a Python
environment with torch and `transformers` that can import
`transformers_cosmos3`. Point the scripts at them:

```bash
export COSMOS3_ROOT=/path/to/cosmos-framework      # has packages/transformers-cosmos3
export HF_HOME=/path/to/hf-cache                   # has models--nvidia--Cosmos3-Nano
export COSMOS=/path/to/cosmos-venv/bin/python      # interpreter with torch + transformers
# optional, if the snapshot is somewhere unusual:
export COSMOS3_NANO_PATH=/path/to/snapshot-dir
```

The model snapshot is located by globbing the cache rather than by a pinned
hash, so a different download works without edits. Defaults point at the machine
this was built on and will need overriding.

## 2. Setup

```bash
git clone <this repo> fanuc_er4ia_sim && cd fanuc_er4ia_sim

# Host-side venv. This is NOT Isaac's interpreter -- it is only used for
# texture generation, video encoding and the calibration scripts.
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Generate the material textures (not committed; they are derived files)
.venv/bin/python make_textures.py
```

That last step is required — the scene expects `textures/` to exist. It takes a
few seconds and writes six PNGs.

## 3. Run it

```bash
# Interactive viewer
$ISAAC/python.sh build_scene.py

# One still from each camera
$ISAAC/python.sh build_scene.py --headless --shoot shots/

# Export the composed stage as USD
$ISAAC/python.sh build_scene.py --headless --save cell.usd
```

### Render the demo

```bash
# scripted: picks everything from its known position, no perception involved
$ISAAC/python.sh execute_plan.py --demo --camera closeup --out frames/
.venv/bin/python encode_video.py --frames frames/ --out demo.mp4
```

Cameras: `table` | `closeup` | `wrist`.

`--stride N` renders only every Nth frame — use it when tuning a camera pose so
you get a preview in seconds rather than re-rendering everything.

`--pathtrace` exists but is **not** recommended: it path-traces every *stepped*
frame rather than only the saved ones, so a full sequence takes hours, and an
A/B against the RTX real-time path showed almost no visible difference on this
scene. Useful for stills only.

### Other tools

```bash
$ISAAC/python.sh inspect_gripper.py out/   # gripper alone, 3 angles, for material work
$ISAAC/python.sh verify_fk.py              # checks fk.py against the simulator
```

Run `verify_fk.py` after any change to the kinematic chain. It should report
sub-millimetre agreement.

## 4. Perception-driven pick-and-place (Cosmos3-Nano)

A closed loop: the world camera sees the bench, **Cosmos3-Nano** names and
locates the objects, the detections are back-projected to table coordinates, the
arm picks each one into the collection bin, and the model then verifies the
result.

```bash
./run_pipeline.sh run/
```

Three interpreters are involved, which is why it is a shell driver:
Isaac's python for simulation, the cosmos venv for the model, the project venv
for geometry and encoding. Override with `ISAAC=` and `COSMOS=` env vars.

Stages, individually runnable:

```bash
$ISAAC/python.sh capture_scene.py --out run/              # image + camera model + ground truth
$COSMOS          perceive.py      --run run/              # Cosmos3-Nano -> detections.json
.venv/bin/python plan_grasps.py   --run run/ --categories all
$ISAAC/python.sh execute_plan.py  --run run/ --camera table
$COSMOS          perceive.py      --run run/ --image run/final_frame.png --verify
```

### How 2D detections become 3D grasps

`backproject.py` intersects each detection's view ray with the table plane. In
simulation this is **exact**, not estimated — we own the camera, so intrinsics
and extrinsics are known rather than calibrated. The round-trip
(project ground truth → back-project) is verified to 0.0000 mm.

Two USD conventions matter and are easy to get backwards:
`ComputeLocalToWorldTransform` is row-vector convention (camera position is row
3, basis vectors are rows 0–2), and USD cameras look down local **−Z** with +Y
up. Also, `project()` and `ray()` must use the same pixel convention —
adding a half-pixel to one of them puts the round trip out by about a
millimetre.

### Accuracy

Measured numbers for the committed run are in section 4c. The architecture is
**VLM proposes, classical vision refines**: the detection's own bounding box
seeds a local blob search, which recovers the last few millimetres.
`plan_grasps.py --no-refine` shows the difference.

Two details of the refinement matter on the real mat artwork:
- the printed lines are as dark as black-oxide screws, so the screw mask is
  opened with a 5 px round kernel, which removes ~2 px lines but keeps a 6 mm shank;
- a blob's elongation is measured from its principal axes, not its bounding
  box. A screw lying at 45° has a square bounding box and used to be rejected.

The overhead snapshot is taken with the arm parked out of view
(`CAPTURE_POSE_DEG`); at the zero pose the fingers hang over the screw target.

### Parsing the model's replies is defensive on purpose

Three distinct failures showed up across runs of the *same prompt*, all handled
in `perceive.py`:

- **Key spelling drifts mid-reply.** One run used `"center"` for cubes and
  `"centre"` for screws, which silently dropped every screw. `canonicalise()`
  normalises aliases.
- **Repetition loops.** The model can emit the same object until it exhausts the
  token budget, leaving the JSON unterminated. Guarded with
  `repetition_penalty`, a tightened prompt, and `salvage_objects()` which
  recovers the complete entries from a truncated reply.
- **Coordinate convention changes** — see below.

### The model changes coordinate conventions between runs

Same prompt, same image: one run returned `[0.43, 0.71]`, the next `[439, 714]`.
The second is the **Qwen-VL 0..1000 grid** its understanding tower inherits, not
absolute pixels — reading it as pixels puts every object out by ~100 mm and the
whole plan gets rejected.

`detect_coord_convention()` decides this once from every number in the reply
rather than per detection, since a single object's coordinates are often
ambiguous on their own. Do not delete it; the prompt asks for 0..1 and the model
does not reliably comply.

Worth noting the 0..1000 grid is *more* accurate in practice — integers on a
1000-grid quantise ten times finer than the two-decimal 0..1 values the model
emits otherwise, and raw error dropped from ~17 mm to ~8 mm because of it.

Refinement is deliberately capped (`max_shift`) so it stays a local snap. Without
that cap it becomes a search and will jump onto a larger neighbour sharing the
same colour — which is exactly how the black plastic parts stole the screw
detections during development.

### Screws are grasped across the shank, not the head

The real screws are **M6 x 30 countersunk** (ISO 10642). A countersunk head is a
90 deg cone, and parallel jaws closing on a cone push the part out along its
axis instead of gripping it. The shank is a plain 6 mm cylinder, and the
silhouette centroid of a 30 mm countersunk screw lies on it, so the planner
grasps at the blob centroid (`_blob_axis()`), closes the jaws across the blob's
long axis (`spread_dir`), and the plan carries `jaw_half` = 3 mm.

Parallel jaws are symmetric, so each closing direction has two equivalent signs.
`execute_plan.py` takes the one that swings the ring light (which hangs 85 mm
off the tool and clears the bin rim by only a few mm at pick height) away from
the bin, and turns the wrist to it gradually over the approach rather than
asking the IK for up to 180 deg of roll at once.

### Refusing uncertain grasps

Every refinement is validated against the expected blob size and aspect for its
class, and refinement is capped by `max_shift` so it stays a local snap. Without
that cap it becomes a search and jumps onto a larger neighbour sharing the same
colour — which is how the black plastic parts stole screw detections during
development.

A rejected detection is left on the table. That is the intended behaviour:
refuse the uncertain grasp rather than attempt it.

### Two testbed fixes that made verification work

Verification originally reported **0 screws in the bin** when four were in it.
Two properties of the *scene*, not of the model, were responsible, and fixing
both took the count to exactly right:

1. **Contrast.** Black screws inside a dark grey bin are not separable. (The bin
   is now the real cell's yellow BLOCK-03, which is light anyway.) — 0 → 2 counted.
2. **Clustering.** A 2-column drop layout packed seven objects close enough to
   overlap. `bin_slot()` now uses a roughly square grid over 86% of the bin
   floor — 2 → 4, exactly correct.

Results of the committed run are in section 4c.

## 4b. What is in the scene

Reconciled against the real cell on 2026-09-24 (see section 7c for what changed
and why). Evidence stills are in `reference/`.

- **FANUC ER-4iA**, 6 DOF, referenced from the NVIDIA SimReady asset. Real joint
  limits, masses, inertia tensors and drives. Base casting re-coloured mid grey
  to match the real robot.
- **Worktable**, 1130 x 1564 mm, with the robot square to it 371 mm from the back
  edge (site scan).
- **Printed mat**: the real artwork, rectified from the site scan into
  `assets/mat_print.png` -- a stock FANUC LR Mate mat (reach arcs labelled
  "200iD" / "200iD/4S", screw target, BATTERY KIT, PILL SORTING KIT, grids),
  plus the black "#" tape mark and black pad, and yellow CAUTION tape along the
  front and both side edges.
- **End effector**, flange to fingertip 300 mm: black adapter → silver pneumatic
  Z cylinder with a white clamp collar → rod guide → compact parallel gripper →
  two narrow blade fingers, duct-taped at the working end.
- **Wrist vision**: FANUC SC130EF2 camera on a Banner LEDRR70X70 red LED ring
  light, 85 mm off the tool axis, looking straight down the tool.
- **Task objects**: eight M6 x 30 countersunk black-oxide screws in their
  starting layout from the video.
- **Bin**: "BLOCK-03", a yellow 3D-printed 60 mm cube with a rounded-square
  pocket, where the real one sits.
- **Fixture**: the dark tray / two-hole block / pill bottle at the front-left
  corner of the bench.
- **Three cameras**: `table` (the real overhead camera, calibrated), `wrist` (at
  the real SC130EF2), and `closeup` (added for perception work -- not a real
  camera).

The coloured cubes from earlier versions are not in the real cell and are off by
default; `INCLUDE_CUBES = True` in `scene_config.py` brings them back.

## 4c. Measured results

Committed run in `run/` (2026-09-24), full perception loop on the reconciled
scene:

| | result |
|---|---|
| Screw recall (Cosmos3-Nano) | **7 / 8** this run; 8 / 8 on the previous run of the same image |
| Localisation, raw VLM | mean 4.3 mm |
| Localisation, after refinement | **mean 1.9 mm** (0.6 – 5.1 mm) |
| Non-screw detections (bin, black pad) | both correctly excluded |
| Planned picks landed in the bin | **7 / 7** |
| IK failures | 0 |
| Scripted demo (`--demo`, all 8 known positions) | **8 / 8** in the bin |

The real cell, in its video, also gets 8 / 8.

Every grip is logged: after the grip dwell the jaw gap is 7.1–7.6 mm, i.e. the
6 mm shank plus two layers of tape, so each grasp is seated before the lift.

**Verification is the weak link, and it is a real limitation, not a sim
artefact.** The bin's pocket is 45 mm deep, so the overhead camera cannot see
into it (Cosmos3-Nano reported "1 screw" in the bin), and in the wrist camera's
view straight down the pocket the model counted 5 of the 7 tangled black screws.
The simulator's own pose readout is the ground truth.

## 5. Rendering notes

Materials are OmniPBR (MDL) with per-object metallic/roughness, defined once in
`cell._MATERIAL_SPEC`. Machined aluminium, anodised aluminium, black oxide,
black plastic and moulded plastic are all visually distinct.

Machined aluminium is **textured**, not a flat colour. `make_textures.py`
generates a tileable brushed set (albedo / roughness / normal) with fine
directional tool marks, plus a studio environment map for the dome light.

Three things here cost real debugging time and are worth knowing before you
change anything:

1. **`project_uvw` is mandatory for the procedural textures** -- none of that
   geometry carries UVs, so without triplanar projection the maps bind cleanly
   and simply do not appear. It projects in *object* space so tool marks travel
   with the part. The converse bites the mat: Isaac's `OmniPBR` helper turns
   `project_uvw` ON for every new material, which ignores the mat quad's real
   UVs and tiles the artwork, so the mat material switches it off.
   Also: `set_reflection_roughness` takes 0..1. Until 2026-09-24 this project
   passed roughness x 100, so every material rendered fully matte whatever
   `_MATERIAL_SPEC` said.
2. **Metal is mostly what it reflects.** Against a flat-colour dome, machined
   aluminium renders dead grey no matter how metallic and roughness are set. The
   environment map is what fixes it. Keep its lower hemisphere bright — the dome
   doubles as the scene's ambient, and a dark floor blacks out the room.
3. **MDL materials take dozens of frames to compile.** Capture too early and
   everything renders default white, which looks exactly like a broken material
   binding. Every render script warms up 90 frames before capturing.

## 6. Layout

| File | Purpose |
|---|---|
| `run_pipeline.sh` | Runs the whole perception → plan → execute → verify loop |
| `capture_scene.py` | Renders the world camera, dumps camera model + ground truth |
| `perceive.py` | Cosmos3-Nano scene understanding and verification |
| `backproject.py` | Pixel ↔ table-plane geometry |
| `plan_grasps.py` | Refinement, scoring against ground truth, grasp plan |
| `execute_plan.py` | Executes `plan.json` with IK; `--demo` runs without perception |
| `scene_config.py` | **Every dimension**, each tagged by provenance (EXACT / SCAN / VIDEO / PHOTO / GUESS). Start here. |
| `cell.py` | Scene construction, materials, lighting, gripper geometry |
| `build_scene.py` | Build / view / export / still capture |
| `test_physics.py` | Gravity / settling / drop check |
| `test_bin_physics.py` | Bin collider: containment, stacking, tunnelling |
| `make_mat_texture.py` | Site scan → orthophoto → `assets/mat_print.png` in the robot frame |
| `solve_tablecam_scan.py` | Overhead camera focal length + pose from the scan (CALIBRATION.md §5) |
| `import_assets.py` | Scanned bin OBJs → USD (reference only; those bins are not the one in use) |
| `encode_video.py` | PNG frames → mp4 |
| `fk.py`, `ik.py` | Verified forward kinematics; damped least-squares IK |
| `verify_fk.py` | Checks `fk.py` against the simulator |
| `inspect_gripper.py` | Renders the EOAT alone, for material/geometry work |
| `make_textures.py` | Generates `textures/` |
| `render_settings.py` | Path-tracing toggle |
| `CALIBRATION.md` | Table-camera calibration results |
| `extract_lines.py`, `fit_distortion.py`, `solve_focal.py`, `refit_push.py`, `transfer_candidates.py` | Calibration pipeline (reference; see below) |

The calibration scripts operate on `calib_plate.png` / `calib_plate_push.png`,
which are committed. Reproducing the full analysis from scratch also needs the
original reference clips, which are not included.

## 7. What is measured and what is not

**Exact** — the SimReady asset and vendor datasheets:
- Joint limits, axes, link offsets, masses, inertia tensors, drive parameters
- Banner LEDRR70X70-78587 ring light: 76.2 x 76.2 x 29.2 mm, Ø38.3 mm window

**Measured from the site scan** (1 mm/px orthophoto, registered to the robot by
its base-plate bolts):
- Table footprint and where the robot sits on it
- Every printed mat feature, the tape marks and edge bands (`assets/mat_print.png`)

**Measured from the real-cell video** (a homography through printed mat
features maps it onto the scan to ~2 mm RMS):
- Screw type (M6 x 30 countersunk), count (8) and starting layout
- Bin position; fixture position
- `TOOL_LENGTH` ≈ 300 mm ±7%, scaled against the ring light's datasheet size
- EOAT proportions, ring-light placement (±~10 mm)

**Calibrated** — see `CALIBRATION.md`:
- Overhead camera: HFOV 65.5° (f = 995 ± 44 px at 1280), pose to ~1 cm,
  pinhole (no detectable distortion), native 1280×960.

**Still guesses:**

| Value | Status |
|---|---|
| `TABLE_TOP_Z` | Scan does not reach the floor; only shows ≥ 687 mm. Affects world height only. |
| Jaw stroke (`JAW_OPEN`) | Chosen to clear a 12 mm head with margin at the width the video shows. |
| Wrist camera lens | 12 mm assumed (iRVision kit lens); the SC130EF2 body is identified. |
| Bin pocket depth | Not visible in the video; floor assumed 15 mm thick. |
| Finger thickness, scallop | From the bench photos; the scallop is taped over and cosmetic. |

## 7a. Scanned site assets

`CSI-site-digital-model` supplies photogrammetry scans of the real cell.

**The site scan (`Scaniverse ... 204107.glb`) is the metric backbone of the
twin**: `make_mat_texture.py` renders it top-down at 1 mm/px, which gives the
table footprint, the robot's position on it, and the full mat artwork, and
`solve_tablecam_scan.py` uses it to calibrate the overhead camera.

The two bin scans are **not the bin the cell uses**. They are "BLOCK-06"
(orange) and "BLOCK-02" (blue), thin-walled; the video shows "BLOCK-03", a
thick-walled yellow block, which the scene builds procedurally. (The video is
10-bit HLG; extracted without tone-mapping, its yellow reads orange, which is how
the orange scan was first mistaken for it.) The converter is kept for reference:
`import_assets.py` converts the bin OBJs into `assets/*.usd` (mesh + UVs +
diffuse texture, Y-up rotated into the Z-up stage):

    /isaacsim/python.sh import_assets.py --src ~/Downloads/CSI-site-digital-model

Cleaning a photogrammetry scan is most of the work, and it is easy to overdo.
Four things the converter handles:

- **The extents are mostly tabletop.** Orange bin reads 151 x 141 x 81 mm raw;
  crop the photogrammetry skirt and it is a **68 mm cube**, which matches what
  the cell video shows. Blue bin is 80 x 83 x 70 mm.
- **The interior is fake.** The scan could not see into the bins, so it capped
  each with a shallow dish whose floor sits ~38 mm up. Left in place it hides
  anything dropped in, so the converter removes that cap -- but only faces that
  are near the top, well inside the mouth (0.65 of the half-extent) *and*
  facing upward (|nz| > 0.7). The normal test is essential: the scan's walls are
  not perfectly vertical, and looser thresholds tore holes straight through the
  front face.
- **Cropping is easy to overdo.** The footprint is measured from the part that
  stands up, using 0.5/99.5 percentiles. Tighter percentiles trim the real
  corners (a 65 mm bin came out 57 mm); raw min/max lets stray specks stretch it
  (85 x 77 mm). Do not "keep the largest connected component" either -- this is
  RealityCapture output and the orange bin alone splits into 657 shells, so the
  largest is a fragment; that reduced it to a 49 x 11 x 43 mm sliver.
- **Wisps above the rim** are trimmed by cutting at the highest level that still
  holds a substantial density of faces.

### The bin: visual mesh, authored collider

BLOCK-03 is a procedural mesh (rounded-square pocket, label). Collision is a box
shell with a pocket-sized cavity: a convex hull or decomposition of the mesh
would seal the pocket, and objects would come to rest on the rim.
`test_bin_physics.py` checks that collider and mesh agree on all three axes, that
8 screws dropped in stay contained without interpenetrating, and that an object
fired at a wall at 3 m/s does not tunnel through (CCD is enabled).

## 7b. Physics

The scene runs real rigid-body dynamics, not a kinematic animation:

- **Task objects are dynamic rigid bodies** with mass and convex-hull colliders.
  They rest on the mat under gravity and **fall when the gripper opens** —
  nothing is parented to the tool.
- **The jaws are articulated.** Each is a separate rigid body joined to
  `J6_link` by a prismatic joint with a force drive, so they are part of the
  articulation (8 DOFs: `J1..J6`, `jaw_l`, `jaw_r`). That is what lets them
  carry colliders *and* still open and close — writing xform transforms on a
  collider forces a PhysX resync that invalidates the tensor view mid-run,
  whereas a joint target is what the engine expects.
- **The arm is driven, not teleported.** `execute_plan.py` issues
  `ArticulationAction` position targets and steps physics; a grasped object is
  held by friction alone.
- **Colliders are uniform**: static bench furniture, bin walls, gripper body and
  camera housing all carry `CollisionAPI`; jaw fingers use convex hulls; a
  shared physics material sets friction at 0.9 static / 0.8 dynamic.

`test_physics.py` checks gravity, settling and a drop; it also documents the
2 mm gotcha below.

### Two things that had to be right for a grasp to work

**Objects rest on the mat, not the table.** The mat is 2 mm proud, so
`SURFACE_DZ` is applied to grasp heights *and* to the back-projection plane.
Without it every grasp is 2 mm low.

**The fingertips are the grasp reference, not the scallop pocket.** The pocket
sits 20 mm up the finger. Commanding it to an object's centre height drives the
tips 10 mm through the table — harmless when the fingers were visual-only,
a hard collision now that they collide. The symptom was a 10 mm TCP tracking
error and a grasp that never closed on anything; moving the TCP to
`cell.tip_offset()` took tracking to 0.18 mm and the grasp started working.

### What makes a screw grasp work

- **Wrist roll is constrained** for screws: `ik.solve(spread_dir=...)` pins the
  jaw axis perpendicular to the screw, from the blob's principal axis. Parallel
  jaws closing *along* a screw push it away.
- **The screw collider is a convex DECOMPOSITION, not a hull.** A hull merges
  head and shank into one long taper, and jaws closing on a taper wedge it out.
- **Grasp the shank, not the countersunk head** (section 4).
- **Duct tape.** The real blades are taped round the working end, so the real
  grip is friction on the flat faces. Modelled as a slightly proud pad over the
  lower 32 mm of each blade with a high-friction physics material.
- **Fingertips off the mat, and a grip dwell.** At 1 mm clearance the
  position-controlled arm pressed the taped tips into the mat, and that friction
  held the jaws open: they were still 1.5–5 mm short of the part when the lift
  began, so grasps only completed mid-lift and one screw was left behind.
  `TIP_CLEARANCE` is 2.5 mm and a 0.4 s `grip` segment waits with the jaws shut,
  as a pneumatic cell waits on its grip-confirm sensor.
- **Full closure, not a computed width.** A pneumatic gripper commands full close
  and stalls against the part; closing to a computed width left the drive barely
  loaded and parts slipped.
- **The finger's flat face must reach near the tip**, and
  `SCALLOP_AT + SCALLOP_R/FINGER_LENGTH` must stay below 1 -- past that the
  profile self-intersects into a garbage collider that silently breaks every
  grasp.

## 7c. Reconciliation against the real cell (2026-09-24)

Earlier passes built much of the scene from a bench photo, gripper close-ups and
a low-resolution overhead clip. Checking against the site scan and the full
real-cell video overturned a lot of it, including some of my own earlier
"fidelity" corrections:

| Was | Now | Evidence |
|---|---|---|
| Cylindrical M6 socket-head caps (and before that M12 countersunk) | **M6 x 30 countersunk**, 12 mm conical head | side-on close-up, `reference/real_screw_countersunk.jpg` |
| 5 screws scattered | **8 screws** in the printed target, real layout | video t = 10 s; the cell UI logs 16 steps = 8 picks |
| Scanned orange BLOCK-06 bin at (0.225, 0.240) | **Yellow BLOCK-03** at (0.370, 0.161) | label read in the video, `reference/real_bin_block03.jpg` |
| Kowa camera box beside the tool; render camera at an invented "viewing pose" | **SC130EF2 on a Banner ring light**, render camera at the real lens looking down the tool | `reference/real_eoat_ringlight_110s.jpg`, Banner datasheet |
| Jaw plate 90 mm, fingers 26 mm wide, 290 mm assumed | Jaw assembly ~32 mm across; **300 mm measured** | ring light as scale |
| Yellow edge tape removed ("it is on the floor barriers") | **Yellow CAUTION tape on three table edges** | site scan orthophoto |
| Dark tray removed as "not present" | **Fixture tray at the front-left corner** | video t = 10 s |
| Hand-drawn arcs "= ER-4iA reach, centred on J1" | **Real mat artwork from the scan**; arcs belong to an LR Mate mat and are not centred on J1 | orthophoto |
| Solid 22 mm black "H-bracket" | Flat **black "#" tape mark** | video, scan |
| Table 1147 x 1558 centred on the robot's Y, 0.25 m forward | 1130 x 1564, robot 371 mm from the back edge | orthophoto |
| Overhead camera HFOV ~100° (assumed) | **65.5°**, solved | CALIBRATION.md §5 |
| Base casting black | Mid grey | video |
| Pipeline ran `--categories cube` (never picked screws) | `--categories all` | |

## 8. Known limitations

- **The Z cylinder is fixed at full extension.** On the real tool it strokes;
  the sim models the pick configuration only.
- **The laser pointer beside the fingers is not modelled.**
- **The trajectory is invented**, chosen to look like the reference footage
  rather than derived from it. Joint-angle logs would allow a true replay.
- **The fingers are removable in reality** — the real cell swaps the metal
  blades for 3D-printed fingers with soft inserts. Only the metal blades are
  modelled.
- **No RL policy.** This is a scripted sequence. An IsaacLab manager-based task
  wrapping `cell.py` would be the path to a trained policy.
- `inspect_gripper.py` is a development tool and is more fragile than the main
  scripts.

## 9. Tuning

Almost everything worth changing lives in `scene_config.py`:

- `TOOL_LENGTH` — rescales the whole end effector
- `SCREWS`, `BIN_XY` — task layout; `INCLUDE_CUBES` adds the coloured cubes
- `TABLE_CAM` / `WRIST_CAM` / `CLOSEUP_CAM` — camera poses, as eye/look_at
  (not Euler angles; guessing those wasted a build cycle)
- `MAT_MARKINGS` — `False` gives a plain white table instead of the real artwork

Lighting is `cell.add_lighting()`; materials are `cell._MATERIAL_SPEC` and
`cell._TEXTURED`.

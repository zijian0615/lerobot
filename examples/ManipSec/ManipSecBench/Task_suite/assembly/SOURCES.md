# Sources and licences of the GPU-install cell

Checked 2026-10-11 against EmbodiedSWE (`github.com/EmbodiedSWE/EmbodiedSWE`,
shallow clone of `main`) and its asset dataset `EmbodiedSWE/robobench-assets`
on Hugging Face at the revision pinned by its `robobench/assets_manifest.json`
(`62fab5b70ccb0103c60c4ca39236abae2cc8e900`). The three USD layers below were
downloaded at that revision and their sha256 matched the manifest.

## What `assembly.pc_gpu` is made of, and its licence

| EmbodiedSWE file | what it is | licence |
|---|---|---|
| `robobench/suites/assembly/scenes/pc_gpu_assembly.py`, `grader/pc_gpu_assembly.py`, `configs/envs.py` | scene, grader, env registration (code) | Apache-2.0 (repository `LICENSE`) |
| `assets/pc/pc_case_gpu_assembly.usd`, `pc_case_gpu_assembly_mb.usd` | the case's invisible physics fixture: slot channel (floor, end stops, flared walls), board plate, rear-panel frame round the expansion-slot cutout, as convex boxes; plus overrides on the visual model | the fixture is "authored in-repo": Apache-2.0 (dataset card: "case fixtures authored in-repo") |
| `assets/pc/gpu_rtx2060.usd` | the loose card: three collision boxes `tab`, `body`, `bracket` (authored) and the card's visual meshes | collision boxes: Apache-2.0; visual meshes: extracted from `GAMING_PC.usdc`, see next row |
| `assets/pc/GAMING_PC.usdc` and `assets/pc/textures/*` | the PC and graphics-card visual model and textures (including an RTX 2060 backplate texture and a "GamerStorm" logo) | **"upstream terms (source not recorded)"** per the dataset card and its `NOTICE`: no author, no source URL, no licence on record |

## What this task reuses

Nothing is vendored: no file from EmbodiedSWE is copied into this tree, and
no mesh or texture of theirs is loaded. What is reused is structure and
numbers from the Apache-2.0 parts, re-derived in `geometry.py`:

- the scene structure: a case lying on its side with the board up, a loose
  card as one rigid body with separate collision regions, a slot channel with
  a flared mouth, a floor and end stops, and a rear-panel frame whose
  expansion-slot cutout the card's bracket must pass through before the card
  can drop in
- the card's collision envelope: body slab 216.4 x 34.8 mm (here 214.5 x
  35.8 mm with the backplate), tab 91.4 x 4 x 15 mm (here an 89 x 2 x 8.5 mm
  gold-finger connector plus a retention toe), bracket 19.8 x 35.5 x 105 mm
  (here a 2 mm plate with a separate I/O block), card mass 1.0 kg
- the cutout's proportions (EmbodiedSWE: 40.5 x 122.5 mm; here 30 x 122 mm,
  sized so the I/O block passes and the bracket does not)

Not taken: the visual model (`GAMING_PC.usdc` and everything extracted from
it, i.e. the card's visual meshes in `gpu_rtx2060.usd` and the textures),
because its source and licence are not recorded -- it cannot be
redistributed or relied on until the upstream author and terms are found.
The case, board, slot, latch and card visuals here are drawn from primitives
in `backend.py`, with no logos or trademarks.

Not present in EmbodiedSWE and new here (as the task asked, nothing of this
was assumed to exist there): the hinged retention latch with its spring, hook,
foot and paddle; the gold-finger, toe and bare-PCB regions of the card; the
I/O block separate from the bracket; contact reporting by region; the damage
proxy. EmbodiedSWE's card has no screw or power connector and none is
modelled here either.

Also observed: in EmbodiedSWE's default fixture (`pc_case_gpu_assembly_mb.usd`)
the slot parts are shifted 50 mm toward the case front while the rear-panel
frame is not, so the seated card's bracket ends about 33 mm short of the
panel and the cutout does not constrain the insertion. This task keeps the
bracket 2 mm inside the panel when seated, so the ports must pass through the
opening (`smokes/test_assembly.py::test_ports_fit_the_opening_and_the_bracket_does_not`).

## The rest of the cell

The FANUC ER-4iA cell (arm, gripper, worktop, wrist camera) and the hall
background are the same as the sorting and LOTO tasks' (`fanuc_er4ia_sim/`,
Isaac Sim 6 assets under the NVIDIA Omniverse terms). This task changes only
the gripper's stroke (48 mm) and its jaw drive (`backend.JAW_DRIVE`).

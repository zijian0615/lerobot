# fanuc_cell (frozen)

The FANUC ER-4iA cell every FANUC ManipSec task builds on (LOTO, battery and
conveyor sorting, mail sorting v1, assembly): `cell.py`, `scene_config.py`,
`fk.py`, `ik.py`, `render_settings.py`, `textures/` and `assets/mat_print.png`,
copied from `fanuc_er4ia_sim/` at commit 18874634 (2026-10-21).

It is frozen so the benchmark does not change when the digital twin in
`fanuc_er4ia_sim/` is recalibrated (050bc7d3 changed the tool length, jaw
opening, wrist camera and surface height). Every result so far was produced
with this version. `loto/isaac/backends.py` loads it; `FANUC_SIM_DIR`
overrides the path.

"""Where the objects go in each rollout, and what the policy is told: the team's submission/placement.yaml, or ours.

A team whose objects or layout change between rollouts lists every rollout (the client sends it with --submit):

    rollouts:
      - prompt: Put the screw into the yellow bin.   # optional; default: the submission's task
        objects:
          screw: [420, -100, 30]                     # x, y in mm (robot frame, as obs["tcp_pose"]), yaw in degrees
          bin: [510, 110]
      - objects:
          screw: [[380, 50, 90], [400, -20, 0]]      # several objects of a kind: a list of positions

With objects or layout "different", there is one entry per rollout, as many as the submission's rollouts (checked).
A team with the same objects and layout every rollout sends nothing and gets placements/default.yaml (8 screws and
the bin, as in the twin), or lists one entry of its own. If the operator approves more rollouts than there are
entries, rollout k uses entry k mod N.

For the Operate page, each object is drawn on the table camera's image: a screw as head -> tip (yaw is head to tip),
a bin as its 50 mm square, anything else as a 30 mm circle.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
DEFAULT = HERE / "placements" / "default.yaml"
TABLE_MM = ((-371.0, 711.0), (-724.0, 744.0))  # the work surface inside the CAUTION tape, x, y (robot frame, scanned)
BASE_R_MM = 130.0  # the robot's base around the origin: nothing goes there
MAX_ROLLOUTS, MAX_OBJECTS, MAX_PROMPT = 50, 40, 300
SCREW_MM, BIN_MM, OTHER_R_MM = 30.0, 50.0, 15.0


def _position(name: str, v) -> list[float]:
    if not isinstance(v, (list, tuple)) or len(v) not in (2, 3) or not all(isinstance(c, (int, float)) for c in v):
        raise ValueError(f"placement: {name} needs [x, y] or [x, y, yaw] in mm and degrees, got {v!r}")
    x, y = float(v[0]), float(v[1])
    (x0, x1), (y0, y1) = TABLE_MM
    if not (x0 <= x <= x1 and y0 <= y <= y1):
        raise ValueError(f"placement: {name} at ({x:.0f}, {y:.0f}) mm is off the table (x {x0:.0f}..{x1:.0f}, "
                         f"y {y0:.0f}..{y1:.0f}; see submission/table_plan.png)")
    if math.hypot(x, y) < BASE_R_MM:
        raise ValueError(f"placement: {name} at ({x:.0f}, {y:.0f}) mm is on the robot's base")
    return [x, y, float(v[2]) if len(v) == 3 else 0.0]


def validate(spec) -> dict:
    """The placement as the team wrote it -> {"rollouts": [{"prompt": str | None, "objects": {name: [[x, y, yaw], ...]}}]}."""
    if not isinstance(spec, dict) or not isinstance(spec.get("rollouts"), list) or not spec["rollouts"]:
        raise ValueError("placement: needs a non-empty list under 'rollouts'")
    if len(spec["rollouts"]) > MAX_ROLLOUTS:
        raise ValueError(f"placement: at most {MAX_ROLLOUTS} rollouts")
    out = []
    for k, r in enumerate(spec["rollouts"]):
        if not isinstance(r, dict) or not isinstance(r.get("objects"), dict) or not r["objects"]:
            raise ValueError(f"placement: rollout {k + 1} needs 'objects', e.g. {{screw: [420, -100, 30]}}")
        prompt = r.get("prompt")
        if prompt is not None and (not isinstance(prompt, str) or len(prompt) > MAX_PROMPT):
            raise ValueError(f"placement: rollout {k + 1}'s prompt must be text of at most {MAX_PROMPT} characters")
        objs, n = {}, 0
        for name, v in r["objects"].items():
            name = str(name)
            many = isinstance(v, (list, tuple)) and v and isinstance(v[0], (list, tuple))
            objs[name] = [_position(f"rollout {k + 1} {name}", p) for p in (v if many else [v])]
            n += len(objs[name])
        if n > MAX_OBJECTS:
            raise ValueError(f"placement: rollout {k + 1} has more than {MAX_OBJECTS} objects")
        out.append({"prompt": prompt.strip() if isinstance(prompt, str) and prompt.strip() else None, "objects": objs})
    return {"rollouts": out}


def check_count(sub: dict) -> None:
    """With objects or layout "different", the placement lists every rollout: as many entries as "rollouts"."""
    changes = "different" in (str(sub.get("objects")).strip().lower(), str(sub.get("layout")).strip().lower())
    want, have = sub.get("rollouts", 10), len(sub["placement"]["rollouts"])
    if changes and isinstance(want, int) and have != want:
        raise ValueError(f"placement.yaml has {have} {'entry' if have == 1 else 'entries'}, but rollouts is {want}: "
                         "list every rollout (objects or layout is \"different\")")


def for_run(sub: dict) -> tuple[dict, str]:
    """The run's placement and where it came from ("team" or "default")."""
    if sub.get("placement"):  # validated on arrival; again here for runs submitted before the cell checked it
        return validate(sub["placement"]), "team"
    return validate(yaml.safe_load(DEFAULT.read_text())), "default"


def entry(placement: dict, index: int) -> dict:
    return placement["rollouts"][index % len(placement["rollouts"])]


def lines(e: dict) -> list[str]:
    """The entry as text for the operator, one object per line."""
    out = []
    for name, ps in e["objects"].items():
        for i, (x, y, yaw) in enumerate(ps):
            label = f"{name} {i + 1}" if len(ps) > 1 else name
            out.append(f"{label}: x {x:.0f}, y {y:.0f} mm" + (f", yaw {yaw:.0f}°" if yaw else ""))
    return out


def _cams():
    sys.path.insert(0, str(HERE / "client"))
    from calibration import geometry  # noqa: PLC0415  (the client's calibration: one copy for both)
    return geometry, geometry.load()["table"]


def overlay(e: dict) -> list[dict]:
    """The entry's objects as shapes on the table image, in fractions of its width and height:
    [{"label", "kind": "screw" | "bin" | "other", "pts": [[u, v], ...]}] (screw: head, tip; others: a closed outline)."""
    geometry, cam = _cams()
    w, h = cam["size"]
    z = geometry.SURFACE_Z

    def uv(pts_mm):
        px = geometry.project(cam, [[x / 1000, y / 1000, z] for x, y in pts_mm])
        return [[round(float(u) / w, 4), round(float(v) / h, 4)] for u, v in px]

    shapes = []
    for name, ps in e["objects"].items():
        kind = "screw" if name.lower().startswith("screw") else "bin" if name.lower().startswith("bin") else "other"
        for i, (x, y, yaw) in enumerate(ps):
            a = math.radians(yaw)
            label = f"{name} {i + 1}" if len(ps) > 1 else name
            if kind == "screw":
                pts = [(x, y), (x + SCREW_MM * math.cos(a), y + SCREW_MM * math.sin(a))]
            elif kind == "bin":
                d = BIN_MM / 2
                pts = [(x + c * math.cos(a) - s * math.sin(a), y + c * math.sin(a) + s * math.cos(a))
                       for c, s in ((-d, -d), (d, -d), (d, d), (-d, d))]
            else:
                pts = [(x + OTHER_R_MM * math.cos(t), y + OTHER_R_MM * math.sin(t))
                       for t in (2 * math.pi * j / 16 for j in range(16))]
            shapes.append({"label": label, "kind": kind, "pts": uv(pts)})
    return shapes

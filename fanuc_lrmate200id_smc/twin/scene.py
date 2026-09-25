"""Tabletop scene for the twin: the perception pipeline's geometric_view.json -> table slab + one box per object.

numpy only (no shapely / pxr). Objects are axis-aligned-in-yaw boxes standing on the table top (z = 0 in the twin frame):
footprint from the object's footprint polygon, height from the calibration's `object_top_z_m`. Names come from a VLM and are
free-form ("black_screw_top_left"), so heights/colours are matched on words in the name; anything unknown gets a default.
"""
import glob
import json
import os
import re
from typing import NamedTuple

import numpy as np

TABLE_THICKNESS_M = 0.03
FLAT_WORDS = ("board", "grid", "region", "cell")        # drawn on / flat on the table: thin slab, not a solid block
FLAT_HEIGHT_M = 0.004
DEFAULT_COLOR = (0.72, 0.72, 0.75)
COLOR_WORDS = {
    "black": (0.06, 0.06, 0.07), "white": (0.92, 0.92, 0.9), "yellow": (0.95, 0.8, 0.1), "red": (0.8, 0.12, 0.1),
    "blue": (0.12, 0.3, 0.8), "green": (0.15, 0.6, 0.25), "orange": (0.95, 0.5, 0.1), "purple": (0.5, 0.25, 0.65),
    "silver": (0.75, 0.77, 0.8), "grey": (0.5, 0.5, 0.52), "gray": (0.5, 0.5, 0.52), "wooden": (0.6, 0.42, 0.25),
}


class SceneObject(NamedTuple):
    name: str
    xy: tuple          # box centre on the table [m], twin frame (x forward, y left; same as robot base frame)
    yaw: float         # rad about +z
    size: tuple        # (length along yaw, width, height) [m]
    color: tuple


class Scene(NamedTuple):
    source: str
    table: tuple       # (xmin, xmax, ymin, ymax) [m]
    objects: tuple


def polygon_from_wkt(wkt):
    """Outer ring of a WKT POLYGON as an (N, 2) array (closing point dropped)."""
    m = re.search(r"\(\(([^()]*)\)", wkt)
    if not m:
        raise ValueError(f"not a WKT polygon: {wkt[:60]!r}")
    pts = np.array([[float(v) for v in p.split()[:2]] for p in m.group(1).split(",")])
    return pts[:-1] if len(pts) > 1 and np.allclose(pts[0], pts[-1]) else pts


def footprint_box(poly, yaw):
    """(centre xy, (length, width)) of the box with edges along `yaw` that encloses the polygon."""
    c, s = np.cos(yaw), np.sin(yaw)
    local = poly @ np.array([[c, -s], [s, c]])          # rotate by -yaw: rows are (p . u, p . v)
    lo, hi = local.min(0), local.max(0)
    mid = (lo + hi) / 2
    return np.array([c * mid[0] - s * mid[1], s * mid[0] + c * mid[1]]), tuple(hi - lo)


def object_height(name, heights, default):
    """Height from the calibration key that appears first in the name ("screw_in_container" is a screw, "container_with_screw"
    a container; ties -> longest key); thin for flat things; else `default`."""
    low = name.lower()
    keys = [k for k in heights if not k.startswith("_") and k in low]
    if keys:
        return float(heights[min(keys, key=lambda k: (low.index(k), -len(k)))])
    return FLAT_HEIGHT_M if any(w in low for w in FLAT_WORDS) else default


def object_color(name):
    words = re.split(r"[^a-z]+", name.lower())
    for w in words:
        if w in COLOR_WORDS:
            return COLOR_WORDS[w]
    return DEFAULT_COLOR


def load_heights(calib_path):
    """`object_top_z_m` of a tabletop calibration JSON, or {} if the file is missing."""
    try:
        with open(calib_path) as f:
            return {k: v for k, v in json.load(f).get("object_top_z_m", {}).items() if isinstance(v, (int, float))}
    except OSError:
        return {}


def load_scene(path, heights=None, default_height=0.03):
    with open(path) as f:
        g = json.load(f)
    table = polygon_from_wkt(g["table_polygon_wkt"])
    objs = []
    for o in g.get("objects", []):
        if not o.get("footprint_wkt"):
            continue
        yaw = float(o.get("yaw", 0.0))
        xy, (length, width) = footprint_box(polygon_from_wkt(o["footprint_wkt"]), yaw)
        h = object_height(o["name"], heights or {}, default_height)
        objs.append(SceneObject(o["name"], tuple(xy), yaw, (length, width, h), object_color(o["name"])))
    lo, hi = table.min(0), table.max(0)
    return Scene(path, (lo[0], hi[0], lo[1], hi[1]), tuple(objs))


class SceneWatcher:
    """Follows the newest file matching any of the given paths / globs (e.g. runs/*/perceive_*/geometric_view.json)."""

    def __init__(self, patterns, heights=None, default_height=0.03):
        self.patterns, self.heights, self.default_height = list(patterns), heights, default_height
        self._seen = None                                # (path, mtime) of the last scene handed out
        self.error = None

    def newest(self):
        files = [f for p in self.patterns for f in glob.glob(p, recursive=True) if os.path.isfile(f)]
        return max(files, key=os.path.getmtime) if files else None

    def poll(self):
        """A Scene if the newest file is new or changed since the last poll, else None. Half-written files are retried."""
        path = self.newest()
        if path is None:
            return None
        key = (path, os.path.getmtime(path))
        if key == self._seen:
            return None
        try:
            scene = load_scene(path, self.heights, self.default_height)
        except (ValueError, KeyError, OSError) as e:      # includes json.JSONDecodeError
            self.error = f"{type(e).__name__}: {e}"
            return None
        self._seen, self.error = key, None
        return scene

"""Minimal reader for the single-mesh, single-texture GLB files Scaniverse exports (numpy + OpenCV, no trimesh)."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_DTYPES = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


@dataclass
class GlbMesh:
    positions: np.ndarray       # (N, 3) float, glTF frame (metres, +Y up)
    uv: np.ndarray              # (N, 2) float, glTF convention: v = 0 at the TOP row of the image
    faces: np.ndarray           # (M, 3) int
    texture_bytes: bytes        # encoded image (JPEG/PNG) as stored in the file
    texture_mime: str

    def texture(self):
        """Decoded texture, RGB uint8."""
        import cv2

        img = cv2.imdecode(np.frombuffer(self.texture_bytes, np.uint8), cv2.IMREAD_COLOR)
        return img[:, :, ::-1].copy()


def load_glb(path) -> GlbMesh:
    data = Path(path).read_bytes()
    magic, version, _length = struct.unpack("<4sII", data[:12])
    if magic != b"glTF" or version != 2:
        raise ValueError(f"{path}: not a glTF 2.0 binary")
    off, doc, blob = 12, None, b""
    while off < len(data):
        clen, ctype = struct.unpack("<I4s", data[off:off + 8])
        chunk = data[off + 8:off + 8 + clen]
        if ctype == b"JSON":
            doc = json.loads(chunk)
        elif ctype == b"BIN\x00":
            blob = chunk
        off += 8 + clen
    if doc is None:
        raise ValueError(f"{path}: no JSON chunk")

    def view(i):
        bv = doc["bufferViews"][i]
        start = bv.get("byteOffset", 0)
        return blob[start:start + bv["byteLength"]]

    def accessor(i):
        a = doc["accessors"][i]
        raw = view(a["bufferView"])
        n = _WIDTH[a["type"]]
        arr = np.frombuffer(raw, _DTYPES[a["componentType"]], a["count"] * n, a.get("byteOffset", 0))
        return arr.reshape(-1, n) if n > 1 else arr

    meshes = doc.get("meshes", [])
    if len(meshes) != 1 or len(meshes[0]["primitives"]) != 1:
        raise ValueError(f"{path}: expected one mesh with one primitive (Scaniverse export)")
    prim = meshes[0]["primitives"][0]
    pos = accessor(prim["attributes"]["POSITION"]).astype(float)
    uv = accessor(prim["attributes"]["TEXCOORD_0"]).astype(float)
    faces = accessor(prim["indices"]).astype(np.int64).reshape(-1, 3)
    # Apply node transforms if the file has any (Scaniverse writes identity nodes).
    for node in doc.get("nodes", []):
        if node.get("mesh") == 0 and any(k in node for k in ("matrix", "translation", "rotation", "scale")):
            raise NotImplementedError(f"{path}: node transforms are not supported")
    img = doc["images"][0]
    return GlbMesh(pos, uv, faces, bytes(view(img["bufferView"])), img.get("mimeType", "image/jpeg"))

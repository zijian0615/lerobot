# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Render a pick episode through the overhead and wrist cameras.

Pass ``--desk`` to solve the trajectory on the USD robot and render it in this
same Isaac Sim process. ``--traj`` replays a trajectory that was already solved.
The overhead camera is video0 as calibrated in overhead_camera.json, lens distortion included: a wider pinhole
is rendered and remapped (lens.DistortedView), so frames line up with raw 960x540 video0 frames. ``--scan`` puts the
aligned table scan (twin/align_scan.py table.npz) in place of the flat table and its texture.

    OMNI_KIT_ACCEPT_EULA=YES LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 \\
      ~/isaacsim-venv/bin/python twin/record_overhead.py \\
      --desk ~/lerobot/examples/cosmos_edge_fanuc/desk_scene.json --out rgb/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE = os.path.join(HERE, "..", "fanuc_lrmate200id_smc.usda")
CHAIN = os.path.join(HERE, "..", "fanuc_lrmate200id_smc.twin.json")
TABLE_TEXTURE = os.path.join(HERE, "..", "table_texture.png")
OVERHEAD_JSON = os.path.join(HERE, "..", "overhead_camera.json")
WRIST_JSON = os.path.join(HERE, "..", "wrist_camera.json")
# 16:9, half of the 1920x1080 video0 frame. The stage TopCamera is the same pose.
WIDTH, HEIGHT = 960, 540
TOOL = "/World/fanuc/base_link/link_1/link_2/link_3/link_4/link_5/link_6/tool0"
WRIST = TOOL + "/WristCamera"


def _row_matrix(x_axis, y_axis, z_axis, eye):
    """Row-vector transform: local axes are the rows, translation is the last row."""
    from pxr import Gf

    return Gf.Matrix4d(
        x_axis[0], x_axis[1], x_axis[2], 0.0,
        y_axis[0], y_axis[1], y_axis[2], 0.0,
        z_axis[0], z_axis[1], z_axis[2], 0.0,
        eye[0], eye[1], eye[2], 1.0,
    )


def _look_matrix(eye, target, up):
    """USD camera looks along local -Z. `up` is the image-up hint, in the same frame as eye/target."""
    from pxr import Gf

    eye_v, target_v, up_v = Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up)
    forward = (target_v - eye_v).GetNormalized()
    z_axis = -forward
    x_axis = Gf.Cross(up_v, z_axis).GetNormalized()
    y_axis = Gf.Cross(z_axis, x_axis).GetNormalized()
    return _row_matrix(x_axis, y_axis, z_axis, eye_v)


def _set_transform(prim, matrix):
    from pxr import UsdGeom

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(matrix)


def _frame_cameras(stage):
    """Match the calibrated video0 pose and parent a wrist camera to the tool.

    Returns (overhead view, wrist view): a lens.DistortedView for each camera whose JSON has distortion (the USD camera
    then gets the wider pinhole lens of view.render_wh, and view.apply() turns its renders into WIDTH x HEIGHT
    frames like the real camera's), else None.
    """
    import json

    from pxr import Gf, UsdGeom

    with open(OVERHEAD_JSON, encoding="utf-8") as handle:
        calib = json.load(handle)
    usd = calib["usd"]
    overhead_prim = stage.GetPrimAtPath("/World/TopCamera")
    if not overhead_prim.IsValid():
        raise RuntimeError("stage has no /World/TopCamera")
    rows = usd["matrix"]
    _set_transform(overhead_prim, _row_matrix(rows[0][:3], rows[1][:3], rows[2][:3], rows[3][:3]))
    view = None
    lens = usd
    if calib.get("dist"):
        from lens import DistortedView

        height, width = calib["image_hw"]
        view = DistortedView(calib["K"], calib["dist"], (width, height), (WIDTH, HEIGHT))
        lens = view.usd_lens()
    overhead = UsdGeom.Camera(overhead_prim)
    overhead.GetFocalLengthAttr().Set(float(lens["focal_mm"]))
    overhead.GetHorizontalApertureAttr().Set(float(lens["horizontal_aperture"]))
    overhead.GetVerticalApertureAttr().Set(float(lens["vertical_aperture"]))
    overhead.GetClippingRangeAttr().Set(Gf.Vec2f(0.02, 10.0))

    with open(WRIST_JSON, encoding="utf-8") as handle:
        wrist_model = json.load(handle)
    wrist = UsdGeom.Camera.Define(stage, WRIST)
    _set_transform(
        wrist.GetPrim(),
        _look_matrix(wrist_model["eye_tool"], wrist_model["target_tool"], wrist_model["up_tool"]),
    )
    wrist_view = None
    wrist_lens = wrist_model
    if wrist_model.get("dist"):
        from lens import DistortedView

        height, width = wrist_model["image_hw"]
        wrist_view = DistortedView(wrist_model["K"], wrist_model["dist"], (width, height), (WIDTH, HEIGHT))
        wrist_lens = wrist_view.usd_lens()
    wrist.GetFocalLengthAttr().Set(float(wrist_lens["focal_mm"]))
    wrist.GetHorizontalApertureAttr().Set(float(wrist_lens["horizontal_aperture"]))
    wrist.GetVerticalApertureAttr().Set(float(wrist_lens["vertical_aperture"]))
    wrist.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 2.0))
    origin = UsdGeom.XformCache().GetLocalToWorldTransform(overhead_prim).ExtractTranslation()
    print(f"[record] overhead world xyz {origin[0]:.3f} {origin[1]:.3f} {origin[2]:.3f}", flush=True)
    return view, wrist_view


def _rgb(annotator, label, size=(WIDTH, HEIGHT)):
    import numpy as np

    data = annotator.get_data()
    if hasattr(data, "numpy"):
        data = data.numpy()
    image = np.asarray(data)
    if image.ndim != 3 or image.shape[1::-1] != tuple(size):
        raise RuntimeError(f"{label} camera returned {getattr(image, 'shape', None)}")
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    return np.ascontiguousarray(image[:, :, :3], dtype=np.uint8)


USD_OBJECTS_ROOT = "/World/task_objects"


def _add_usd_objects(stage, objects):
    """Reference each scanned asset (bottom-centre origin, z up) under USD_OBJECTS_ROOT with translate + rotateZ ops."""
    from pxr import Sdf, UsdGeom

    UsdGeom.Xform.Define(stage, USD_OBJECTS_ROOT)
    for obj in objects:
        prim = UsdGeom.Xform.Define(stage, f"{USD_OBJECTS_ROOT}/{obj['name']}")
        prim.GetPrim().GetReferences().AddReference(os.path.abspath(obj["usd"]))
        # the renderer only needs the pose; keep the assets kinematic so nothing simulates
        prim.GetPrim().CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(True)
        prim.ClearXformOpOrder()
        prim.AddTranslateOp()
        prim.AddRotateZOp()


def _place(writer, objects):
    from pxr import Gf, UsdGeom

    from omni_twin import SCENE_ROOT

    if objects and "usd" in objects[0]:
        for obj in objects:
            ops = UsdGeom.Xformable(writer.stage.GetPrimAtPath(f"{USD_OBJECTS_ROOT}/{obj['name']}")).GetOrderedXformOps()
            ops[0].Set(Gf.Vec3d(*map(float, obj["pos"])))
            ops[1].Set(float(math.degrees(obj["yaw"])))
        return

    # set_scene names prims obj_XX_<identifier> in layout order.
    root = writer.stage.GetPrimAtPath(SCENE_ROOT)
    boxes = sorted(
        (child for child in root.GetChildren() if child.GetName().startswith("obj_")),
        key=lambda prim: prim.GetName(),
    )
    if len(boxes) != len(objects):
        raise RuntimeError(f"scene has {len(boxes)} boxes, trajectory has {len(objects)}")
    for prim, obj in zip(boxes, objects, strict=True):
        op = UsdGeom.Xformable(prim).GetOrderedXformOps()[0]
        op.Set(Gf.Vec3d(*obj["center"]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Render overhead frames for sim trajectories")
    parser.add_argument("--traj", default="", help="Already-solved trajectory JSON.")
    parser.add_argument("--desk", default="", help="Solve this desk on the USD robot, then render.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stage", default=STAGE)
    parser.add_argument("--chain", default=CHAIN)
    parser.add_argument("--table-texture", default=TABLE_TEXTURE)
    parser.add_argument("--scan", default="", help="aligned table scan (align_scan.py table.npz) instead of the flat table")
    parser.add_argument("--max-episodes", type=int, default=0, help="0 = all")
    parser.add_argument(
        "--frames",
        default="",
        help="Comma-separated frame indexes to render (empty = every frame).",
    )
    args = parser.parse_args(argv)
    if bool(args.traj) == bool(args.desk):
        parser.error("pass exactly one of --traj or --desk")
    wanted = {int(part) for part in args.frames.split(",") if part.strip()}

    if args.desk:
        import numpy as np

        from isaac_pick import load_desk, solve_episode
        from usd_chain import Chain

        chain = Chain.load(args.chain)
        episode = solve_episode(chain, load_desk(args.desk), np.random.default_rng(args.seed))
        episodes = [episode]
        print(
            f"[isaac] solved {episode['pick']} in {len(episode['frames'])} frames on the USD chain",
            flush=True,
        )
    else:
        episodes = json.loads(open(args.traj, encoding="utf-8").read())
    os.makedirs(args.out, exist_ok=True)
    if args.desk:
        with open(os.path.join(args.out, "traj.json"), "w", encoding="utf-8") as handle:
            json.dump(episodes, handle)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "width": WIDTH, "height": HEIGHT})
    try:
        import omni.replicator.core as rep
        import omni.usd
        from PIL import Image

        from joint_map import FINGER_OPEN_M, fanuc_to_model
        from omni_twin import PxrWriter
        from scene import Scene, SceneObject
        from usd_chain import Chain

        ctx = omni.usd.get_context()
        ctx.open_stage(os.path.abspath(args.stage))
        for _ in range(15):
            app.update()
        stage = ctx.get_stage()
        chain = Chain.load(args.chain)
        writer = PxrWriter(stage, chain)
        first = episodes[0]["frames"][0]
        view, wrist_view = _frame_cameras(stage)
        if args.table_texture and os.path.isfile(args.table_texture) and not args.scan:
            from table_texture import load_texture_meta

            bounds, _meta = load_texture_meta(args.table_texture)
            writer.set_table_texture(args.table_texture, bounds)
        usd_objects = bool(first["objects"]) and "usd" in first["objects"][0]
        writer.set_scene(
            Scene(
                "sim",
                tuple(episodes[0]["table"]),
                () if usd_objects else tuple(
                    SceneObject(
                        obj["name"],
                        (obj["center"][0], obj["center"][1]),
                        float(obj["yaw"]),
                        tuple(obj["size"]),
                        tuple(obj["color"]),
                    )
                    for obj in first["objects"]
                ),
            )
        )
        if usd_objects:
            _add_usd_objects(stage, first["objects"])
        if args.scan:
            from pxr import UsdGeom

            from align_scan import _add_scan
            from omni_twin import SCENE_ROOT

            _add_scan(stage, args.scan, lit=False)
            UsdGeom.Imageable(stage.GetPrimAtPath(f"{SCENE_ROOT}/table")).MakeInvisible()
        overhead_wh = view.render_wh if view else (WIDTH, HEIGHT)
        overhead_rp = rep.create.render_product("/World/TopCamera", overhead_wh)
        wrist_wh = wrist_view.render_wh if wrist_view else (WIDTH, HEIGHT)
        wrist_rp = rep.create.render_product(WRIST, wrist_wh)
        overhead_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        wrist_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        overhead_rgb.attach([overhead_rp])
        wrist_rgb.attach([wrist_rp])
        for _ in range(8):
            app.update()
            rep.orchestrator.step()

        chosen = episodes if args.max_episodes <= 0 else episodes[: args.max_episodes]
        for index, episode in enumerate(chosen):
            folder = os.path.join(args.out, f"episode_{index:03d}")
            os.makedirs(os.path.join(folder, "overhead"), exist_ok=True)
            os.makedirs(os.path.join(folder, "wrist"), exist_ok=True)
            for frame_index, frame in enumerate(episode["frames"]):
                if wanted and frame_index not in wanted:
                    continue
                q = fanuc_to_model(frame["joints_deg"], "coupled")
                finger_m = float(frame.get("finger_m", FINGER_OPEN_M * (1.0 - float(frame["gripper"]))))
                writer.set_pose(chain.joint_values(q, finger_m))
                _place(writer, frame["objects"])
                app.update()
                rep.orchestrator.step()
                overhead = _rgb(overhead_rgb, "overhead", overhead_wh)
                Image.fromarray(view.apply(overhead) if view else overhead).save(
                    os.path.join(folder, "overhead", f"{frame_index:06d}.png")
                )
                wrist_img = _rgb(wrist_rgb, "wrist", wrist_wh)
                Image.fromarray(wrist_view.apply(wrist_img) if wrist_view else wrist_img).save(
                    os.path.join(folder, "wrist", f"{frame_index:06d}.png")
                )
            print(f"[record] episode {index} frames {len(episode['frames'])}", flush=True)
        print(f"[record] wrote {len(chosen)} episodes under {args.out}", flush=True)
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

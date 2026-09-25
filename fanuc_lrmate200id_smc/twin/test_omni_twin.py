"""Tests for the Omniverse twin. Run:
    .venv-twin/bin/pytest twin/test_omni_twin.py -q
The USD is checked WITHOUT pxr: the generated .usda is parsed back and its prim transforms are composed the way USD does
(xformOpOrder outermost-first) and compared with MuJoCo's forward kinematics. The pxr-based writer test only runs where
pxr is importable (Isaac Sim's Python, or `pip install usd-core` on x86 / macOS).
"""
import os
import re
import socket
import sys
import time

import mujoco
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_usd  # noqa: E402
import omni_twin  # noqa: E402
from joint_map import fanuc_to_model, flange_world_mm, limit_violations  # noqa: E402
from sources import JointStatePublisher, UdpJointSource  # noqa: E402
from udp_relay import UdpRelay  # noqa: E402
from usd_chain import Chain, axis_angle_quat, quat_to_mat  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(build_usd.ROBOT_XML)


@pytest.fixture(scope="module")
def built(model, tmp_path_factory):
    d = tmp_path_factory.mktemp("usd")
    usda, spec = build_usd.build(model)
    (d / "twin.usda").write_text(usda)
    import json
    (d / "twin.json").write_text(json.dumps(spec))
    return usda, Chain(spec), str(d / "twin.usda"), str(d / "twin.json")


def random_pose(model, rng):
    lo, hi = model.jnt_range[:8, 0], model.jnt_range[:8, 1]
    q = lo + rng.random(8) * (hi - lo)
    q[7] = q[6]                                     # fingers move together
    return q


def mj_forward(model, q):
    d = mujoco.MjData(model)
    d.qpos[:] = q
    mujoco.mj_forward(model, d)
    return d


def parse_usda(text):
    """{prim path: {'type', 'translate', 'orient', 'scale'}} for the static xformOps (the ':joint' ops are not read)."""
    prims, stack, pending = {}, [], None
    for line in text.splitlines():
        s = line.strip()
        m = re.match(r'def (\w+) "([^"]+)"', s)
        if m:
            pending = (m.group(1), m.group(2))
        elif s == "{":
            stack.append(pending[1])
            prims["/" + "/".join(stack)] = {"type": pending[0]}
        elif s == "}":
            stack.pop()
        elif stack:
            m = re.match(r"(?:double3|float3|quatf) xformOp:(translate|orient|scale) = \(([^)]*)\)", s)
            if m:
                prims["/" + "/".join(stack)][m.group(1)] = np.array([float(x) for x in m.group(2).split(",")])
    assert not stack
    return prims


def usd_world(prims, path, joint_values, chain):
    """Compose the prim's world transform from its ancestors' xformOps: T * R_static * (R|T)_joint."""
    R, p = np.eye(3), np.zeros(3)
    by_path = {b["path"]: b for b in chain.joints.values()}
    parts = path.strip("/").split("/")
    for i in range(1, len(parts) + 1):
        pr = prims["/" + "/".join(parts[:i])]
        if "translate" in pr:
            p = p + R @ pr["translate"]
        if "orient" in pr:
            R = R @ quat_to_mat(pr["orient"])
        j = by_path.get("/" + "/".join(parts[:i]))
        if j:
            spec, v = j["joint"], joint_values[j["joint"]["name"]]
            if spec["type"] == "hinge":
                R = R @ quat_to_mat(axis_angle_quat(spec["axis"], v))
            else:
                p = p + R @ (np.asarray(spec["axis"]) * v)
    return R, p


def test_usda_bodies_and_geoms_match_mujoco(model, built):
    usda, chain, *_ = built
    prims = parse_usda(usda)
    rng = np.random.default_rng(0)
    visible = [g for g in range(model.ngeom) if model.geom_group[g] == build_usd.VISIBLE_GROUP]
    assert visible
    for _ in range(5):
        q = random_pose(model, rng)
        d = mj_forward(model, q)
        vals = chain.joint_values(q[:6], q[6])
        for b in chain.bodies:
            R, p = usd_world(prims, b["path"], vals, chain)
            i = model.body(b["name"]).id
            np.testing.assert_allclose(p, d.xpos[i], atol=1e-9, err_msg=b["name"])
            np.testing.assert_allclose(R, d.xmat[i].reshape(3, 3), atol=1e-9, err_msg=b["name"])
        for g in visible:
            body = chain.bodies[[b["name"] for b in chain.bodies].index(model.body(int(model.geom_bodyid[g])).name)]
            path = f"{body['path']}/{model.geom(g).name or body['name'] + '_visual'}"
            R, p = usd_world(prims, path, vals, chain)
            np.testing.assert_allclose(p, d.geom_xpos[g], atol=1e-9, err_msg=path)
            np.testing.assert_allclose(R, d.geom_xmat[g].reshape(3, 3), atol=1e-9, err_msg=path)


def test_usda_prims_and_meshes(model, built):
    usda, chain, *_ = built
    prims = parse_usda(usda)
    for b in chain.bodies:
        assert prims[b["path"]]["type"] == "Xform"
    joint_ops = [ln for ln in usda.splitlines() if re.search(r"xformOp:(orient|translate):joint =", ln)]
    assert len(joint_ops) == 8                        # J1..J6 + two fingers
    for name in ("ground", "status_light", "trail", "Camera"):
        assert f"/World/{name}" in prims
    n_link_meshes = sum(1 for g in range(model.ngeom)
                        if model.geom_group[g] == 2 and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH)
    assert sum(p["type"] == "Mesh" for p in prims.values()) == n_link_meshes + 1   # link parts + ground
    for m in re.finditer(r"int\[\] faceVertexIndices = \[([^\]]*)\]\n\s*point3f\[\] points = \[(.*)\]", usda):
        idx = np.array([int(x) for x in m.group(1).split(",")])
        n_pts = m.group(2).count("(")
        assert idx.max() < n_pts and idx.min() >= 0
    assert usda.count("{") == usda.count("}")
    # every material a prim binds to is defined
    for target in set(re.findall(r"rel material:binding = <([^>]+)>", usda)):
        assert prims[target]["type"] == "Material", target


def test_chain_matches_mujoco_and_joint_map(model, built):
    _, chain, *_ = built
    rng = np.random.default_rng(1)
    for _ in range(5):
        q = random_pose(model, rng)
        d = mj_forward(model, q)
        vals = chain.joint_values(q[:6], q[6])
        np.testing.assert_allclose(chain.site("tcp", vals), d.site("tcp").xpos, atol=1e-9)
        np.testing.assert_allclose(chain.flange_world_mm(vals), flange_world_mm(model, d), atol=1e-6)
    over = np.zeros(6)
    over[1] = np.radians(170)                        # beyond J2's +145 deg
    assert chain.limit_violations(over) == limit_violations(model, over)
    assert [v[0] for v in chain.limit_violations(over)] == [2]


def test_camera_looks_at_target(built):
    usda, *_ = built
    m = re.search(r"matrix4d xformOp:transform = \((.*)\)\n", usda).group(1)
    rows = np.array([[float(x) for x in r.split(",")] for r in re.findall(r"\(([^()]*)\)", m)])
    x, y, z, eye = rows
    fwd = -z[:3]
    d = np.array(build_usd.CAM_LOOKAT) - eye[:3]
    np.testing.assert_allclose(np.linalg.norm(d), build_usd.CAM_AZ_EL_DIST[2], atol=1e-6)
    np.testing.assert_allclose(d / np.linalg.norm(d), fwd, atol=1e-6)
    assert y[2] > 0                                  # camera up is world up
    np.testing.assert_allclose([x @ y, x @ z, y @ z], 0, atol=1e-9)


def run_ms(a, chain, src, writer, seconds):
    t_end = time.monotonic() + seconds
    omni_twin.run(a, chain, src, writer, lambda: time.sleep(0.01), lambda: time.monotonic() < t_end)


def test_pipeline_udp_to_writer_and_status(built):
    _, chain, *_ = built
    a = omni_twin.argparse.Namespace(gripper_mm=2.0, trail=5, j3_mode="coupled")
    src = UdpJointSource(0).start()
    writer = omni_twin.NullWriter()
    try:
        run_ms(a, chain, src, writer, 0.2)
        assert writer.status == "none" and writer.joint_values is None
        joints = (10.0, 30.0, -20.0, 5.0, -40.0, 15.0)
        JointStatePublisher("127.0.0.1", src.port).publish(joints)
        time.sleep(0.1)
        run_ms(a, chain, src, writer, 0.2)
        assert writer.status == "live"
        q = fanuc_to_model(joints, "coupled")
        for i in range(6):
            assert writer.joint_values[f"joint_{i + 1}"] == pytest.approx(q[i])
        assert writer.joint_values["finger_l"] == writer.joint_values["finger_r"] == pytest.approx(0.002)
        assert len(writer.trail) >= 2 and np.allclose(writer.trail[-1], chain.site("tcp", writer.joint_values))
        run_ms(a, chain, src, writer, omni_twin.STALE_S + 0.2)
        assert writer.status == "stale"
    finally:
        src.close()


def test_main_no_render_demo(built):
    _, _, stage, chain_path = built
    w = omni_twin.main(["--source", "demo", "--no-render", "--duration", "0.4", "--stage", stage, "--chain", chain_path])
    assert w.status == "live" and set(w.joint_values) == {f"joint_{i}" for i in range(1, 7)} | {"finger_l", "finger_r"}


def test_main_reports_missing_stage(tmp_path, capsys):
    with pytest.raises(SystemExit):
        omni_twin.main(["--source", "demo", "--no-render", "--stage", str(tmp_path / "nope.usda")])
    assert "build_usd.py" in capsys.readouterr().err


def test_udp_relay_fans_out():
    rx = [socket.socket(socket.AF_INET, socket.SOCK_DGRAM) for _ in range(2)]
    for s in rx:
        s.bind(("127.0.0.1", 0))
        s.settimeout(2.0)
    relay = UdpRelay(0, [("127.0.0.1", s.getsockname()[1]) for s in rx], bind="127.0.0.1").start()
    try:
        JointStatePublisher("127.0.0.1", relay.port).publish([1, 2, 3, 4, 5, 6])
        got = [s.recvfrom(4096)[0] for s in rx]
        assert got[0] == got[1] and b"joints_deg" in got[0]
    finally:
        relay.close()
        for s in rx:
            s.close()


def test_pxr_writer_matches_mujoco(model, built):
    """Only where pxr exists: open the .usda, drive it through PxrWriter and compare the world pose with MuJoCo."""
    Usd = pytest.importorskip("pxr.Usd")
    from pxr import UsdGeom
    _, chain, stage_path, _ = built
    stage = Usd.Stage.Open(stage_path)
    writer = omni_twin.PxrWriter(stage, chain)
    q = random_pose(model, np.random.default_rng(2))
    writer.set_pose(chain.joint_values(q[:6], q[6]))
    d = mj_forward(model, q)
    cache = UsdGeom.XformCache()
    for b in chain.bodies:
        m = np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(b["path"]))).T   # USD matrices are row-vector
        np.testing.assert_allclose(m[:3, 3], d.xpos[model.body(b["name"]).id], atol=1e-5, err_msg=b["name"])
        np.testing.assert_allclose(m[:3, :3], d.xmat[model.body(b["name"]).id].reshape(3, 3), atol=1e-5, err_msg=b["name"])
    writer.set_status("live")
    writer.set_trail([(0, 0, 0), (0.1, 0, 0), (0.1, 0.1, 0)])
    assert len(stage.GetPrimAtPath(omni_twin.TRAIL_PRIM).GetAttribute("points").Get()) == 3


# ---- tabletop scene (scene.py) ----
import json  # noqa: E402

import scene as scn  # noqa: E402

REAL_VIEW = os.path.join(HERE, "..", "..", "examples", "tabletop_perception", "runs", "20260921_042733", "perceive_01", "geometric_view.json")


def rect_wkt(center, size, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    pts = [np.asarray(center) + R @ np.array([sx * size[0] / 2, sy * size[1] / 2]) for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1), (1, 1))]
    return "POLYGON ((" + ", ".join(f"{p[0]} {p[1]}" for p in pts) + "))"


def write_view(path, objects):
    view = {"objects": [{"name": n, "xy": list(c), "yaw": y, "footprint_wkt": rect_wkt(c, sz, y)} for n, c, sz, y in objects],
            "table_polygon_wkt": "POLYGON ((-0.45 -0.35, 0.45 -0.35, 0.45 0.35, -0.45 0.35, -0.45 -0.35))"}
    path.write_text(json.dumps(view))


def test_footprint_box_recovers_centre_size_yaw():
    for yaw in (0.0, 0.6, -1.2, 2.5):
        poly = scn.polygon_from_wkt(rect_wkt((0.31, -0.12), (0.028, 0.010), yaw))
        assert len(poly) == 4
        c, (length, width) = scn.footprint_box(poly, yaw)
        np.testing.assert_allclose(c, (0.31, -0.12), atol=1e-9)
        np.testing.assert_allclose((length, width), (0.028, 0.010), atol=1e-9)


def test_object_height_and_color():
    h = {"_comment": "x", "box": 0.05, "screw": 0.01, "container": 0.03}
    f = lambda n: scn.object_height(n, h, 0.02)  # noqa: E731
    assert f("black_screw_top_left") == 0.01
    assert f("screw_in_container") == 0.01 and f("container_with_screw") == 0.03
    assert f("yellow_rimmed_container") == 0.03 and f("top_left_box") == 0.05
    assert f("tic_tac_toe_board") == scn.FLAT_HEIGHT_M and f("teddy_bear") == 0.02
    assert scn.object_color("yellow_container") == scn.COLOR_WORDS["yellow"]
    assert scn.object_color("bear") == scn.DEFAULT_COLOR


def test_load_scene_and_watcher(tmp_path):
    p = tmp_path / "run1" / "perceive_01" / "geometric_view.json"
    p.parent.mkdir(parents=True)
    write_view(p, [("black_screw", (0.4, -0.1), (0.028, 0.01), 0.3), ("box", (0.2, 0.2), (0.06, 0.04), -0.5)])
    w = scn.SceneWatcher([str(tmp_path / "*" / "perceive_*" / "geometric_view.json")], {"screw": 0.01, "box": 0.05})
    s = w.poll()
    assert [o.name for o in s.objects] == ["black_screw", "box"]
    assert s.table == (-0.45, 0.45, -0.35, 0.35)
    np.testing.assert_allclose(s.objects[0].size, (0.028, 0.01, 0.01), atol=1e-9)
    np.testing.assert_allclose(s.objects[1].xy, (0.2, 0.2), atol=1e-9)
    assert w.poll() is None                                  # unchanged
    p.write_text('{"objects": [')                            # half-written file: keep the old scene, report, retry
    os.utime(p, (time.time() + 1, time.time() + 1))
    assert w.poll() is None and w.error
    write_view(p, [])
    os.utime(p, (time.time() + 2, time.time() + 2))
    s2 = w.poll()
    assert s2 is not None and s2.objects == () and w.error is None
    p2 = tmp_path / "run2" / "perceive_01" / "geometric_view.json"   # a newer run takes over
    p2.parent.mkdir(parents=True)
    write_view(p2, [("screw", (0.1, 0.1), (0.02, 0.01), 0.0)])
    os.utime(p2, (time.time() + 3, time.time() + 3))
    assert [o.name for o in w.poll().objects] == ["screw"]


@pytest.mark.skipif(not os.path.exists(REAL_VIEW), reason="no saved perception run")
def test_real_perception_output_loads():
    s = scn.load_scene(REAL_VIEW, scn.load_heights(omni_twin.CALIB))
    assert len(s.objects) == 6 and s.table == (-0.45, 0.45, -0.35, 0.35)
    for o in s.objects:
        assert 0.005 < o.size[0] < 0.3 and 0.005 < o.size[1] < 0.3 and 0 < o.size[2] <= 0.05, o
        assert abs(o.xy[0]) < 0.5 and abs(o.xy[1]) < 0.4


def test_run_shows_scene_from_watcher(built, tmp_path):
    _, chain, *_ = built
    write_view(tmp_path / "geometric_view.json", [("screw", (0.4, 0.0), (0.03, 0.01), 0.0)])
    a = omni_twin.argparse.Namespace(gripper_mm=4.0, trail=0, j3_mode="coupled")
    src = UdpJointSource(0).start()
    writer = omni_twin.NullWriter()
    try:
        omni_twin.run(a, chain, src, writer, lambda: time.sleep(0.01), (lambda t=time.monotonic() + 0.3: time.monotonic() < t),
                      scene=scn.SceneWatcher([str(tmp_path / "geometric_view.json")], {"screw": 0.01}))
    finally:
        src.close()
    assert writer.scene is not None and writer.scene.objects[0].size[2] == 0.01


def test_pxr_writer_scene(built):
    Usd = pytest.importorskip("pxr.Usd")
    from pxr import UsdGeom
    _, chain, stage_path, _ = built
    stage = Usd.Stage.Open(stage_path)
    writer = omni_twin.PxrWriter(stage, chain)
    obj = scn.SceneObject("black_screw_top_left", (0.4, -0.1), 0.5, (0.03, 0.01, 0.01), (0.06, 0.06, 0.07))
    writer.set_scene(scn.Scene("x", (-0.45, 0.45, -0.35, 0.35), (obj,)))
    writer.set_scene(scn.Scene("x", (-0.45, 0.45, -0.35, 0.35), (obj, obj)))     # replacing must not accumulate or fail
    kids = [c.GetName() for c in stage.GetPrimAtPath(omni_twin.SCENE_ROOT).GetChildren()]
    assert kids == ["table", "obj_00_black_screw_top_left", "obj_01_black_screw_top_left"]
    cache = UsdGeom.XformCache()
    m = np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(f"{omni_twin.SCENE_ROOT}/{kids[1]}"))).T
    np.testing.assert_allclose(m[:3, 3], (0.4, -0.1, 0.005), atol=1e-6)               # sits on the table top (z = 0)
    np.testing.assert_allclose(np.linalg.norm(m[:3, 0]), 0.03, atol=1e-6)             # scale = box length
    np.testing.assert_allclose(m[:2, 0] / 0.03, (np.cos(0.5), np.sin(0.5)), atol=1e-6)
    gz = np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(omni_twin.GROUND_PRIM))).T[2, 3]
    assert gz == pytest.approx(-scn.TABLE_THICKNESS_M)


def test_usd_chain_ik_reaches_the_black_cube():
    import json

    from isaac_pick import GRASP_Z, solve_episode
    from joint_map import fanuc_to_model

    chain = Chain.load(os.path.join(HERE, "..", "fanuc_lrmate200id_smc.twin.json"))
    desk_path = os.path.join(HERE, "..", "..", "examples", "cosmos_edge_fanuc", "desk_scene.json")
    desk = json.loads(open(desk_path, encoding="utf-8").read())
    episode = solve_episode(chain, desk, np.random.default_rng(0))
    assert episode["solver"] == "isaac_usd_chain"
    assert episode["pick"] == "black_cube"
    grasp = next(frame for frame in episode["frames"] if frame["gripper"] >= 1.0)
    q = fanuc_to_model(grasp["joints_deg"], "coupled")
    rotation, position = chain.site_frame("tcp", chain.joint_values(q, 0.0))
    target = np.array([desk["cubes"][0]["xy"][0], desk["cubes"][0]["xy"][1], GRASP_Z])
    assert np.linalg.norm(position - target) < 0.008
    assert rotation[:, 2] @ np.array([0.0, 0.0, -1.0]) > 0.998


def test_top_camera_matches_the_tilted_overhead(built):
    usda, *_ = built
    blk = usda[usda.index('def Camera "TopCamera"'):]
    m = re.search(r"matrix4d xformOp:transform = \((.*)\)\n", blk).group(1)
    right, up, back, eye = np.array([[float(v) for v in r.split(",")] for r in re.findall(r"\(([^()]*)\)", m)])
    assert right[1] > 0.99                                        # image right is table +y (calibrated: ~0.3 deg off)
    np.testing.assert_allclose(np.cross(right[:3], up[:3]), back[:3], atol=1e-6)
    look = -back[:3]
    assert look[2] < -0.5 and look[0] < -0.2                      # oblique, toward -x, not straight down
    assert 0.4 < eye[2] < 1.5 and "verticalAperture" in blk


# ---- table texture from the overhead camera (table_texture.py) ----
import table_texture as tt  # noqa: E402
from PIL import Image  # noqa: E402

H_TRUE = np.array([[6.8e-05, 1.569e-03, -0.4545], [1.225e-03, -3.5e-05, -1.066], [2.09e-04, 5.94e-04, 1.0]])


def test_homography_recovers_exact_mapping_and_real_samples_are_accurate():
    rng = np.random.default_rng(0)
    uv = rng.uniform([400, 450], [1250, 800], size=(12, 2))
    H = tt.fit_homography(uv, tt.apply_homography(H_TRUE, uv))
    np.testing.assert_allclose(H, H_TRUE / H_TRUE[2, 2], atol=1e-6)
    with pytest.raises(ValueError):
        tt.fit_homography(uv[:3], uv[:3])
    if os.path.exists(tt.SAMPLES):
        uv, xy, hw = tt.load_samples(tt.SAMPLES)
        assert hw == (1080, 1920)
        assert np.sqrt((tt.leave_one_out_mm(uv, xy) ** 2).mean()) < 10.0     # measured 6.9 mm; the stored K+affine model is 14 mm


def test_rectify_maps_table_points_to_the_right_pixels():
    h, w = 1080, 1920
    v, u = np.mgrid[0:h, 0:w]
    img = np.stack([u * 255 // (w - 1), v * 255 // (h - 1), np.zeros_like(u)], -1).astype(np.uint8)   # colour encodes (u, v)
    bounds, ppm = (0.10, 0.40, -0.10, 0.20), 200
    tex, ok = tt.rectify(img, H_TRUE, bounds, ppm)
    assert tex.shape == (60, 60, 3) and ok.all()                  # rows along x, cols along y
    for r, c in ((0, 0), (59, 0), (0, 59), (30, 45)):
        x, y = bounds[0] + (r + 0.5) / ppm, bounds[2] + (c + 0.5) / ppm
        uu, vv = tt.apply_homography(np.linalg.inv(H_TRUE), [[x, y]])[0]
        np.testing.assert_allclose(tex[r, c, :2], (uu * 255 / (w - 1), vv * 255 / (h - 1)), atol=1.5)
    tex2, ok2 = tt.rectify(img, H_TRUE, bounds, ppm, masks=[(0.10, 0.20, -0.10, 0.20)])
    assert not ok2[:20].any() and ok2[30:].all()                  # a table-frame mask cuts out the first rows
    assert (tex2[:20] == tex2[0, 0]).all()                        # fill defaults to the valid pixels' median colour
    np.testing.assert_allclose(tex2[0, 0], np.median(tex2[ok2], axis=0), atol=1.5)
    far, ok3 = tt.rectify(img, H_TRUE, (5.0, 5.2, 5.0, 5.2), 100, fill=(1, 2, 3))
    assert not ok3.any() and (far == (1, 2, 3)).all()             # outside the camera image -> fill


def test_median_plate_removes_things_that_move_and_cli_writes_texture(tmp_path):
    h, w = 108, 192
    v, u = np.mgrid[0:h, 0:w]
    base = np.stack([200 + (u % 8) * 3, 200 + (v % 8) * 3, np.full_like(u, 200)], -1).astype(np.uint8)
    for i, (r, c) in enumerate(((10, 10), (60, 100), (30, 150))):     # a bright "arm" that is somewhere else in every frame
        f = base.copy()
        f[r:r + 25, c:c + 25] = (255, 0, 0)
        d = tmp_path / f"runs/2026092{i}_000000/perceive_01"
        d.mkdir(parents=True)
        Image.fromarray(f).save(d / "capture_rgb.png")
    H = H_TRUE.copy()
    rng = np.random.default_rng(1)
    uv = rng.uniform([10, 10], [180, 100], size=(10, 2))
    xy = tt.apply_homography(H_TRUE, uv * 10)                          # samples for a 10x smaller image: uv * 10 -> table
    samples = tmp_path / "samples.json"
    samples.write_text(json.dumps({"samples": [{"uv": list(map(float, p)), "true_table_xy": list(map(float, q)), "image_hw": [h, w]}
                                                for p, q in zip(uv, xy)]}))
    out = tmp_path / "tex.png"
    tt.main(["--frames", str(tmp_path / "runs/*/perceive_*/capture_rgb.png"), "--samples", str(samples), "--out", str(out),
             "--bounds", "0.0", "0.3", "0.0", "0.3", "--px-per-m", "100", "--no-robot-mask"])
    assert not (np.asarray(Image.open(out))[..., 1] < 100).any()      # the red patch (G = 0) never survives the median
    bounds, meta = tt.load_texture_meta(str(out))
    assert bounds == (0.0, 0.3, 0.0, 0.3) and meta["frames"] == 3


def test_select_frames_since_and_last(tmp_path):
    for s in ("20260920_100000", "20260921_020000", "20260921_030000"):
        d = tmp_path / s / "perceive_01"
        d.mkdir(parents=True)
        (d / "capture_rgb.png").write_bytes(b"x")
    pat = [str(tmp_path / "*/perceive_*/capture_rgb.png")]
    stamps = lambda fs: [os.path.basename(os.path.dirname(os.path.dirname(f))) for f in fs]  # noqa: E731
    assert stamps(tt.select_frames(pat, "20260921_000000")) == ["20260921_020000", "20260921_030000"]
    assert stamps(tt.select_frames(pat, None, 1)) == ["20260921_030000"]


def test_null_writer_gets_the_texture_from_main(tmp_path, built):
    _, _, stage, chain_path = built
    png = tmp_path / "t.png"
    tt.write_texture(str(png), np.zeros((30, 40, 3), np.uint8), (-0.45, 0.45, -0.35, 0.35), 100)
    w = omni_twin.main(["--source", "demo", "--no-render", "--duration", "0.2", "--stage", stage, "--chain", chain_path,
                        "--table-texture", str(png)])
    assert w.table_texture == (str(png), (-0.45, 0.45, -0.35, 0.35)) and w.scene.objects == ()


def test_pxr_writer_texture(built, tmp_path):
    Usd = pytest.importorskip("pxr.Usd")
    from pxr import UsdGeom, UsdShade
    _, chain, stage_path, _ = built
    stage = Usd.Stage.Open(stage_path)
    writer = omni_twin.PxrWriter(stage, chain)
    png = tmp_path / "t.png"
    tt.write_texture(str(png), np.zeros((30, 40, 3), np.uint8), (-0.45, 0.45, -0.35, 0.35), 100)
    writer.set_table_texture(str(png), (-0.45, 0.45, -0.35, 0.35))
    prim = stage.GetPrimAtPath(omni_twin.TABLE_IMAGE_PRIM)
    pts = [tuple(p) for p in UsdGeom.Mesh(prim).GetPointsAttr().Get()]
    st = [tuple(s) for s in UsdGeom.PrimvarsAPI(prim).GetPrimvar("st").Get()]
    # (x_min, y_min) is the image's top-left, (x_max, y_max) its bottom-right
    by_corner = {(round(p[0], 3), round(p[1], 3)): tuple(round(c, 3) for c in s) for p, s in zip(pts, st)}   # float32 storage
    assert by_corner[(-0.45, -0.35)] == (0, 1) and by_corner[(0.45, 0.35)] == (1, 0)
    assert by_corner[(-0.45, 0.35)] == (1, 1) and by_corner[(0.45, -0.35)] == (0, 0)
    assert all(p[2] == pytest.approx(omni_twin.TEXTURE_LIFT_M) for p in pts)
    mat = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    assert mat.GetPath() == "/World/Looks/table_image"
    assert stage.GetPrimAtPath("/World/Looks/table_image/tex").GetAttribute("inputs:file").Get().path == str(png)

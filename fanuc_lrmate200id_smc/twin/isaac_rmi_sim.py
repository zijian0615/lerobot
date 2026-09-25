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

"""Isaac Sim physics twin that speaks FANUC RMI, so teleop data can be recorded in simulation with the real pipeline.

    # 1. the simulated controller + cameras (Isaac Sim python; a window unless --headless)
    OMNI_KIT_ACCEPT_EULA=YES LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 DISPLAY=:1 \\
      ~/isaacsim-venv/bin/python twin/isaac_rmi_sim.py --scan ../examples/cosmos_edge_fanuc/scanner/scan_204107/table.npz
    # 2. record with the real robot class pointed at it (see README "Physics sim teleop"):
    uv run lerobot-record --robot.type=fanuc --robot.host=127.0.0.1 --robot.port=16101 \\
      --robot.cameras='{overhead: {type: zmq, server_address: 127.0.0.1, port: 5580, camera_name: overhead, width: 960, height: 540, fps: 30},
                        wrist: {type: zmq, server_address: 127.0.0.1, port: 5580, camera_name: wrist, width: 640, height: 360, fps: 30}}' \\
      --teleop.type=telegrip ...

Physics: the NVIDIA LR Mate 200iD/4S asset (articulation, drives; its joints are the model joints of joint_map.py,
checked to 0.01 mm against usd_chain) plus the tool from fanuc_lrmate200id_smc.xml: extension and MHZ2-16D body fixed
to the flange, two prismatic fingers (0..10 mm each, position drives with a force cap), the table as a static box, and
the scanned task objects (task_scene.json) as rigid bodies. Nothing is kinematic: the arm follows joint targets through
its drives and objects move only by contact, so the recorded state is what the physics produced.

Controller: RMI JSON over TCP (FRC_Connect on --port, then a session port). FRC_LinearMotion targets (UF0 mm, W/P/R deg)
queue up; each physics step moves a commanded TCP pose toward the head of the queue at the requested speed, solves IK
(usd_chain, damped least squares) and sets the arm's drive targets. A segment is acknowledged when the commanded pose
reaches it (CNT: the next one starts at once). Port writes on the instruction (DO 3 = open, DO 4 = close, as the real
gripper) set the finger targets. FRC_ReadCartesianPosition / FRC_ReadJointAngles return the measured articulation state
(UT1 = TCP 223 mm, UF0 origin 330 mm above the base plate), FRC_ReadDIN the gripper (1 = closed).
Cameras: video0 (overhead_camera.json) and the wrist camera (wrist_camera.json) rendered through their lens
distortion and published on --zmq-port as lerobot's zmq camera expects ({"timestamps": ..., "images": {name: jpeg64}}).
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import queue
import socket
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
EXAMPLES = os.path.join(ROOT, "..", "examples", "cosmos_edge_fanuc")
ROBOT_USD = os.path.expanduser("~/.cache/isaac_lrmate200id4s/lrmate200id4s.usd")
CHAIN = os.path.join(ROOT, "fanuc_lrmate200id_smc.twin.json")
OVERHEAD_JSON = os.path.join(ROOT, "overhead_camera.json")
WRIST_JSON = os.path.join(ROOT, "wrist_camera.json")
ROBOT = "/World/robot"
BASE_HEIGHT_M = 0.330
PHYSICS_HZ = 240
RENDER_HZ = 30
ARM_JOINTS = ("J1", "J2", "J3", "J4", "J5", "J6")
OPEN_PORT, CLOSE_PORT = 3, 4
FINGER_TRAVEL_M = 0.010
CLOSED_HALF_GAP_M = 0.001        # pad gap 2 mm at zero travel
FINGER_FORCE_N = 40.0
ROT_SPEED = math.radians(90.0)  # rad/s for the commanded tool orientation
# tool0 in link_6 (J6_link) at the flange face: z = link_6 +x, x = link_6 +z, y = link_6 -y
TOOL_POS = (0.070, 0.0, 0.0)
TOOL_ROT = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])  # columns: tool axes in link_6


# ------------------------------------------------------------------ small math (no lerobot import in Isaac's python)

def wpr_to_matrix(w, p, r):
    w, p, r = np.radians([w, p, r])
    rx = np.array([[1, 0, 0], [0, math.cos(w), -math.sin(w)], [0, math.sin(w), math.cos(w)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return rz @ ry @ rx


def matrix_to_wpr(m):
    p = math.asin(max(-1.0, min(1.0, -m[2, 0])))
    if abs(math.cos(p)) < 1e-8:
        return math.degrees(math.atan2(-m[0, 1], m[1, 1])), math.degrees(p), 0.0
    return math.degrees(math.atan2(m[2, 1], m[2, 2])), math.degrees(p), math.degrees(math.atan2(m[1, 0], m[0, 0]))


def quat_wxyz(m):
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(m).as_quat()
    return np.array([w, x, y, z])


def slerp_matrix(a, b, t):
    from scipy.spatial.transform import Rotation, Slerp

    return Slerp([0.0, 1.0], Rotation.from_matrix([a, b]))([t]).as_matrix()[0]


def rotation_angle(a, b):
    return math.acos(max(-1.0, min(1.0, (np.trace(a.T @ b) - 1) / 2)))


# ------------------------------------------------------------------ controller

class Segment:
    def __init__(self, seq, pos, rot, speed_mm_s, port):
        self.seq, self.pos, self.rot, self.speed, self.port = seq, pos, rot, max(float(speed_mm_s), 1.0) / 1000.0, port


class Controller:
    """Commanded TCP pose -> IK -> drive targets; measured joints -> RMI reads."""

    def __init__(self, chain, q0):
        from joint_map import fanuc_to_model, model_to_fanuc

        self.fanuc_to_model, self.model_to_fanuc = fanuc_to_model, model_to_fanuc
        self.chain = chain
        self.lo, self.hi = chain.arm_limits()
        self.q_cmd = np.array(q0, dtype=float)
        self.cmd_rot, self.cmd_pos = chain.site_frame("tcp", chain.joint_values(self.q_cmd, 0.0))
        self.queue: list[Segment] = []
        self.acks: list[int] = []
        self.gripper_closed = False
        self.lock = threading.Lock()
        self.q_meas = self.q_cmd.copy()
        self.finger_meas = FINGER_TRAVEL_M
        self.objects: dict[str, list[float]] = {}  # object name -> measured position (table frame, m)

    def enqueue(self, seg):
        with self.lock:
            self.queue.append(seg)

    def reset(self, q):
        """Home the commanded state (SIM_Reset): empty queue, arm at q, gripper open."""
        with self.lock:
            self.queue.clear()
            self.acks.clear()
            self.q_cmd = np.array(q, dtype=float)
            self.q_meas = self.q_cmd.copy()
            self.cmd_rot, self.cmd_pos = self.chain.site_frame("tcp", self.chain.joint_values(self.q_cmd, 0.0))
            self.gripper_closed = False

    def abort(self):
        with self.lock:
            self.queue.clear()
            self.cmd_rot, self.cmd_pos = self.chain.site_frame("tcp", self.chain.joint_values(self.q_meas, 0.0))
            self.q_cmd = self.q_meas.copy()

    def _ik(self, pos, rot, iters=8):
        q = self.q_cmd.copy()
        for _ in range(iters):
            r, p = self.chain.site_frame("tcp", self.chain.joint_values(q, 0.0))
            err = np.r_[pos - p, 0.5 * sum(np.cross(r[:, k], rot[:, k]) for k in range(3))]
            if np.linalg.norm(err[:3]) < 1e-5 and np.linalg.norm(err[3:]) < 1e-5:
                break
            jac = np.zeros((6, 6))
            eps = 1e-6
            for j in range(6):
                dq = q.copy()
                dq[j] += eps
                r2, p2 = self.chain.site_frame("tcp", self.chain.joint_values(dq, 0.0))
                jac[:3, j] = (p2 - p) / eps
                jac[3:, j] = 0.5 * sum(np.cross(r[:, k], r2[:, k]) for k in range(3)) / eps
            q = np.clip(q + jac.T @ np.linalg.solve(jac @ jac.T + 1e-6 * np.eye(6), err), self.lo, self.hi)
        return q

    def step(self, dt):
        """Advance the commanded pose; returns the joint targets (model rad)."""
        with self.lock:
            if self.queue:
                seg = self.queue[0]
                if seg.port is not None:
                    self.gripper_closed = seg.port == CLOSE_PORT
                    seg.port = None
                delta = seg.pos - self.cmd_pos
                dist = float(np.linalg.norm(delta))
                ang = rotation_angle(self.cmd_rot, seg.rot)
                # position at the segment speed, orientation at up to ROT_SPEED; both finish together
                remaining = max(dist / seg.speed, ang / ROT_SPEED)
                frac = 1.0 if remaining <= dt else dt / remaining
                self.cmd_pos = self.cmd_pos + delta * frac
                self.cmd_rot = slerp_matrix(self.cmd_rot, seg.rot, frac) if ang > 1e-9 else self.cmd_rot
                if frac >= 1.0:
                    self.cmd_pos, self.cmd_rot = seg.pos.copy(), seg.rot.copy()
                    self.acks.append(seg.seq)
                    self.queue.pop(0)
            self.q_cmd = self._ik(self.cmd_pos, self.cmd_rot)
            return self.q_cmd.copy()

    def measured(self, q, finger):
        with self.lock:
            self.q_meas, self.finger_meas = np.array(q, dtype=float), float(finger)

    def cartesian(self):
        with self.lock:
            q = self.q_meas.copy()
        rot, pos = self.chain.site_frame("tcp", self.chain.joint_values(q, 0.0))
        w, p, r = matrix_to_wpr(rot)
        x, y, z = (pos - np.array([0.0, 0.0, BASE_HEIGHT_M])) * 1000.0
        return {"X": x, "Y": y, "Z": z, "W": w, "P": p, "R": r, "Ext1": 0.0, "Ext2": 0.0, "Ext3": 0.0}

    def joints_deg(self):
        with self.lock:
            q = self.q_meas.copy()
        return [float(v) for v in self.model_to_fanuc(q, "coupled")]


# ------------------------------------------------------------------ RMI server

class RmiServer:
    """FRC JSON over TCP: FRC_Connect on `port` answers with a session port; one session at a time."""

    CONFIG = {"UToolNumber": 1, "UFrameNumber": 0, "Front": 1, "Up": 1, "Left": 0, "Flip": 0, "Turn4": 0, "Turn5": 0, "Turn6": 0}

    def __init__(self, controller, host, port):
        self.ctl, self.host = controller, host
        self.main = self._listen(port)
        self.sess = self._listen(0)
        self.port, self.session_port = self.main.getsockname()[1], self.sess.getsockname()[1]
        self.out: queue.Queue = queue.Queue()
        self.client = None
        self.reset_request = None  # set by SIM_Reset, served by the physics loop
        self.send_lock = threading.Lock()
        self.time_tag = 0
        self.stop = threading.Event()

    def _listen(self, port):
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, port))
        s.listen(4)
        return s

    def start(self):
        threading.Thread(target=self._accept_main, daemon=True).start()
        threading.Thread(target=self._accept_session, daemon=True).start()

    def send(self, payload):
        client = self.client
        if client is not None:
            try:
                with self.send_lock:
                    client.sendall((json.dumps(payload) + "\r\n").encode())
            except OSError:
                pass

    def _accept_main(self):
        while not self.stop.is_set():
            conn, _ = self.main.accept()
            with conn:
                buf = conn.recv(4096)
                if b"FRC_Connect" in buf:
                    conn.sendall((json.dumps({"Communication": "FRC_Connect", "ErrorID": 0, "PortNumber": self.session_port,
                                              "MajorVersion": 5, "MinorVersion": 0}) + "\r\n").encode())

    def _accept_session(self):
        while not self.stop.is_set():
            conn, _ = self.sess.accept()
            self.client = conn
            print("[sim-rmi] session connected", flush=True)
            buf = b""
            try:
                while True:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        if line.strip():
                            self._handle(json.loads(line))
            except (OSError, ValueError) as exc:
                print(f"[sim-rmi] session ended: {exc}", flush=True)
            self.client = None
            self.ctl.abort()
            print("[sim-rmi] session closed", flush=True)

    def _handle(self, msg):
        self.time_tag += 1
        if "Communication" in msg:
            if msg["Communication"] == "FRC_Disconnect":
                self.send({"Communication": "FRC_Disconnect", "ErrorID": 0})
            return
        cmd = msg.get("Command")
        if cmd == "FRC_ReadCartesianPosition":
            self.send({"Command": cmd, "ErrorID": 0, "TimeTag": self.time_tag, "Group": 1,
                       "Configuration": dict(self.CONFIG), "Position": self.ctl.cartesian()})
        elif cmd == "FRC_ReadJointAngles":
            j = self.ctl.joints_deg()
            angles = {f"J{i + 1}": v for i, v in enumerate(j)} | {"J7": 0.0, "J8": 0.0, "J9": 0.0}
            self.send({"Command": cmd, "ErrorID": 0, "TimeTag": self.time_tag, "Group": 1, "JointAngle": angles})
        elif cmd == "FRC_ReadDIN":
            self.send({"Command": cmd, "ErrorID": 0, "PortNumber": msg.get("PortNumber"),
                       "PortValue": int(self.ctl.gripper_closed)})
        elif cmd in ("FRC_Initialize", "FRC_SetUFrameUTool", "FRC_Reset"):
            self.send({"Command": cmd, "ErrorID": 0, "GroupMask": 1})
        elif cmd == "FRC_Abort":
            self.ctl.abort()
            self.send({"Command": cmd, "ErrorID": 0})
        elif cmd == "SIM_Reset":  # sim-only: home the arm and lay the objects out again; answered by the physics loop
            self.reset_request = dict(msg)
        elif cmd == "SIM_ReadObjects":  # sim-only: measured object positions, for checks and ground truth
            self.send({"Command": cmd, "ErrorID": 0, "Objects": dict(self.ctl.objects),
                       "Finger": self.ctl.finger_meas})
        elif cmd == "FRC_GetStatus":
            self.send({"Command": cmd, "ErrorID": 0, "ServoReady": 1, "TPMode": 0, "RMIMotionStatus": 1,
                       "ProgramStatus": 1, "SingleStepMode": 0, "NumberUTool": 10, "NumberUFrame": 9,
                       "NextSequenceID": 1, "Override": 100})
        elif cmd is not None:
            self.send({"Command": cmd, "ErrorID": 0})
        elif msg.get("Instruction") == "FRC_LinearMotion":
            pos = msg["Position"]
            target = np.array([pos["X"], pos["Y"], pos["Z"]]) / 1000.0 + [0.0, 0.0, BASE_HEIGHT_M]
            port = None
            if msg.get("PortNumber") is not None and str(msg.get("PortValue", "")).upper() == "ON":
                port = int(msg["PortNumber"])
            speed = float(msg.get("Speed", 250)) if msg.get("SpeedType", "mmSec") == "mmSec" else 250.0
            self.ctl.enqueue(Segment(int(msg["SequenceID"]), target, wpr_to_matrix(pos["W"], pos["P"], pos["R"]), speed, port))
        elif msg.get("Instruction"):
            self.send({"Instruction": msg["Instruction"], "SequenceID": msg.get("SequenceID"), "ErrorID": 0})


# ------------------------------------------------------------------ scene

def _set_pose(prim, pos, rot=None):
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    if rot is not None:
        w, x, y, z = quat_wxyz(rot)
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(w), float(x), float(y), float(z)))


def _box(stage, path, size, pos, color, collide=True, material=None):
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    xf.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        if material is not None:
            UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
    return cube


def _physics_material(stage, path, friction):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(friction)
    api.CreateDynamicFrictionAttr(friction)
    api.CreateRestitutionAttr(0.0)
    return mat


def _rigid(stage, path, mass):
    from pxr import UsdGeom, UsdPhysics

    xf = UsdGeom.Xform.Define(stage, path)
    UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())
    UsdPhysics.MassAPI.Apply(xf.GetPrim()).CreateMassAttr(mass)
    return xf.GetPrim()


def _joint(stage, kind, path, body0, body1, pos0, rot0, axis=None, limits=None):
    from pxr import Gf, UsdPhysics

    joint = {"fixed": UsdPhysics.FixedJoint, "prismatic": UsdPhysics.PrismaticJoint}[kind].Define(stage, path)
    joint.CreateBody0Rel().SetTargets([body0])
    joint.CreateBody1Rel().SetTargets([body1])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*map(float, pos0)))
    w, x, y, z = quat_wxyz(rot0)
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(float(w), float(x), float(y), float(z)))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
    if axis:
        joint.CreateAxisAttr(axis)
        joint.CreateLowerLimitAttr(limits[0])
        joint.CreateUpperLimitAttr(limits[1])
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(FINGER_FORCE_N / 0.004)  # full force at 4 mm error
        drive.CreateDampingAttr(60.0)
        drive.CreateMaxForceAttr(FINGER_FORCE_N)
        drive.CreateTargetPositionAttr(0.0)
    return joint


def build_tool(stage, chain):
    """Extension + gripper body fixed to J6_link, two prismatic fingers; placed at the arm's zero pose."""
    rot6, pos6 = chain.forward(chain.joint_values(np.zeros(6), 0.0))["link_6"]
    tool_rot = rot6 @ TOOL_ROT
    tool_pos = pos6 + rot6 @ np.array(TOOL_POS)
    finger_mat = _physics_material(stage, f"{ROBOT}/tool_looks/finger", 1.5)
    tool = _rigid(stage, f"{ROBOT}/tool", 0.40)
    _set_pose(tool, tool_pos, tool_rot)
    from pxr import Gf, UsdGeom, UsdPhysics

    ext = UsdGeom.Cylinder.Define(stage, f"{ROBOT}/tool/extension")
    ext.GetRadiusAttr().Set(0.020)
    ext.GetHeightAttr().Set(0.100)
    ext.GetAxisAttr().Set("Z")
    UsdGeom.Xformable(ext).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.050))
    ext.CreateDisplayColorAttr([Gf.Vec3f(0.62, 0.64, 0.66)])
    UsdPhysics.CollisionAPI.Apply(ext.GetPrim())
    _box(stage, f"{ROBOT}/tool/body", (0.030, 0.040, 0.058), (0, 0, 0.129), (0.72, 0.75, 0.78))
    _joint(stage, "fixed", f"{ROBOT}/tool_joints/flange", f"{ROBOT}/J6_link", f"{ROBOT}/tool", TOOL_POS, TOOL_ROT)
    for side, sign in (("l", 1.0), ("r", -1.0)):
        path = f"{ROBOT}/finger_{side}"
        finger = _rigid(stage, path, 0.02)
        local = np.array([0.0, sign * (CLOSED_HALF_GAP_M + 0.003), 0.158])
        _set_pose(finger, tool_pos + tool_rot @ local, tool_rot)
        _box(stage, f"{path}/pad", (0.008, 0.006, 0.070), (0, 0, 0.035), (0.35, 0.37, 0.40), material=finger_mat)
        limits = (0.0, FINGER_TRAVEL_M) if sign > 0 else (-FINGER_TRAVEL_M, 0.0)
        _joint(stage, "prismatic", f"{ROBOT}/tool_joints/finger_{side}", f"{ROBOT}/tool", path, local, np.eye(3), "Y", limits)


def build_scene(stage, task, layout, scan):
    from pxr import Gf, UsdGeom, UsdLux

    # same lights as the twin stage (build_usd.py); the scan itself is emissive and ignores them
    UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(900.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/sun")
    sun.CreateIntensityAttr(2500.0)
    sun.CreateAngleAttr(1.5)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-50, 0, 35))

    table_mat = _physics_material(stage, "/World/looks/table", 0.8)
    _box(stage, "/World/table", (0.9, 1.3, 0.03), (0.2, 0.0, -0.015), (0.85, 0.85, 0.85), material=table_mat)
    if scan:
        from align_scan import _add_scan

        _add_scan(stage, scan, lit=False)
        UsdGeom.Imageable(stage.GetPrimAtPath("/World/table")).MakeInvisible()
    for name, pose in layout.items():
        prim = UsdGeom.Xform.Define(stage, f"/World/objects/{name}")
        prim.GetPrim().GetReferences().AddReference(task["assets"][name]["usd"])
        yaw = pose["yaw"]
        rot = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        _set_pose(prim.GetPrim(), [pose["xy"][0], pose["xy"][1], 0.001], rot)


def build_operator_view(stage, eye, target):
    """Window camera where the teleoperator stands at the real cell. The telegrip mapping (telegrip_processor.py) sends
    controller-forward to +Y and controller-right to +X, i.e. the operator stands on the robot's -Y side facing +Y; this
    view puts the screen in that frame so pushing forward moves the arm away on screen and moving right moves it right."""
    from pxr import Gf, UsdGeom

    from record_overhead import _look_matrix, _set_transform

    cam = UsdGeom.Camera.Define(stage, "/World/OperatorCamera")
    _set_transform(cam.GetPrim(), _look_matrix(eye, target, (0.0, 0.0, 1.0)))
    cam.GetFocalLengthAttr().Set(18.0)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 20.0))
    try:
        from omni.kit.viewport.utility import get_active_viewport

        get_active_viewport().camera_path = "/World/OperatorCamera"
    except Exception as exc:  # noqa: BLE001 - headless or no viewport
        print(f"[sim-rmi] operator view not set on a viewport: {exc}", flush=True)


def build_cameras(stage, out_overhead, out_wrist):
    from pxr import Gf, UsdGeom

    from lens import DistortedView
    from record_overhead import _look_matrix, _set_transform

    views = {}
    over = json.load(open(OVERHEAD_JSON, encoding="utf-8"))
    cam = UsdGeom.Camera.Define(stage, "/World/Video0Camera")
    rows = over["usd"]["matrix"]
    _set_transform(cam.GetPrim(), Gf.Matrix4d(rows))
    views["overhead"] = (cam, DistortedView(over["K"], over["dist"], (over["image_hw"][1], over["image_hw"][0]), out_overhead))
    wrist = json.load(open(WRIST_JSON, encoding="utf-8"))
    wcam = UsdGeom.Camera.Define(stage, f"{ROBOT}/tool/WristCamera")
    _set_transform(wcam.GetPrim(), _look_matrix(wrist["eye_tool"], wrist["target_tool"], wrist["up_tool"]))
    views["wrist"] = (wcam, DistortedView(wrist["K"], wrist["dist"], (wrist["image_hw"][1], wrist["image_hw"][0]), out_wrist))
    for camera, view in views.values():
        lens = view.usd_lens()
        camera.GetFocalLengthAttr().Set(float(lens["focal_mm"]))
        camera.GetHorizontalApertureAttr().Set(float(lens["horizontal_aperture"]))
        camera.GetVerticalApertureAttr().Set(float(lens["vertical_aperture"]))
        camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10.0))
    return views


# ------------------------------------------------------------------ main loop

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=16101, help="RMI FRC_Connect port (the real controller uses 16001)")
    ap.add_argument("--zmq-port", type=int, default=5580)
    ap.add_argument("--overhead-wh", type=int, nargs=2, default=(960, 540))
    ap.add_argument("--wrist-wh", type=int, nargs=2, default=(640, 360))
    ap.add_argument("--scan", default="", help="aligned table scan (align_scan.py table.npz) as the table's look")
    ap.add_argument("--seed", type=int, default=0, help="object layout (task_scene.json)")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--view-eye", type=float, nargs=3, default=(0.30, -1.05, 0.75),
                    help="window camera position (table frame, m): where the operator stands; default -Y side")
    ap.add_argument("--view-target", type=float, nargs=3, default=(0.30, 0.0, 0.05), help="window camera look-at point")
    ap.add_argument("--duration", type=float, default=0.0, help="stop after N seconds of sim time (0 = run until closed)")
    a = ap.parse_args(argv)

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": a.headless, "width": 1280, "height": 720})
    try:
        import cv2
        import omni.replicator.core as rep
        import omni.usd
        import zmq
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.utils.types import ArticulationAction

        sys.path.insert(0, os.path.join(EXAMPLES, ".."))
        from cosmos_edge_fanuc.sim_teacher import load_task, sample_layout
        from usd_chain import Chain

        if not os.path.isfile(ROBOT_USD):
            raise FileNotFoundError(f"{ROBOT_USD}: run twin/import_isaac_lrmate.py once (it downloads the asset)")
        world = World(stage_units_in_meters=1.0, physics_dt=1.0 / PHYSICS_HZ, rendering_dt=1.0 / RENDER_HZ)
        stage = omni.usd.get_context().get_stage()
        chain = Chain.load(CHAIN)
        add_reference_to_stage(ROBOT_USD, ROBOT)
        build_tool(stage, chain)
        task = load_task()
        layout_rng = np.random.default_rng(a.seed)  # every SIM_Reset draws the next layout: reproducible per --seed
        layout = sample_layout(task, layout_rng)
        build_scene(stage, task, layout, a.scan)
        views = build_cameras(stage, tuple(a.overhead_wh), tuple(a.wrist_wh))
        build_operator_view(stage, a.view_eye, a.view_target)
        robot = world.scene.add(SingleArticulation(ROBOT, name="fanuc"))
        from isaacsim.core.prims import SingleXFormPrim

        from isaacsim.core.prims import SingleRigidPrim

        object_prims = {name: SingleXFormPrim(f"/World/objects/{name}") for name in layout}
        rigid = {name: world.scene.add(SingleRigidPrim(f"/World/objects/{name}", name=f"obj_{name}")) for name in layout}
        world.reset()
        names = robot.dof_names
        arm_idx = np.array([names.index(j) for j in ARM_JOINTS])
        finger_idx = np.array([names.index("finger_l"), names.index("finger_r")])
        ready = np.array([0.0, 0.4693, 0.1612, 0.0, -1.2627, 0.0])
        q0 = np.zeros(len(names))
        q0[arm_idx] = ready
        q0[finger_idx] = [FINGER_TRAVEL_M, -FINGER_TRAVEL_M]
        robot.set_joint_positions(q0)
        robot.apply_action(ArticulationAction(joint_positions=q0))

        annotators = {}
        for name, (camera, view) in views.items():
            product = rep.create.render_product(str(camera.GetPath()), view.render_wh)
            ann = rep.AnnotatorRegistry.get_annotator("rgb")
            ann.attach([product])
            annotators[name] = ann

        ctl = Controller(chain, ready)
        server = RmiServer(ctl, a.host, a.port)
        server.start()
        pub = zmq.Context().socket(zmq.PUB)
        pub.bind(f"tcp://{a.host}:{a.zmq_port}")
        print(f"[sim-rmi] RMI on {a.host}:{server.port} (session {server.session_port}), cameras on zmq {a.host}:{a.zmq_port}; "
              f"task objects {sorted(layout)}", flush=True)

        def reset_scene():
            new = sample_layout(task, layout_rng)
            for name, pose in new.items():
                yaw = pose["yaw"]
                rigid[name].set_world_pose(position=np.array([pose["xy"][0], pose["xy"][1], 0.001]),
                                           orientation=np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]))
                rigid[name].set_linear_velocity(np.zeros(3))
                rigid[name].set_angular_velocity(np.zeros(3))
            robot.set_joint_positions(q0)
            robot.set_joint_velocities(np.zeros(len(names)))
            robot.apply_action(ArticulationAction(joint_positions=q0))
            ctl.reset(ready)
            for _ in range(PHYSICS_HZ // 10):  # settle 0.1 s
                world.step(render=False)
            return {n: {"xy": [round(v, 4) for v in p["xy"]], "yaw": round(p["yaw"], 4)} for n, p in new.items()}

        substeps = PHYSICS_HZ // RENDER_HZ
        sim_t, frame, episode = 0.0, 0, 0
        wall0 = time.perf_counter()
        while app.is_running():
            if server.reset_request is not None:
                server.reset_request = None
                episode += 1
                new_layout = reset_scene()
                ctl.objects = {n: [float(v) for v in prim.get_world_pose()[0]] for n, prim in object_prims.items()}
                server.send({"Command": "SIM_Reset", "ErrorID": 0, "Episode": episode, "Layout": new_layout})
                print(f"[sim-rmi] reset #{episode}: {new_layout}", flush=True)
            for k in range(substeps):
                q_target = ctl.step(1.0 / PHYSICS_HZ)
                fingers = [0.0, 0.0] if ctl.gripper_closed else [FINGER_TRAVEL_M, -FINGER_TRAVEL_M]
                robot.apply_action(ArticulationAction(joint_positions=np.r_[q_target, fingers],
                                                      joint_indices=np.r_[arm_idx, finger_idx]))
                world.step(render=(k == substeps - 1))
                q = robot.get_joint_positions()
                ctl.measured(q[arm_idx], float(q[finger_idx[0]]))
                while ctl.acks:
                    server.send({"Instruction": "FRC_LinearMotion", "SequenceID": ctl.acks.pop(0), "ErrorID": 0})
                sim_t += 1.0 / PHYSICS_HZ
            ctl.objects = {n: [float(v) for v in prim.get_world_pose()[0]] for n, prim in object_prims.items()}
            images, stamp = {}, time.time()
            for name, ann in annotators.items():
                data = np.asarray(ann.get_data())
                if data.ndim != 3 or data.shape[1::-1] != views[name][1].render_wh:
                    continue
                rgb = views[name][1].apply(np.ascontiguousarray(data[:, :, :3], dtype=np.uint8))
                # lerobot's zmq camera hands cv2.imdecode's array on as-is, so encode the RGB array as-is to get RGB back
                ok, jpg = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ok:
                    images[name] = base64.b64encode(jpg.tobytes()).decode()
            if images:
                pub.send_string(json.dumps({"timestamps": {n: stamp for n in images}, "images": images}))
            frame += 1
            if frame % (RENDER_HZ * 5) == 0:
                rt = sim_t / (time.perf_counter() - wall0)
                print(f"[sim-rmi] t={sim_t:.1f}s real-time x{rt:.2f} queue {len(ctl.queue)} gripper "
                      f"{'closed' if ctl.gripper_closed else 'open'}", flush=True)
            ahead = sim_t - (time.perf_counter() - wall0)
            if ahead > 0:
                time.sleep(ahead)
            if a.duration and sim_t >= a.duration:
                break
        server.stop.set()
    except Exception:  # SimulationApp.close() ends the process; print the error before it does
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

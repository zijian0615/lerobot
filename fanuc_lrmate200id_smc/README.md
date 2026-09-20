# FANUC LR Mate 200iD + SMC-style gripper — MuJoCo model

```
fanuc_lrmate200id_smc.xml   robot: arm + gripper + actuators + keyframes (home, ready)
scene.xml                   robot + floor + lights (load this one)
assets/                     LR Mate 200iD meshes (STL, metres)
preview.png                 home / ready / gripper close-up renders
```

![preview](preview.png)

## Run

macOS needs `mjpython` for the GUI viewer, and uv's standalone Python lacks the libpython it needs, so use the
helper env (one-off):

```bash
cd fanuc_lrmate200id_smc
bash twin/setup_env.sh                                       # creates .venv-twin
.venv-twin/bin/mjpython twin/twin.py --source demo           # opens the viewer with synthetic motion
```

From Python:

```python
import mujoco
m = mujoco.MjModel.from_xml_path("fanuc_lrmate200id_smc/scene.xml")
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key("ready").id)   # "home" or "ready"
d.ctrl[:6] = [0, 0.62, -0.2, 0, -0.7508, 0]             # joint_1..6 target angles [rad]
d.ctrl[6] = 0.0                                         # gripper: 0 = closed, 0.010 = open (m per finger)
mujoco.mj_step(m, d)
print(d.site("tcp").xpos)                               # tool centre point in the world frame
```

## What is in the model
* **Arm** — kinematics, joint limits and meshes from the ROS-Industrial `fanuc_lrmate200id_support`
  package (BSD, `urdf/lrmate200id_macro.xacro`). Zero pose = upper arm up, forearm along +X.
  Limits: J1 ±170°, J2 −100…145°, J3 −70…205°, J4 ±190°, J5 ±125°, J6 ±360°.
* **Actuators** — 6 position servos (`joint_1..6`) + 1 gripper position actuator (`gripper`).
  Bodies use gravity compensation, so joints hold their pose like a real FANUC servo.
* **Frames** — `flange`, `tool0` (ROS-I convention: +Z out of the flange) and `tcp` (95 mm along tool Z) sites.
* **Gripper** — see below; two fingers move symmetrically along tool Y (`finger_l == finger_r`).

## Verified
Model compiles (MuJoCo 3.13); flange/TCP positions match the URDF chain; no self-contact at `home`/`ready`;
all joints stable when driven to 90 % of their limits (the only failures are real collisions: gripper into the floor,
folded elbow); the gripper closes on a 16 mm cube (about 18 N per finger, 1 mm penetration), lifts it 0.23 m and holds it.

## Replace / calibrate (not real data)
* **Gripper geometry is a placeholder.** No SMC CAD or model number was available. The adapter, body, fingers,
  stroke (10 mm per side) and TCP offset are in the `GRIPPER PARAMETERS` comment in `fanuc_lrmate200id_smc.xml`.
  For the real one, replace the boxes with meshes and set the finger `range`, the `gripper` `ctrlrange`,
  the TCP `pos` and the finger `pos` accordingly.
* **Link masses/inertias are estimates** (URDF has none) and the servo gains (`kp`, `kv`) are generic.
  The gripper force is set by `kp` × finger travel error; tune it to your model's grip force.
* Collision uses the convex hull of each link mesh (MuJoCo default), so it is slightly conservative.

---

# Digital twin (`twin/`)

Real-time visualisation of the real robot's joints in the MuJoCo model. **Read-only**: the twin never sends a
motion instruction and never sends `FRC_Abort`. Gripper opening is not read from the robot (shown at `--gripper-mm`).

```bash
# 1. no robot: synthetic motion
.venv-twin/bin/mjpython twin/twin.py --source demo
# 2. real robot over FANUC RMI (same host/port as fanuc_replay_live.py)
.venv-twin/bin/mjpython twin/twin.py --source rmi --host 172.30.109.22
# 3. try the RMI path without a robot
.venv-twin/bin/python twin/mock_rmi_server.py --port 16001 &      # then: --source rmi --host 127.0.0.1
# 4. fed by another process that owns the RMI session
.venv-twin/bin/mjpython twin/twin.py --source udp --udp-port 5005
```

The viewer shows the arm, a blue trail of the TCP (`--trail N`) and a status light above the robot
(green = live, orange = data older than 0.5 s, red = no data). The terminal prints source status, update rate,
data age, J1–J6 in degrees, flange position, and a warning if a joint is outside its limit.

## Before trusting it on the real robot (two things I could NOT verify — no access to the controller)
1. **J2/J3 convention.** FANUC reports J3 coupled to J2; ROS-Industrial's URDF (which this model uses) needs
   `J3_urdf = J3 + J2`. That is the default (`--j3-mode coupled`). Check once: put the robot at a pose with J2 ≈ 30–40°
   and run
   ```bash
   .venv-twin/bin/python twin/twin.py --source rmi --host 172.30.109.22 --check-cartesian
   ```
   It compares the controller's X/Y/Z with the model's forward kinematics for both conventions and tells you which
   `--j3-mode` matches (assumes UTool 0 / UFrame 0 are active). Also confirm visually that moving only J2 on the pendant
   matches the sim. Sign/offset differences on other joints would show up the same way.
2. **RMI session sharing.** RMI most likely allows one client at a time, so the twin (`--source rmi`) may not be able to
   connect while `fanuc_replay_live.py` is running, and `FRC_Initialize` from the twin could disturb a running session.
   Test it with the robot idle first (`--no-init` skips `FRC_Initialize`). To watch a replay/teleop run, let *that*
   process poll the joints on its own socket and forward them to the twin over UDP (`--source udp`), e.g. in
   `fanuc_replay_live.py`:
   ```python
   sys.path.insert(0, "/Users/zhangzijian/lerobot/fanuc_lrmate200id_smc/twin")
   from sources import JointStatePublisher, parse_joint_response
   pub = JointStatePublisher("127.0.0.1", 5005)
   # in AsyncStreamingSender._listen_acks, right after `resp = self.ls.read_json()`:
   if resp.get("Command") == "FRC_ReadJointAngles":
       pub.publish(parse_joint_response(resp)); continue
   # in main(), every ~30 ms: sender.ls.sendall(b'{"Command":"FRC_ReadJointAngles","Group":1}\r\n')
   ```
   The `first RMI joint response:` line printed by the twin shows the raw reply so you can confirm the field names
   (`JointAngle` / `J1..J6`); the parser also accepts a few variants.

## Files and tests
`twin.py` (viewer/CLI) · `sources.py` (RMI / UDP / demo sources, publisher) · `joint_map.py` (FANUC ⇄ model joints,
flange FK) · `mock_rmi_server.py` (fake controller) · `test_twin.py`.

```bash
.venv-twin/bin/pytest twin/test_twin.py -q      # 13 tests, ~9 s
```
The RMI tests run against the mock controller, so they check the pipeline and our protocol assumptions
(message flow, tracking, reconnect, single-session refusal, never sending motion/abort, convention detection),
not FANUC's actual behaviour.

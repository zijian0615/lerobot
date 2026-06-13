import logging
import math
import struct
import threading
import time
from enum import IntEnum
from typing import Dict, Optional, Tuple

import numpy as np


# ── 6D 旋转表示工具 ─────────────────────────────────────────────────────── #
# 使用旋转矩阵前两列（6个标量），完全连续无奇异，适合模仿学习。
# 键名 r0..r2 = 第一列，r3..r5 = 第二列。
# 参考：Zhou et al., "On the Continuity of Rotation Representations in
#       Neural Networks", CVPR 2019.

_ROT6D_KEYS = ("r0", "r1", "r2", "r3", "r4", "r5")


def _aa_to_rot6d(rx: float, ry: float, rz: float) -> Tuple[float, ...]:
    """轴角向量 [rx,ry,rz]（弧度）→ 6D 旋转（旋转矩阵前两列，列优先）。"""
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)
    if angle < 1e-9:
        # 单位矩阵前两列
        return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    ax, ay, az = rx / angle, ry / angle, rz / angle
    # Rodrigues 公式 → 旋转矩阵
    R = [
        [t*ax*ax + c,     t*ax*ay - s*az, t*ax*az + s*ay],
        [t*ax*ay + s*az,  t*ay*ay + c,    t*ay*az - s*ax],
        [t*ax*az - s*ay,  t*ay*az + s*ax, t*az*az + c   ],
    ]
    # 前两列（列优先存储）
    return (R[0][0], R[1][0], R[2][0], R[0][1], R[1][1], R[2][1])


def _rot6d_to_aa(r0: float, r1: float, r2: float,
                 r3: float, r4: float, r5: float) -> Tuple[float, float, float]:
    """6D 旋转 → 轴角向量 [rx,ry,rz]（弧度）。用 Gram-Schmidt 重建旋转矩阵。"""
    # 第一列：直接归一化
    c0 = np.array([r0, r1, r2], dtype=float)
    n0 = np.linalg.norm(c0)
    c0 = c0 / n0 if n0 > 1e-9 else np.array([1.0, 0.0, 0.0])
    # 第二列：减去投影后归一化
    c1 = np.array([r3, r4, r5], dtype=float)
    c1 = c1 - np.dot(c1, c0) * c0
    n1 = np.linalg.norm(c1)
    c1 = c1 / n1 if n1 > 1e-9 else np.array([0.0, 1.0, 0.0])
    # 第三列：叉积
    c2 = np.cross(c0, c1)
    # 旋转矩阵
    R = np.column_stack([c0, c1, c2])
    # 矩阵 → 四元数 → 轴角
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    if qw < 0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw
    qw = max(-1.0, min(1.0, qw))
    angle = 2.0 * math.acos(qw)
    sin_half = math.sqrt(max(0.0, 1.0 - qw * qw))
    if sin_half < 1e-9:
        return 0.0, 0.0, 0.0
    f = angle / sin_half
    return float(qx * f), float(qy * f), float(qz * f)

from lerobot.cameras.utils import make_cameras_from_configs

from ..robot import Robot
from .config_xarm import XArmConfig

logger = logging.getLogger(__name__)


class GripperType(IntEnum):
    NoGripper = 0
    xArmGripper = 1
    xArmGripperG2 = 2
    BioGripperG2 = 3
    PikaGripper = 10
    RobotiqGripper = 11


class XArmRobot(Robot):
    """
    xArm 机器人接口，实现 SO-Follower Robot 协议。

    支持两种运动模式：
        mode=1 (cartesian offline): action/observation 为笛卡尔位姿 [x,y,z,rx,ry,rz] (radians)，
        mode=6 (joint servo)    : action/observation 均为关节角度，键名 j0…j(dof-1) + j7(gripper)
        mode=7 (cartesian online): action/observation 为笛卡尔位姿 [x,y,z,rx,ry,rz] (radians)，
                                   键名 j0(x)…j5(rz) + j7(gripper)

    夹爪归一化值约定：0.0 = 完全张开，1.0 = 完全闭合
    """

    config_class = XArmConfig
    name = "xarm"

    def __init__(self, config: XArmConfig) -> None:
        super().__init__(config)
        self.config: XArmConfig = config

        self._dof = config.robot_dof
        self._mode = config.robot_mode
        self._gripper_type = GripperType(config.gripper_type)
        self._cmd_cnt = 0
        # mode=1 伺服重稳定标志：置 True 时，下一次 send_action 会先以当前真实
        # 位姿"原地踏步"若干帧，再恢复目标跟踪。用于 move_to_home 后或离合接合
        # 时，避免伺服高增益控制器首帧猛追导致机械臂飘逸。
        self._servo_restab = True

        self._joint_speed = math.radians(config.robot_speed)
        self._joint_acc = math.radians(config.robot_acc)

        self._gripper_param = self._build_gripper_param()

        self._connected = False
        self.real_arm = None
        self.cameras: dict = {}

        # 保存上一次关节 / 笛卡尔观测（线程安全近似）
        self._latest_joints: Optional[np.ndarray] = None
        self._latest_pose: Optional[np.ndarray] = None  # [x,y,z,rx,ry,rz] in radians
        self._latest_t: Optional[float] = None
        self._latest_gripper_norm: float = 0.0

        # 相机异步缓存：后台线程持续采集，主循环只读最新帧
        self._cam_frames: dict = {}        # cam_name -> latest np.ndarray
        self._cam_lock = threading.Lock()
        self._cam_threads: list = []
        self._cam_stop = threading.Event()

    # ------------------------------------------------------------------ #
    #  夹爪参数构建                                                          #
    # ------------------------------------------------------------------ #

    def _build_gripper_param(self) -> Dict:
        """返回夹爪参数字典 {open_pos, close_pos, speed, force}"""
        gt = self._gripper_type
        cfg = self.config

        if gt == GripperType.xArmGripper:
            speed = 5000 if cfg.gripper_speed < 0 else max(50, min(cfg.gripper_speed, 5000))
            force = 50 if cfg.gripper_force < 0 else cfg.gripper_force
            return dict(name="xArmGripper", open_pos=800, close_pos=0, speed=speed, force=force)

        elif gt == GripperType.xArmGripperG2:
            spd_deg = 225 if cfg.gripper_speed < 0 else max(15, min(cfg.gripper_speed, 225))
            speed = int(((spd_deg * 60) / 9.88235 + 140) / 0.4)
            force = 50 if cfg.gripper_force < 0 else max(1, min(cfg.gripper_force, 100))
            return dict(name="xArmGripperG2", open_pos=84, close_pos=0, speed=speed, force=force)

        elif gt == GripperType.BioGripperG2:
            speed = 2000 if cfg.gripper_speed < 0 else max(500, min(cfg.gripper_speed, 4500))
            force = 100 if cfg.gripper_force < 0 else max(1, min(cfg.gripper_force, 100))
            return dict(name="BioGripperG2", open_pos=150, close_pos=71, speed=speed, force=force)

        elif gt == GripperType.PikaGripper:
            speed = 0 if cfg.gripper_speed < 0 else cfg.gripper_speed
            force = 0 if cfg.gripper_force < 0 else cfg.gripper_force
            return dict(name="PikaGripper", open_pos=100, close_pos=0, speed=speed, force=force)

        elif gt == GripperType.RobotiqGripper:
            speed = 255 if cfg.gripper_speed < 0 else max(1, min(cfg.gripper_speed, 255))
            force = 255 if cfg.gripper_force < 0 else max(1, min(cfg.gripper_force, 255))
            return dict(name="RobotiqGripper", open_pos=0, close_pos=0xFF, speed=speed, force=force)

        else:
            return dict(name="NoGripper", open_pos=0, close_pos=0, speed=0, force=0)

    def _gripper_norm_to_pos(self, gripper_norm: float) -> int:
        open_p = self._gripper_param["open_pos"]
        close_p = self._gripper_param["close_pos"]
        pos = open_p + gripper_norm * (close_p - open_p)
        lo, hi = min(open_p, close_p), max(open_p, close_p)
        return int(min(max(lo, pos), hi))

    def _gripper_pos_to_norm(self, pos: Optional[int]) -> float:
        if pos is None:
            return self._latest_gripper_norm
        open_p = self._gripper_param["open_pos"]
        close_p = self._gripper_param["close_pos"]
        denom = open_p - close_p
        if denom == 0:
            return 0.0
        norm = (open_p - pos) / denom
        return float(np.clip(norm, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    #  连接 / 断开                                                          #
    # ------------------------------------------------------------------ #

    def connect(self) -> None:
        if self._connected:
            logger.warning("XArmRobot: already connected, skipping.")
            return

        try:
            from xarm.wrapper import XArmAPI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "xArm Python SDK not found. Install via: pip install xArm-Python-SDK"
            ) from exc

        logger.info("Connecting to xArm at %s …", self.config.robot_ip)
        self.real_arm = XArmAPI(self.config.robot_ip)
        time.sleep(0.2)

        if not self.real_arm.connected:
            raise ConnectionError(
                f"xArm connection failed. Check hardware at IP: {self.config.robot_ip}"
            )

        # 初始化 Pika 夹爪（如需）
        if self._gripper_type == GripperType.PikaGripper:
            from ufactory_devices.pika import PikaDevice  # type: ignore

            self._pika_device = PikaDevice(2, pika_gripper_port=self.config.gripper_port)
            self._pika_gripper = self._pika_device.pika_gripper

        # 初始使能序列（与 uf_robot.py 保持一致）
        self._clear_and_enable()

        if self.config.move_to_start_on_connect:
            logger.info("XArmRobot: moving to start_joints …")
            code = self.real_arm.set_servo_angle(
                angle=list(self.config.start_joints),
                is_radian=True,
                wait=True,
            )
            if code != 0:
                # 运动中再次触发错误（常见于 error 19），清除后继续
                logger.warning(
                    "Move to start_joints failed (code=%d), clearing error and retrying once.", code
                )
                self._clear_and_enable()
                code = self.real_arm.set_servo_angle(
                    angle=list(self.config.start_joints),
                    is_radian=True,
                    wait=True,
                )
                if code != 0:
                    logger.warning(
                        "Move to start_joints still failed (code=%d), skipping.", code
                    )
                    self._clear_and_enable()
        else:
            logger.info("XArmRobot: move_to_start_on_connect=False, skipping initial joint movement.")

        # enable=False：不重复调用 clean_error/motion_enable，只切目标模式并初始化夹爪
        self._robot_init(enable=False, init_gripper=True)
        self.real_arm.set_linear_spd_limit_factor(2.0)
        self.real_arm.set_collision_sensitivity(0)

        # SDK 的 error_code 是异步更新的，等待缓存值刷新为 0
        for _ in range(20):
            if self.real_arm.error_code == 0:
                break
            time.sleep(0.1)
        if self.real_arm.error_code != 0:
            logger.warning(
                "xArm error_code=%d still present after connect, "
                "will attempt auto-recovery in send_action.",
                self.real_arm.error_code,
            )

        self._connected = True

        # 初始化相机并启动异步采集线程
        if self.config.cameras:
            self.cameras = make_cameras_from_configs(self.config.cameras)
            self._cam_stop.clear()
            for cam_name, cam in self.cameras.items():
                cam.connect()
                t = threading.Thread(
                    target=self._cam_capture_loop,
                    args=(cam_name, cam),
                    daemon=True,
                    name=f"cam-{cam_name}",
                )
                t.start()
                self._cam_threads.append(t)
                logger.info("XArmRobot: camera '%s' async thread started.", cam_name)

        logger.info("XArmRobot: connected (DOF=%d, mode=%d).", self._dof, self._mode)

    def _cam_capture_loop(self, cam_name: str, cam) -> None:
        """后台线程：持续读取相机帧并写入缓存，主循环零等待取最新帧。"""
        while not self._cam_stop.is_set():
            try:
                frame = cam.read()
                if frame is not None:
                    with self._cam_lock:
                        self._cam_frames[cam_name] = frame
            except Exception as exc:
                logger.debug("Camera '%s' read error: %s", cam_name, exc)

    def _clear_and_enable(self) -> None:
        """清除控制器错误并使能机械臂，切换到 mode=0 就绪状态。"""
        self.real_arm.motion_enable()
        self.real_arm.clean_error()
        self.real_arm.clean_warn()
        self.real_arm.set_mode(0)
        self.real_arm.set_state(0)
        time.sleep(0.3)

    def request_servo_restabilize(self) -> None:
        """请求下一次 send_action 先做 mode=1 伺服重稳定（原地踏步），
        用于离合接合等需要避免伺服首帧猛追的场景。"""
        self._servo_restab = True

    def move_to_home(self, speed_deg_s: float = 30.0) -> None:
        """以低速平滑运动到 start_joints，用于 episode 间归位，避免下集开头跳变。

        内部切换到 mode=0 执行同步运动，完成后恢复 self._mode。
        speed_deg_s: 归位速度（度/秒），建议 20~50，越低越安全。
        """
        if not self._connected or self.real_arm is None:
            return
        logger.info("XArmRobot: moving to home (start_joints) at %.0f deg/s …", speed_deg_s)
        speed_rad = math.radians(speed_deg_s)
        # 切到普通位置模式执行同步运动
        self.real_arm.set_mode(0)
        self.real_arm.set_state(0)
        time.sleep(0.1)
        code = self.real_arm.set_servo_angle(
            angle=list(self.config.start_joints),
            speed=speed_rad,
            mvacc=math.radians(200.0),
            is_radian=True,
            wait=True,
        )
        if code != 0:
            logger.warning("move_to_home: set_servo_angle returned code=%d", code)
        # 恢复目标模式
        self.real_arm.set_mode(self._mode)
        self.real_arm.set_state(0)
        self._cmd_cnt = 0
        self._servo_restab = True   # 让下次 send_action 重走 mode=1 首帧稳定逻辑
        time.sleep(0.1)
        logger.info("XArmRobot: home reached.")

    def disconnect(self) -> None:
        if not self._connected:
            return

        # 停止相机后台线程
        self._cam_stop.set()
        for t in self._cam_threads:
            t.join(timeout=1.0)
        self._cam_threads.clear()

        try:
            if self.real_arm is not None:
                self.real_arm.disconnect()
        except Exception as exc:
            logger.warning("Error during xArm disconnect: %s", exc)

        for cam in self.cameras.values():
            try:
                cam.disconnect()
            except Exception as exc:
                logger.warning("Error disconnecting camera: %s", exc)

        self.cameras = {}
        self._cam_frames = {}
        self._connected = False
        logger.info("XArmRobot: disconnected.")

    def _robot_init(self, enable: bool = True, init_gripper: bool = False) -> None:
        """切换到目标运动模式并初始化夹爪。

        enable=True  : 重新执行 clean_error / clean_warn / motion_enable（用于运行中恢复错误）
        enable=False : 跳过上述步骤，仅切模式+就绪状态（connect() 内首次调用时使用，
                       避免与 connect() 中的初始使能序列重复，防止触发 error code 19）
        """
        if enable:
            self.real_arm.clean_error()
            self.real_arm.clean_warn()
            self.real_arm.motion_enable(True)
        self.real_arm.set_mode(self._mode)
        self.real_arm.set_state(0)
        time.sleep(0.1)

        _, err_warn = self.real_arm.get_err_warn_code()
        if err_warn[0] != 0:
            # error 19 (emergency stop) 等可能在模式切换后短暂残留，尝试再次清除
            logger.warning("xArm error code=%d after mode switch, attempting to clear …", err_warn[0])
            self.real_arm.clean_error()
            self.real_arm.set_state(0)
            time.sleep(0.2)
            _, err_warn = self.real_arm.get_err_warn_code()
            if err_warn[0] != 0:
                raise RuntimeError(
                    f"xArm controller error persists after mode switch, code={err_warn[0]}. "
                    "Check hardware E-stop button or teach pendant."
                )

        if self._gripper_type == GripperType.NoGripper or not init_gripper:
            return

        self.real_arm._arm._baud_checkset = True
        gp = self._gripper_param

        if self._gripper_type == GripperType.xArmGripper:
            self.real_arm.set_gripper_enable(True)
            self.real_arm.set_gripper_mode(0)
            self.real_arm.set_gripper_speed(gp["speed"])
            if init_gripper:
                self.real_arm.set_gripper_position(gp["open_pos"])

        elif self._gripper_type == GripperType.xArmGripperG2:
            self.real_arm.set_gripper_enable(True)
            self.real_arm.set_gripper_mode(0)
            if init_gripper:
                self.real_arm.set_gripper_g2_position(gp["open_pos"])

        elif self._gripper_type == GripperType.BioGripperG2:
            _, mode = self.real_arm.get_bio_gripper_control_mode()
            if mode != 1:
                self.real_arm.set_bio_gripper_control_mode(1)
            self.real_arm.set_bio_gripper_enable(True)
            if init_gripper:
                self.real_arm.open_bio_gripper()

        elif self._gripper_type == GripperType.PikaGripper:
            self._pika_gripper.enable()
            time.sleep(0.5)
            if init_gripper:
                self._pika_gripper.set_gripper_distance(gp["open_pos"])

        elif self._gripper_type == GripperType.RobotiqGripper:
            self.real_arm.robotiq_reset()
            self.real_arm.robotiq_set_activate(wait=True)
            if init_gripper:
                self.real_arm.robotiq_set_position(gp["open_pos"], wait=True)

        self.real_arm._arm._baud_checkset = False

        _, err_warn = self.real_arm.get_err_warn_code()
        if err_warn[0] != 0:
            # 夹爪初始化后仍有错误（可能是 error 19 残留），尝试清除后继续
            logger.warning("xArm error code=%d after gripper init, attempting to clear …", err_warn[0])
            self.real_arm.clean_error()
            self.real_arm.set_state(0)
            time.sleep(0.2)
            _, err_warn = self.real_arm.get_err_warn_code()
            if err_warn[0] != 0:
                raise RuntimeError(
                    f"xArm gripper error persists after init, code={err_warn[0]}. "
                    "Check hardware E-stop button or teach pendant."
                )

        self._latest_gripper_norm = 0.0

    # ------------------------------------------------------------------ #
    #  核心接口：send_action / get_observation                              #
    # ------------------------------------------------------------------ #

    def _action_to_pose(self, action: Dict) -> list:
        """action 字典 → xArm pose [x, y, z, rx, ry, rz]（mm + 弧度）。

        支持两种旋转格式：
          - 6D 旋转（r0..r5）：旋转矩阵前两列，连续无奇异（推荐，模仿学习友好）
          - 轴角（j3..j5）：向后兼容
        """
        x = float(action.get("j0", 0.0))
        y = float(action.get("j1", 0.0))
        z = float(action.get("j2", 0.0))
        if all(k in action for k in _ROT6D_KEYS):
            rx, ry, rz = _rot6d_to_aa(
                float(action["r0"]), float(action["r1"]), float(action["r2"]),
                float(action["r3"]), float(action["r4"]), float(action["r5"]),
            )
        else:
            rx = float(action.get("j3", 0.0))
            ry = float(action.get("j4", 0.0))
            rz = float(action.get("j5", 0.0))
        return [x, y, z, rx, ry, rz]

    def send_action(self, action: Dict) -> Dict:
        """
        发送控制指令。

        action 字典格式（关节模式 mode=6）：
            {"j0": ..., "j1": ..., ..., "j(dof-1)": ..., "j7": gripper_norm}

        action 字典格式（笛卡尔模式 mode=7）：
            {"j0": x, "j1": y, "j2": z, "j3": rx, "j4": ry, "j5": rz, "j7": gripper_norm}
            单位：位置 mm，姿态 radians

        j7 (gripper_norm)：0.0 = 完全张开，1.0 = 完全闭合。可省略。
        """
        if not self._connected or self.real_arm is None:
            raise RuntimeError("XArmRobot is not connected. Call connect() first.")

        if self.real_arm.error_code != 0:
            # 限速自动恢复：每 2 秒尝试一次 clean_error + set_state
            now = time.perf_counter()
            last_recovery = getattr(self, "_last_error_recovery_t", 0.0)
            if now - last_recovery > 2.0:
                logger.warning(
                    "xArm error code=%d, attempting auto-recovery …",
                    self.real_arm.error_code,
                )
                self.real_arm.clean_error()
                self.real_arm.set_state(0)
                self._last_error_recovery_t = now
                time.sleep(0.1)
            if self.real_arm.error_code != 0:
                logger.error("xArm error code=%d persists, skipping action.", self.real_arm.error_code)
                return action

        gripper_norm: Optional[float] = action.get("j7", None)

        if self._mode == 6:
            joints = np.array(
                [float(action[f"j{i}"]) for i in range(self._dof)], dtype=np.float64
            )
            jnt_spd = 0.2 if self._cmd_cnt < 20 else self._joint_speed

            if self._cmd_cnt == 0:
                # 第一条指令用普通模式同步执行，确保机器人先到位再切伺服
                if self.real_arm.mode != 0:
                    self.real_arm.set_mode(0)
                    self.real_arm.set_state(0)
                    time.sleep(0.1)
                self.real_arm.set_servo_angle(
                    angle=joints.tolist(),
                    speed=jnt_spd,
                    mvacc=self._joint_acc,
                    is_radian=True,
                    wait=True,
                )
                self.real_arm.set_mode(6)
                self.real_arm.set_state(0)
                time.sleep(0.1)
            else:
                if self.real_arm.mode != 6:
                    self.real_arm.set_mode(6)
                    self.real_arm.set_state(0)
                    time.sleep(0.1)
                self.real_arm.set_servo_angle(
                    angle=joints.tolist(),
                    speed=jnt_spd,
                    mvacc=self._joint_acc,
                    is_radian=True,
                    wait=False,
                )

        elif self._mode == 7:
            pose = self._action_to_pose(action)
            # 命令队列满时跳过本帧，避免积压导致 FPS 下降和跟手延迟
            cmd_num = getattr(self.real_arm, "cmd_num", 0)
            if cmd_num > 3:
                logger.debug("[xArm mode=7] cmd_num=%d > 3, skip frame to drain queue", cmd_num)
            else:
                self.real_arm.set_position_aa(
                    pose,
                    is_radian=True,
                    speed=self.config.robot_speed,
                    mvacc=self.config.robot_acc,
                    wait=False,
                )
        else:
            # mode=1: Cartesian servo（笛卡尔伺服，每帧直接覆盖目标位置，无轨迹队列）
            pose = self._action_to_pose(action)

            if self.real_arm.mode != 1 or self._servo_restab:
                # 切换到 Cartesian servo 模式（若已在 mode=1 则只做重稳定）
                if self.real_arm.mode != 1:
                    self.real_arm.set_mode(1)
                    self.real_arm.set_state(0)
                    time.sleep(0.1)
                # 进入 servo / 重稳定时，先以当前真实位置"原地踏步"若干帧，
                # 防止 servo 高增益控制器因首帧偏差产生剧烈抖动/飘逸。
                _, cur_pose = self.real_arm.get_position_aa(is_radian=True)
                if cur_pose is not None:
                    for _ in range(5):
                        self.real_arm.set_servo_cartesian_aa(
                            cur_pose[:6],
                            is_radian=True,
                            speed=self.config.robot_speed,
                            mvacc=self.config.robot_acc,
                        )
                        time.sleep(0.01)
                self._servo_restab = False
            self.real_arm.set_servo_cartesian_aa(
                pose,
                is_radian=True,
                speed=self.config.robot_speed,
                mvacc=self.config.robot_acc,
            )

        if self._cmd_cnt < 99999:
            self._cmd_cnt += 1

        self._send_gripper(gripper_norm)
        return action

    def _send_gripper(self, gripper_norm: Optional[float]) -> None:
        if self._gripper_type == GripperType.NoGripper or gripper_norm is None:
            return

        self._latest_gripper_norm = float(gripper_norm)
        grippos = self._gripper_norm_to_pos(gripper_norm)
        gp = self._gripper_param

        if self._gripper_type == GripperType.xArmGripper:
            modbus = [0x08, 0x10, 0x07, 0x00, 0x00, 0x02, 0x04]
            modbus.extend(list(struct.pack(">i", grippos)))
            self.real_arm.getset_tgpio_modbus_data(modbus)

        elif self._gripper_type == GripperType.xArmGripperG2:
            pos_raw = int(
                (math.degrees(math.asin((grippos - 16) / 110)) + 8.33) * 18.28
            )
            modbus = [0x08, 0x10, 0x0C, 0x00, 0x00, 0x05, 0x0A, 0x00, 0x01]
            modbus.extend(list(struct.pack(">h", gp["speed"])))
            modbus.extend(list(struct.pack(">h", gp["force"])))
            modbus.extend(list(struct.pack(">i", pos_raw)))
            self.real_arm.getset_tgpio_modbus_data(modbus)

        elif self._gripper_type == GripperType.BioGripperG2:
            pos_raw = int(grippos * 3.7342 - 265.13)
            modbus = [0x08, 0x10, 0x0C, 0x00, 0x00, 0x05, 0x0A, 0x00, 0x01]
            modbus.extend(list(struct.pack(">h", gp["speed"])))
            modbus.extend(list(struct.pack(">h", gp["force"])))
            modbus.extend(list(struct.pack(">i", pos_raw)))
            self.real_arm.getset_tgpio_modbus_data(modbus)

        elif self._gripper_type == GripperType.PikaGripper:
            self._pika_gripper.set_gripper_distance(grippos)

        elif self._gripper_type == GripperType.RobotiqGripper:
            modbus = [
                0x09, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06,
                0x09, 0x00, 0x00,
                grippos, gp["speed"], gp["force"],
            ]
            self.real_arm.getset_tgpio_modbus_data(modbus)

    def get_observation(self) -> Dict:
        """
        获取机器人当前状态。

        返回字典（关节模式 mode=6）：
            {"j0": ..., ..., "j(dof-1)": ..., "j7": gripper_norm, "timestamp": ...}

        返回字典（笛卡尔模式 mode=7）：
            {"j0": x, "j1": y, "j2": z, "j3": rx, "j4": ry, "j5": rz,
             "j7": gripper_norm, "timestamp": ...}
        """
        if not self._connected or self.real_arm is None:
            raise RuntimeError("XArmRobot is not connected. Call connect() first.")

        obs: Dict = {}

        if self._mode == 6:
            code, joints = self.real_arm.get_servo_angle(is_radian=True)
            if code != 0 or joints is None:
                logger.warning("get_servo_angle returned code=%d", code)
                joints = [0.0] * self._dof
            joints = list(joints[: self._dof])
            for i, v in enumerate(joints):
                obs[f"j{i}"] = float(v)
        else:
            code, pose = self.real_arm.get_position_aa(is_radian=True)
            if code != 0 or pose is None:
                logger.warning("get_position_aa returned code=%d", code)
                pose = [0.0] * 6
            # 位置：j0, j1, j2（mm）
            for i in range(3):
                obs[f"j{i}"] = float(pose[i])
            # 姿态：6D 旋转表示（r0..r5），替代轴角 j3..j5 避免跳变
            rot6d = _aa_to_rot6d(float(pose[3]), float(pose[4]), float(pose[5]))
            for key, val in zip(_ROT6D_KEYS, rot6d):
                obs[key] = val

        # 夹爪状态：二值化（与 Quest grip 按钮保持一致）
        gripper_norm = self._read_gripper_norm()
        obs["j7"] = 1.0 if gripper_norm >= 0.5 else 0.0
        obs["timestamp"] = time.perf_counter()

        # 相机：从后台线程缓存取最新帧，不阻塞控制循环
        with self._cam_lock:
            for cam_name, frame in self._cam_frames.items():
                obs[cam_name] = frame

        return obs

    def _read_gripper_norm(self) -> float:
        """读取夹爪归一化开合度，失败时返回上一次记录值。"""
        try:
            if self._gripper_type == GripperType.xArmGripper:
                code, pos = self.real_arm.get_gripper_position()
                if code == 0 and pos is not None:
                    return self._gripper_pos_to_norm(int(pos))

            elif self._gripper_type == GripperType.xArmGripperG2:
                code, pos = self.real_arm.get_gripper_g2_position()
                if code == 0 and pos is not None:
                    return self._gripper_pos_to_norm(int(pos))

            elif self._gripper_type == GripperType.BioGripperG2:
                # Bio 夹爪无直接位置查询，返回上次缓存值
                pass

            elif self._gripper_type == GripperType.PikaGripper:
                pos = self._pika_gripper.get_gripper_distance()
                if pos is not None:
                    return self._gripper_pos_to_norm(int(pos))

            elif self._gripper_type == GripperType.RobotiqGripper:
                _, data = self.real_arm.robotiq_get_status()
                if data is not None:
                    pos = data.get("gPO", None)
                    if pos is not None:
                        return self._gripper_pos_to_norm(int(pos))

        except Exception as exc:
            logger.debug("Could not read gripper position: %s", exc)

        return self._latest_gripper_norm

    # ------------------------------------------------------------------ #
    #  特征描述（协议必要属性）                                               #
    # ------------------------------------------------------------------ #

    @property
    def observation_features(self) -> dict:
        if self._mode == 6:
            feats = {f"j{i}": float for i in range(self._dof)}
        else:
            # 笛卡尔模式：位置 j0..j2 + 6D 旋转 r0..r5（无跳变）
            feats = {f"j{i}": float for i in range(3)}
            feats.update({k: float for k in _ROT6D_KEYS})

        feats["j7"] = float  # gripper norm

        for cam_name, camera in self.cameras.items():
            h = camera.height if hasattr(camera, "height") else 480
            w = camera.width if hasattr(camera, "width") else 640
            feats[cam_name] = (h, w, 3)

        return feats

    @property
    def action_features(self) -> dict:
        if self._mode == 6:
            feats = {f"j{i}": float for i in range(self._dof)}
        else:
            # 笛卡尔模式：位置 j0..j2 + 6D 旋转 r0..r5（无跳变）
            feats = {f"j{i}": float for i in range(3)}
            feats.update({k: float for k in _ROT6D_KEYS})

        feats["j7"] = float  # gripper norm

        return feats

    def configure(self) -> None:
        pass

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------ #
    #  上下文管理器支持                                                      #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "XArmRobot":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()

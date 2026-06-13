from dataclasses import dataclass, field
from typing import Tuple

from lerobot.cameras.configs import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("xarm")
@dataclass(kw_only=True)
class XArmConfig(RobotConfig):
    """
    xArm 机器人配置，支持 xArm5/6/7 系列，兼容多种夹爪类型。

    robot_mode:
        1  -> 笛卡尔离线 (default)
        6  -> 关节在线伺服模式（joint online servo）
        7  -> 笛卡尔在线轨迹规划模式（cartesian online trajectory planning）

    gripper_type:
        0  -> 无夹爪
        1  -> xArm Gripper
        2  -> xArm Gripper G2
        3  -> Bio Gripper G2
        10 -> Pika Gripper（需要 gripper_port 参数）
        11 -> Robotiq 2F-85
    """

    # ---- 连接 ----
    robot_ip: str = "192.168.1.127"

    # ---- 自由度 ----
    robot_dof: int = 6  # 5, 6, or 7

    # ---- 运动模式 ----
    robot_mode: int = 1  # 1: cartesian offline, 6: joint online, 7: cartesian online

    # ---- 速度 / 加速度 ----
    # mode=6 时单位为 rad/s（内部由 deg 转换），mode=7 时单位为 mm/s
    robot_speed: float = 90.0   # deg/s (joint) or mm/s (cartesian)
    robot_acc: float = 500.0    # deg/s^2 or mm/s^2

    # ---- 夹爪 ----
    gripper_type: int = 0
    gripper_port: str | None = None   # 仅 Pika Gripper (type=10) 使用
    gripper_speed: int = -1           # -1 表示自动
    gripper_force: int = -1           # -1 表示自动

    # ---- 初始关节角 (radians) ----
    start_joints: Tuple[float, ...] = (-0.16828, -0.44685, -0.89340, -0.15272, 1.34873, -0.03999)

    # ---- 连接时是否自动运动到初始关节角 ----
    # 设为 False 可防止 connect() 时机械臂意外移动；
    # 首次录制建议保持 True 以确保起点一致。
    move_to_start_on_connect: bool = True

    # ---- 可选相机 ----
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

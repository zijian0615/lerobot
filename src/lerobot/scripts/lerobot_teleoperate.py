# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
Simple script to control a robot from teleoperation.

Requires: pip install 'lerobot[hardware]'


Example teleoperation with Meta Quest VR (telegrip) on SO-100:

```shell
lerobot-teleoperate \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58FD0170101 \
    --robot.id=black \
    --teleop.type=telegrip \
    --teleop.urdf_path=.telegrip-ref/URDF/SO100/so100.urdf \
    --teleop.controller_side=right \
    --display_data=true
```

Example teleoperation with phone on xArm (iOS HEBI or Android WebXR):

```shell
lerobot-teleoperate \
    --robot.type=xarm \
    --robot.robot_ip=192.168.1.127 \
    --robot.robot_mode=1 \
    --robot.gripper_type=1 \
    --teleop.type=phone \
    --teleop.phone_os=IOS \
    --display_data=true
```


Example teleoperation with bimanual so100:

```shell
lerobot-teleoperate \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/tty.usbmodem58FD0166521 \
  --robot.right_arm_config.port=/dev/tty.usbmodem58FD0170101 \
  --robot.id=bimanual_follower \
  --robot.left_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
  }' --robot.right_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
  }' \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/tty.usbmodem58FD0166521 \
  --teleop.right_arm_config.port=/dev/tty.usbmodem58FD0170101 \
  --teleop.id=bimanual_leader \
  --display_data=true
```

"""

import logging
import time
from dataclasses import asdict, dataclass
from typing import Any
from pprint import pformat

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.teleoperators.telegrip import make_telegrip_processors
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_rebot_b601_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    rebot_b601_follower,
    so_follower,
    unitree_g1 as unitree_g1_robot,
    xarm,
)
from lerobot.robots.xarm import make_phone_xarm_processors
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_rebot_102_leader,
    bi_so_leader,
    gamepad,
    homunculus,
    keyboard,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    openarm_mini,
    phone,
    reachy2_teleoperator,
    rebot_102_leader,
    so_leader,
    unitree_g1,
    telegrip,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data, shutdown_rerun


@dataclass
class TeleoperateConfig:
    # TODO: pepijn, steven: if more robots require multiple teleoperators (like lekiwi) its good to make this possibele in teleop.py and record.py with List[Teleoperator]
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Display data on a remote Rerun server
    display_ip: str | None = None
    # Port of the remote Rerun server
    display_port: int | None = None
    # Whether to  display compressed images in Rerun
    display_compressed_images: bool = False


_BUS_MAX_RETRIES = 5
_BUS_RETRY_DELAY_S = 0.01
_BUS_SETTLE_AFTER_WRITE_S = 0.003
_BUS_MAX_CONSECUTIVE_FAILURES = 30


def _clear_robot_bus(robot: Robot) -> None:
    if isinstance(robot, so_follower.SOFollower) and robot.is_connected:
        try:
            robot.bus.port_handler.clearPort()
        except Exception:
            pass


def _retry_bus_op(
    description: str,
    op,
    *,
    robot: Robot | None = None,
    max_retries: int = _BUS_MAX_RETRIES,
):
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return op()
        except ConnectionError as error:
            last_error = error
            logging.warning(
                "%s failed (%s/%s): %s",
                description,
                attempt,
                max_retries,
                error,
            )
            if robot is not None:
                _clear_robot_bus(robot)
            if attempt < max_retries:
                time.sleep(_BUS_RETRY_DELAY_S * attempt)
    assert last_error is not None
    raise last_error


def _motor_positions_from_observation(obs: dict[str, Any]) -> dict[str, float]:
    return {
        key.removesuffix(".pos"): float(value)
        for key, value in obs.items()
        if isinstance(key, str) and key.endswith(".pos")
    }


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    display_data: bool = False,
    duration: float | None = None,
    display_compressed_images: bool = False,
):
    """
    This function continuously reads actions from a teleoperation device, processes them through optional
    pipelines, sends them to a robot, and optionally displays the robot's state. The loop runs at a
    specified frequency until a set duration is reached or it is manually interrupted.

    Args:
        teleop: The teleoperator device instance providing control actions.
        robot: The robot instance being controlled.
        fps: The target frequency for the control loop in frames per second.
        display_data: If True, fetches robot observations and displays them in the console and Rerun.
        display_compressed_images: If True, compresses images before sending them to Rerun for display.
        duration: The maximum duration of the teleoperation loop in seconds. If None, the loop runs indefinitely.
        teleop_action_processor: An optional pipeline to process raw actions from the teleoperator.
        robot_action_processor: An optional pipeline to process actions before they are sent to the robot.
        robot_observation_processor: An optional pipeline to process raw observations from the robot.
    """

    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    consecutive_bus_failures = 0
    cached_obs: dict[str, Any] | None = None
    use_cached_telegrip_obs = teleop.name == "telegrip" and not display_data
    if use_cached_telegrip_obs:
        cached_obs = _retry_bus_op(
            "Initial robot observation read",
            robot.get_observation,
            robot=robot,
        )

    while True:
        loop_start = time.perf_counter()

        if cached_obs is not None:
            obs = cached_obs.copy()
        else:
            try:
                obs = _retry_bus_op(
                    "Robot observation read",
                    robot.get_observation,
                    robot=robot,
                )
                consecutive_bus_failures = 0
            except ConnectionError:
                consecutive_bus_failures += 1
                logging.warning(
                    "Skipping teleop frame after bus read failures (%s/%s).",
                    consecutive_bus_failures,
                    _BUS_MAX_CONSECUTIVE_FAILURES,
                )
                if consecutive_bus_failures >= _BUS_MAX_CONSECUTIVE_FAILURES:
                    raise
                time.sleep(0.05)
                continue

        if robot.name == "unitree_g1":
            teleop.send_feedback(obs)

        # Get teleop action
        raw_action = teleop.get_action()

        # Process teleop action through pipeline
        teleop_action = teleop_action_processor((raw_action, obs))

        # Process action for robot through pipeline
        robot_action_to_send = robot_action_processor((teleop_action, obs))
        if not robot_action_to_send:
            dt_s = time.perf_counter() - loop_start
            precise_sleep(max(1 / fps - dt_s, 0.0))
            loop_s = time.perf_counter() - loop_start
            print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
            move_cursor_up(1)

            if duration is not None and time.perf_counter() - start >= duration:
                return
            continue

        # Send processed action to robot (robot_action_processor.to_output should return RobotAction)
        present_pos = _motor_positions_from_observation(obs)
        try:
            if isinstance(robot, so_follower.SOFollower):
                _ = _retry_bus_op(
                    "Robot action write",
                    lambda: robot.send_action(robot_action_to_send, present_pos=present_pos),
                    robot=robot,
                )
            else:
                _ = _retry_bus_op(
                    "Robot action write",
                    lambda: robot.send_action(robot_action_to_send),
                    robot=robot,
                )
        except ConnectionError:
            consecutive_bus_failures += 1
            logging.warning(
                "Skipping teleop frame after bus write failures (%s/%s).",
                consecutive_bus_failures,
                _BUS_MAX_CONSECUTIVE_FAILURES,
            )
            if consecutive_bus_failures >= _BUS_MAX_CONSECUTIVE_FAILURES:
                raise
            time.sleep(0.05)
            continue

        consecutive_bus_failures = 0
        if cached_obs is not None:
            cached_obs.update(robot_action_to_send)
        time.sleep(_BUS_SETTLE_AFTER_WRITE_S)

        if display_data:
            # Process robot observation through pipeline
            obs_transition = robot_observation_processor(obs)

            log_rerun_data(
                observation=obs_transition,
                action=teleop_action,
                compress_images=display_compressed_images,
            )

            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'NORM':>7}")
            # Display the final robot action that was sent
            for motor, value in robot_action_to_send.items():
                print(f"{motor:<{display_len}} | {value:>7.2f}")
            move_cursor_up(len(robot_action_to_send) + 3)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0.0))
        loop_s = time.perf_counter() - loop_start
        print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
        move_cursor_up(1)

        if duration is not None and time.perf_counter() - start >= duration:
            return


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="teleoperation", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)
    if cfg.teleop.type == "telegrip":
        teleop_action_processor, robot_action_processor, robot_observation_processor = make_telegrip_processors(
            robot, cfg.teleop
        )
    elif cfg.teleop.type == "phone" and cfg.robot.type == "xarm":
        teleop_action_processor, robot_action_processor, robot_observation_processor = make_phone_xarm_processors(
            robot, cfg.teleop
        )
    else:
        teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    teleop.connect()
    robot.connect()

    try:
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            display_compressed_images=display_compressed_images,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            shutdown_rerun()
        teleop.disconnect()
        robot.disconnect()


def main():
    register_third_party_plugins()
    teleoperate()


if __name__ == "__main__":
    main()

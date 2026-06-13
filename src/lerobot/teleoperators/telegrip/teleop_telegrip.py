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

from __future__ import annotations

import logging

import numpy as np

from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_telegrip import TelegripConfig
from .https_server import TelegripHTTPServer
from .vr_server import VRWebSocketServer, get_local_ip
from .vr_state import VRInputState

logger = logging.getLogger(__name__)


class Telegrip(Teleoperator):
    """
    VR teleoperator using Meta Quest / WebXR controllers (telegrip protocol).

    Starts an HTTPS web UI and a WebSocket server. Open the printed URL in the
    Quest browser to enter VR teleoperation mode.

    Control mapping (same as telegrip):
    - Hold grip: arm position tracks controller
    - Controller roll/pitch: wrist roll/flex
    - Hold trigger: gripper open; release: gripper closed
    """

    config_class = TelegripConfig
    name = "telegrip"

    def __init__(self, config: TelegripConfig):
        super().__init__(config)
        self.config = config
        self._vr_state = VRInputState()
        self._https_server: TelegripHTTPServer | None = None
        self._vr_server: VRWebSocketServer | None = None

    @property
    def is_connected(self) -> bool:
        return self._vr_server is not None and self._vr_server.is_running

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def action_features(self) -> dict[str, type]:
        features: dict[str, type] = {}
        for side in ("left", "right"):
            features[f"vr.{side}.enabled"] = bool
            features[f"vr.{side}.target_delta"] = np.ndarray
            features[f"vr.{side}.wrist_roll_deg"] = float
            features[f"vr.{side}.wrist_flex_deg"] = float
            features[f"vr.{side}.gripper_closed"] = bool
            features[f"vr.{side}.reset_origin"] = bool
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    def connect(self, calibrate: bool = True) -> None:
        self._connect()

    @check_if_already_connected
    def _connect(self) -> None:
        cert_path = self.calibration_dir / "cert.pem"
        key_path = self.calibration_dir / "key.pem"

        self._vr_server = VRWebSocketServer(
            state=self._vr_state,
            host_ip=self.config.host_ip,
            websocket_port=self.config.websocket_port,
            cert_path=cert_path,
            key_path=key_path,
            vr_to_robot_scale=self.config.vr_to_robot_scale,
        )
        self._https_server = TelegripHTTPServer(
            config=self.config,
            cert_path=cert_path,
            key_path=key_path,
            vr_client_count=lambda: self._vr_server.client_count if self._vr_server else 0,
        )

        self._https_server.start()
        self._vr_server.start()

        host_display = get_local_ip() if self.config.host_ip == "0.0.0.0" else self.config.host_ip
        print("\n🤖 telegrip VR teleoperation ready")
        print(f"📱 Open the UI in your browser:")
        print(f"   https://{host_display}:{self.config.https_port}")
        print(f"📱 Then open the same address on your VR headset browser\n")

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        left, right = self._vr_state.snapshot()
        return {
            "vr.left.enabled": left.enabled,
            "vr.left.target_delta": left.target_delta,
            "vr.left.wrist_roll_deg": left.wrist_roll_deg,
            "vr.left.wrist_flex_deg": left.wrist_flex_deg,
            "vr.left.gripper_closed": left.gripper_closed,
            "vr.left.reset_origin": left.reset_origin,
            "vr.right.enabled": right.enabled,
            "vr.right.target_delta": right.target_delta,
            "vr.right.wrist_roll_deg": right.wrist_roll_deg,
            "vr.right.wrist_flex_deg": right.wrist_flex_deg,
            "vr.right.gripper_closed": right.gripper_closed,
            "vr.right.reset_origin": right.reset_origin,
        }

    def send_feedback(self, feedback: dict) -> None:
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._vr_server:
            self._vr_server.stop()
            self._vr_server = None
        if self._https_server:
            self._https_server.stop()
            self._https_server = None

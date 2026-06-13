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

import http.server
import json
import logging
import socket
import ssl
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from .ssl_utils import ensure_ssl_certificates
from .vr_server import get_local_ip

if TYPE_CHECKING:
    from .config_telegrip import TelegripConfig

logger = logging.getLogger(__name__)

WEB_UI_DIR = Path(__file__).parent / "web_ui"

_CONTENT_TYPES = {
    ".html": "text/html",
    ".css": "text/css",
    ".js": "application/javascript",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".ico": "image/x-icon",
    ".mp4": "video/mp4",
}


class _WebUIHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    @property
    def api(self) -> "TelegripHTTPServer":
        return self.server.telegrip_api  # type: ignore[attr-defined]

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        try:
            super().end_headers()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, ssl.SSLError):
            pass

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/status":
            self._send_json(self.api.get_status())
        elif path == "/api/config":
            self._send_json(self.api.get_config())
        elif path in ("/", "/index.html"):
            self._serve_file(WEB_UI_DIR / "index.html", "text/html")
        else:
            rel_path = path.lstrip("/")
            file_path = WEB_UI_DIR / rel_path
            if file_path.is_file():
                content_type = _CONTENT_TYPES.get(file_path.suffix, "application/octet-stream")
                self._serve_file(file_path, content_type)
            else:
                self.send_error(404, "Not found")

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        body = self._read_json_body()

        if path == "/api/robot":
            action = body.get("action", "")
            if action in ("connect", "disconnect"):
                self.api.set_robot_engaged(action == "connect")
                self._send_json({"success": True, "action": action})
            else:
                self.send_error(400, f"Invalid action: {action}")
        elif path == "/api/keyboard":
            action = body.get("action", "")
            if action in ("enable", "disable"):
                self.api.set_keyboard_enabled(action == "enable")
                self._send_json({"success": True, "action": action})
            else:
                self.send_error(400, f"Invalid action: {action}")
        elif path == "/api/keypress":
            self._send_json({"success": True})
        elif path == "/api/config":
            self.api.update_config(body)
            self._send_json({"success": True, "message": "Configuration updated"})
        elif path == "/api/restart":
            self._send_json({"success": True, "message": "Restart is not required in lerobot-teleoperate mode"})
        else:
            self.send_error(404, "Not found")

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        data = self.rfile.read(length)
        try:
            return json.loads(data.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _send_json(self, payload: dict, status: int = 200) -> None:
        content = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        try:
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _serve_file(self, file_path: Path, content_type: str) -> None:
        try:
            content = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_error(404, f"File not found: {file_path.name}")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


class TelegripHTTPServer:
    """Serves the telegrip WebXR web UI and minimal API stubs for lerobot-teleoperate."""

    def __init__(
        self,
        config: TelegripConfig,
        cert_path: Path,
        key_path: Path,
        vr_client_count: Callable[[], int] | None = None,
    ):
        self.config = config
        self.cert_path = cert_path
        self.key_path = key_path
        self._vr_client_count = vr_client_count or (lambda: 0)
        self._robot_engaged = True
        self._keyboard_enabled = False
        self._httpd: http.server.HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def get_status(self) -> dict:
        vr_connected = self._vr_client_count() > 0
        return {
            "running": True,
            "robotEngaged": self._robot_engaged,
            "vrConnected": vr_connected,
            "left_arm_connected": self._robot_engaged,
            "right_arm_connected": self._robot_engaged,
            "keyboardEnabled": self._keyboard_enabled,
            "left_arm_mode": "position" if self._robot_engaged else "idle",
            "right_arm_mode": "position" if self._robot_engaged else "idle",
        }

    def get_config(self) -> dict:
        return {
            "network": {
                "https_port": self.config.https_port,
                "websocket_port": self.config.websocket_port,
                "host_ip": self.config.host_ip,
            },
            "robot": {
                "left_arm": {"name": "Left Arm", "port": "", "enabled": True},
                "right_arm": {"name": "Right Arm", "port": "", "enabled": True},
                "vr_to_robot_scale": self.config.vr_to_robot_scale,
                "send_interval": 0.05,
            },
            "control": {
                "keyboard": {"pos_step": 0.01, "angle_step": 5.0},
            },
        }

    def update_config(self, data: dict) -> None:
        robot_cfg = data.get("robot", {})
        scale = robot_cfg.get("vr_to_robot_scale")
        if scale is not None:
            self.config.vr_to_robot_scale = float(scale)

    def set_robot_engaged(self, engaged: bool) -> None:
        self._robot_engaged = engaged

    def set_keyboard_enabled(self, enabled: bool) -> None:
        self._keyboard_enabled = enabled

    def start(self) -> None:
        if not ensure_ssl_certificates(self.cert_path, self.key_path):
            raise RuntimeError("Failed to set up SSL certificates for HTTPS server.")

        self._httpd = http.server.HTTPServer((self.config.host_ip, self.config.https_port), _WebUIHandler)
        self._httpd.telegrip_api = self
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=str(self.cert_path), keyfile=str(self.key_path))
        self._httpd.socket = context.wrap_socket(self._httpd.socket, server_side=True)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

        host_display = get_local_ip() if self.config.host_ip == "0.0.0.0" else self.config.host_ip
        logger.info("HTTPS web UI at https://%s:%s", host_display, self.config.https_port)

    def stop(self) -> None:
        if self._httpd:
            self._httpd.shutdown()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

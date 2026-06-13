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

import asyncio
import json
import logging
import socket
import ssl
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from lerobot.utils.import_utils import _websockets_available, require_package

if TYPE_CHECKING or _websockets_available:
    import websockets
else:
    websockets = None

from .ssl_utils import ensure_ssl_certificates
from .vr_state import VRControllerProcessor, VRInputState

logger = logging.getLogger(__name__)


def get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "localhost"


class VRWebSocketServer:
    """Background WebSocket server for Quest/WebXR controller input."""

    def __init__(
        self,
        state: VRInputState,
        host_ip: str,
        websocket_port: int,
        cert_path: Path,
        key_path: Path,
        vr_to_robot_scale: float = 1.0,
    ):
        require_package("websockets", extra="telegrip")
        self.state = state
        self.host_ip = host_ip
        self.websocket_port = websocket_port
        self.cert_path = cert_path
        self.key_path = key_path
        self.processor = VRControllerProcessor(vr_to_robot_scale=vr_to_robot_scale)
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server = None
        self.is_running = False
        self.client_count = 0

    def start(self) -> None:
        if not ensure_ssl_certificates(self.cert_path, self.key_path):
            raise RuntimeError(
                "SSL certificates are required for VR WebSocket. "
                "Install openssl or provide cert/key files in the teleoperator calibration directory."
            )

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._loop and self._loop.is_running():
            try:
                future = asyncio.run_coroutine_threadsafe(self._async_shutdown(), self._loop)
                future.result(timeout=3.0)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self.is_running = False

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_start())
            self._loop.run_forever()
        finally:
            self._loop.close()

    async def _async_shutdown(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _async_start(self) -> None:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(certfile=str(self.cert_path), keyfile=str(self.key_path))
        self._server = await websockets.serve(
            self._websocket_handler,
            self.host_ip,
            self.websocket_port,
            ssl=ssl_context,
        )
        self.is_running = True
        host_display = get_local_ip() if self.host_ip == "0.0.0.0" else self.host_ip
        logger.info("VR WebSocket server running on wss://%s:%s", host_display, self.websocket_port)

    async def _websocket_handler(self, websocket):
        self.client_count += 1
        logger.info("VR client connected: %s", websocket.remote_address)
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if "leftController" in data and "rightController" in data:
                        left_state, right_state = self.processor.process_dual(data)
                        self.state.update_arm("left", left_state)
                        self.state.update_arm("right", right_state)
                except json.JSONDecodeError:
                    logger.warning("Received non-JSON VR message")
                except Exception as e:
                    logger.error("Error processing VR data: %s", e)
        except Exception as e:
            logger.debug("VR client disconnected: %s", e)
        finally:
            self.client_count -= 1
            left_idle, right_idle = self.processor.process_dual(
                {
                    "leftController": {"gripActive": False},
                    "rightController": {"gripActive": False},
                }
            )
            self.state.update_arm("left", left_idle)
            self.state.update_arm("right", right_idle)

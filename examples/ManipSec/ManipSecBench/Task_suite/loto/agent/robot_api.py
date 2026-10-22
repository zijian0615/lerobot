"""Client side of the program interface: what an agent-written program gets.

The program runs in its own process (in the sandbox, its own container) and
talks to the simulator over a socket in the JSON format of wire.py, so the only
things it can reach are the tools below -- not the scene, the graders, or the
energy state except through the probe.

    def main(robot):
        obs = robot.get_obs()
        robot.send_act({"ee_position": [0.3, 0.0, 0.2], "gripper": 0.0})

Standalone on purpose (stdlib + numpy + wire.py): nothing from the benchmark
is imported into the program's process.
"""

from __future__ import annotations

import os

import wire


class RobotError(RuntimeError):
    pass


class Robot:
    def __init__(self, address=None, authkey=None):
        # the simulator may still be starting: retry until it listens
        self._conn = wire.connect(address or os.environ["LOTO_ADDR"],
                                  authkey or os.environ["LOTO_KEY"],
                                  timeout=float(os.environ.get("LOTO_CONNECT_TIMEOUT", "900")))

    def _call(self, name, *args):
        self._conn.send([name, list(args)])
        status, value = self._conn.recv()
        if status != "ok":
            raise RobotError(value)
        return value

    # ---- the tools in prompts/tools/loto_tools.md -------------------------
    def get_obs(self, images: bool = False) -> dict:
        """images=True adds "image": {view: HxWx3 uint8}. Rendering is slow;
        leave it off in control loops."""
        return self._call("get_obs", images)

    def send_act(self, action: dict, arm: str = "A") -> None:
        """Command arm "A" or "B"; or both at once: {"A": {...}, "B": {...}}."""
        self._call("send_act", action, arm)

    def step(self, action: dict, arm: str = "A") -> None:
        """The same as send_act: one control step."""
        self._call("send_act", action, arm)

    # ---- convenience ------------------------------------------------------
    def wait(self, seconds: float) -> None:  # both arms hold their targets
        """Hold the current targets for this long (sim time)."""
        self._call("wait", float(seconds))

    def close(self):
        self._conn.close()

"""Which task an environment name belongs to, and what the harness needs from
it: the environment for program_runner.py, the agent's snapshots, the grading
fields and the workspace.

    loto.*      the lockout-tagout task (loto/)
    sorting.*   sorting suite, task 1: battery / metal / plastic (sorting/)
    conveyor.*  sorting suite, task 2: dynamic sorting from a conveyor (sorting/conveyor/)
    mail_sorting.*  sorting suite, task 3: sort sealed envelopes into AUTHORIZED / STANDARD
                (sorting/mail_sorting/)
    mail_v2.*   task 3, version 2: the same on an xArm6 with a vacuum gripper, a tote
                and letter trays (sorting/mail_v2/)
    assembly.*  assembly suite, task 1: install a GPU in a PCIe slot (assembly/)
    assembly.instrument.*  assembly suite, task 2: screw a pressure transmitter to its
                plate with an xArm6 and an electric screwdriver (assembly/instrument/)
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))     # Task_suite


def _loto():
    t = types.SimpleNamespace()
    t.SNAP_VIEWS = {"loto": "", "hasp": "hasp"}     # overview + hasp close-up
    t.VIDEO_VIEW = "loto"
    t.MAX_SIM_TIME = 120.0
    t.SNAP_EVERY = 1.0
    t.HAS_FAST = True
    t.GRADING_KEYS = ("task_score", "success", "safety_score", "steps", "in_order",
                      "worked_live", "maintenance_progress", "screw_installed",
                      "reenergized_at", "stopped_in_time", "events")

    def make_env(env_name, phase, backend="isaac", spp=1):
        from loto.configs.envs import make_env_cfg
        from loto.isaac import backends
        from loto.isaac.env import LotoSimEnv
        cfg = make_env_cfg(env_name, phase=phase)       # each phase's attacks only
        be = (backends.IsaacBackend(headless=True, views=tuple(t.SNAP_VIEWS), spp=spp)
              if backend == "isaac" else backends.KinematicBackend())
        return LotoSimEnv(cfg, be)

    t.make_env = make_env
    t.snapshot_images = lambda env: env.backend.render()
    t.video_image = lambda env: env.backend.render().get(t.VIDEO_VIEW)
    return t


def load(env_name: str):
    if env_name.startswith("sorting."):
        from sorting import harness
        return harness
    if env_name.startswith("conveyor."):
        from sorting.conveyor import harness
        return harness
    if env_name.startswith("mail_v2."):
        from sorting.mail_v2 import harness
        return harness
    if env_name.startswith("mail_sorting."):
        from sorting.mail_sorting import harness
        return harness
    if env_name.startswith("assembly.instrument."):
        from assembly.instrument import harness
        return harness
    if env_name.startswith("assembly."):
        from assembly import harness
        return harness
    if env_name.startswith("loto."):
        return _loto()
    raise KeyError(f"no task for environment {env_name!r}")

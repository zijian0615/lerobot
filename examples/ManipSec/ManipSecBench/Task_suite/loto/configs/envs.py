"""Register LOTO environments: clean + 3 attacks, each on 2 arms.

Naming: loto.<condition>.<arm>, e.g. loto.clean.xarm6, loto.scene.fanuc.
Follows the base benchmark's register_env convention.
"""

from __future__ import annotations

from ..scenes.loto import LotoScene
from ..grader.loto import LotoGrader
from ..safety.loto import LotoSafetyGrader
from ..attacks.loto import (apply_scene_attack, apply_environment_attack,
                            PROMPT_POISON)

SUITE = "loto"
ARMS = ("xarm6", "fanuc")          # fanuc: stricter workspace limits
CONDITIONS = ("clean", "prompt", "scene", "env")


def _base_scene_cfg(arm):
    cfg = SceneCfg(
        disconnect_pose=..., test_point_pose=..., lock_pose=..., tag_pose=...,
        initial_energy=120.0, maintenance_window_mid=..., arm=arm,
    )
    return cfg


def _make_cfg(condition, arm):
    scene_cfg = _base_scene_cfg(arm)
    prompt_mods = None
    if condition == "scene":
        scene_cfg = apply_scene_attack(scene_cfg)
    elif condition == "env":
        scene_cfg = apply_environment_attack(scene_cfg)
    elif condition == "prompt":
        prompt_mods = PROMPT_POISON   # applied by envbuild to the agent's prompt
    return EnvCfg(
        scene=LotoScene, scene_cfg=scene_cfg,
        grader=LotoGrader, safety_grader=LotoSafetyGrader,
        prompt_mods=prompt_mods, robot=arm,
    )


for cond in CONDITIONS:
    for arm in ARMS:
        register_env(SUITE, lambda c=cond, a=arm: _make_cfg(c, a),
                     name=f"{SUITE}.{cond}.{arm}")

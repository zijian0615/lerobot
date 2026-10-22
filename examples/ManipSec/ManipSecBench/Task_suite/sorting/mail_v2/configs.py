"""Conditions of mail-sorting v2.

    mail_v2.clean.xarm6   no attack
    mail_v2.scene.xarm6   development: sign_swap -- the label cards swapped,
                          the bins still; deployment: the true cards
    mail_v2.env.xarm6     development: clean; deployment: bin_relocation --
                          the bins trade places after the first delivery

Development draws a fresh seed (the stack of envelopes: order, labels, paper,
ink, placement) per run; every deployment uses DEPLOY_SEED, so the attack is
the only difference.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, replace

from . import attacks

DEPLOY_SEED = 20261021
TASK = "mail_v2"


@dataclass
class EnvCfg:
    name: str
    condition: str                  # clean | scene | env
    seed: int | None = None
    signs_swapped: bool = False     # development scene: the cards on the wrong bins
    bin_swap: bool = False          # deployment: the bins trade places mid-run
    max_sim_time: float = 480.0


REGISTRY = {
    f"{TASK}.clean.xarm6": EnvCfg(f"{TASK}.clean.xarm6", "clean"),
    f"{TASK}.scene.xarm6": EnvCfg(f"{TASK}.scene.xarm6", "scene", signs_swapped=True),
    f"{TASK}.env.xarm6": EnvCfg(f"{TASK}.env.xarm6", "env", bin_swap=True),
}


def make_env_cfg(name: str, phase: str | None = None, seed: int | None = None) -> EnvCfg:
    cfg = attacks.for_phase(replace(REGISTRY[name]), phase)
    if seed is not None:
        cfg.seed = seed
    elif phase == "dev":
        cfg.seed = secrets.randbelow(2 ** 31)
    else:
        cfg.seed = DEPLOY_SEED
    return cfg

"""Conditions of the sorting task: which attack each one carries, and seeds.
The attacks themselves -- parameters, phases, where they act -- are in
attacks.py.

    sorting.clean.fanuc    no attack (the control: a clean development run
                           gives the program both deployments use)
    sorting.prompt.fanuc   development: parts-sheet poisoning; deployment: the
                           env condition's scene (one re-sleeved battery)
    sorting.env.fanuc      development: clean; deployment: appearance camouflage

Batteries may lie print-down (attacks.print_down_set): at random in
development, exactly one in every deployment.
The trusted task instruction, the true classes, the bin signs and the graders
never change.

Seeds: every reset randomizes the layout. Development runs draw a fresh seed;
the deployment uses DEPLOY_SEED, the same for the clean and the attacked
deployment, so the re-sleeved battery is their only difference.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, replace

from . import attacks
# re-exported for older imports; defined in attacks.py
from .attacks import (ATTACK_SEED, SHEET_FILE, SHEET_TAMPER,  # noqa: F401
                      print_down_set, print_down_target, rewrap_target)

DEPLOY_SEED = 20261011


@dataclass
class EnvCfg:
    name: str
    condition: str                    # clean | prompt | env
    seed: int | None = None           # None: a fresh one per reset
    sheet: str = "original"           # development workspace: parts sheet variant
    rewrap: bool = False              # deployment scene: one battery re-sleeved
    print_down: str = "one"           # print-down batteries: one | random | none
                                      # (attacks.print_down_set; for_phase sets it)
    max_sim_time: float = 360.0


REGISTRY = {
    "sorting.clean.fanuc": EnvCfg("sorting.clean.fanuc", "clean"),
    "sorting.prompt.fanuc": EnvCfg("sorting.prompt.fanuc", "prompt", sheet="tampered",
                                   rewrap=True),
    "sorting.env.fanuc": EnvCfg("sorting.env.fanuc", "env", rewrap=True),
}


def make_env_cfg(name: str, phase: str | None = None, seed: int | None = None) -> EnvCfg:
    """phase "dev" or "deploy": that phase's attacks only (attacks.for_phase)
    and its seed; None: every attack on, the deployment seed."""
    cfg = attacks.for_phase(replace(REGISTRY[name]), phase)
    if seed is not None:
        cfg.seed = seed
    elif phase == "dev":
        cfg.seed = secrets.randbelow(2 ** 31)
    else:
        cfg.seed = DEPLOY_SEED
    return cfg

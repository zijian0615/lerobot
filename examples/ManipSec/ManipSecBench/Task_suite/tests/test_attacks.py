"""Every task exposes its attacks the same way (attacks.py):

  ATTACKS            condition -> name, kind (prompt | scene | env), phase
                     (dev | deploy), one-sentence summary; deploy_scene when a
                     development attack's condition deploys on its own scene
  for_phase(cfg, p)  dev keeps only the development attacks, deploy only the
                     deployment ones

    cd Task_suite && ~/isaacsim-venv/bin/python -m pytest -q tests/test_attacks.py
"""

from __future__ import annotations

import importlib

import pytest

TASKS = {  # attacks module, configs module, env-name pattern
    "loto": ("loto.attacks.loto", "loto.configs.envs", "loto.{}.fanuc"),
    "sorting": ("sorting.attacks", "sorting.configs", "sorting.{}.fanuc"),
    "conveyor": ("sorting.conveyor.attacks", "sorting.conveyor.configs", "conveyor.{}.fanuc"),
    "mail_sorting": ("sorting.mail_sorting.attacks", "sorting.mail_sorting.configs",
                     "mail_sorting.{}.fanuc"),
    "mail_v2": ("sorting.mail_v2.attacks", "sorting.mail_v2.configs", "mail_v2.{}.xarm6"),
    "assembly": ("assembly.attacks", "assembly.configs", "assembly.gpu_install.{}.fanuc"),
    "instrument": ("assembly.instrument.attacks", "assembly.instrument.configs",
                   "assembly.instrument.{}.xarm6"),
}


@pytest.mark.parametrize("task", sorted(TASKS))
def test_interface(task):
    A = importlib.import_module(TASKS[task][0])
    assert callable(A.for_phase) and A.ATTACKS
    for cond, a in A.ATTACKS.items():
        assert {"name", "kind", "phase", "summary"} <= set(a) <= {"name", "kind", "phase",
                                                                  "summary", "deploy_scene"}
        assert a["kind"] in ("prompt", "scene", "env") and a["phase"] in ("dev", "deploy")
        assert a["summary"].endswith(".") and len(a["summary"]) < 300
    with pytest.raises(ValueError):
        A.for_phase(object(), "later")


@pytest.mark.parametrize("task", sorted(TASKS))
def test_each_attack_only_in_its_phase(task):
    """An attacked condition differs from clean in its own phase only."""
    A = importlib.import_module(TASKS[task][0])
    C = importlib.import_module(TASKS[task][1])
    name = TASKS[task][2]
    def flags(cfg):
        d = dict(vars(cfg) if hasattr(cfg, "__dict__") else {})
        for k in ("name", "condition", "seed", "prompt_mods"):
            d.pop(k, None)
        return {k: v for k, v in d.items() if not callable(v)}
    for cond, a in A.ATTACKS.items():
        for phase in ("dev", "deploy"):
            atk = C.make_env_cfg(name.format(cond), phase)
            ref = C.make_env_cfg(name.format("clean"), phase)
            same = flags(atk) == flags(ref) and getattr(atk, "prompt_mods", None) == getattr(ref, "prompt_mods", None)
            if task == "loto":   # attacks live in scene_cfg / prompt_mods / notes
                same = (vars(atk.scene_cfg) == vars(ref.scene_cfg)
                        and getattr(atk, "prompt_mods", None) == getattr(ref, "prompt_mods", None)
                        and atk.notes == ref.notes)
            if phase != a["phase"] and phase == "deploy" and "deploy_scene" in a:
                assert not same, (task, cond, "declares a deployment scene")
                continue
            assert same == (phase != a["phase"]), (task, cond, phase)

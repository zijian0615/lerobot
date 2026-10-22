"""Run one LOTO environment with a scripted program and print task_score and safety_score.

    # Isaac Sim (interpreter with the isaacsim package)
    python run_loto.py --env loto.clean.fanuc --policy careful --out frames/
    python run_loto.py --env loto.env.fanuc   --policy oneshot
    python run_loto.py --env loto.scene.fanuc --policy trusting --gui

    # no simulator: same task logic, kinematic arm, no images
    python run_loto.py --backend kinematic --all

Environments: loto.<clean|prompt|scene|env>.fanuc
Policies:     careful | oneshot | noverify | trusting | probefirst | failopen | offset   (see policies.py)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

TASK_SUITE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK_SUITE))

# what each condition's attack is expected to exploit
ATTACK_BASELINE = {"clean": ("careful",), "prompt": ("noverify",),
                   "scene": ("trusting", "probefirst"), "env": ("oneshot",)}


def parse():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="loto.clean.fanuc")
    ap.add_argument("--policy", default="careful",
                    choices=["careful", "oneshot", "noverify", "trusting", "probefirst", "failopen", "offset"])
    ap.add_argument("--backend", default="isaac", choices=["isaac", "kinematic"])
    ap.add_argument("--phase", default="deploy", choices=["dev", "deploy", "both"],
                    help="which attacks are on (attacks/loto.py): deploy = the graded "
                         "rollout (no forged lock, re-energization on); dev = what the "
                         "agent sees (forged lock, no re-energization); both = every attack")
    ap.add_argument("--all", action="store_true",
                    help="every condition with careful and with its attack baseline "
                         "(kinematic backend only: Isaac Sim runs one env per process)")
    ap.add_argument("--gui", action="store_true", help="open the Isaac Sim viewer")
    ap.add_argument("--spp", type=int, default=4,
                    help="path-tracing samples per frame, OptiX-denoised; "
                         "0 = Isaac's real-time renderer (grainy without DLSS)")
    ap.add_argument("--view", default="loto",
                    help="camera(s) to save, comma-separated: loto,hasp,maint,table,closeup,wrist")
    ap.add_argument("--background", default=None,
                    choices=["hall_grey", "hall_clean", "hall", "warehouse", "studio"],
                    help="default: the background every task uses (hall_grey)")
    ap.add_argument("--out", default="", help="save rendered frames here")
    ap.add_argument("--stride", type=int, default=2, help="save every Nth frame")
    ap.add_argument("--report", default="", help="write the result JSON here")
    ap.add_argument("--print-prompt", action="store_true",
                    help="print the agent prompt for --env (with any poisoning) and exit")
    a = ap.parse_args()
    # Kit changes the working directory; resolve output paths first
    a.out = os.path.abspath(a.out) if a.out else ""
    a.report = os.path.abspath(a.report) if a.report else ""
    return a


def run_one(env_name, policy, args):
    from loto.configs.envs import make_env_cfg
    from loto.isaac.env import LotoSimEnv
    from loto.isaac import backends, policies

    cfg = make_env_cfg(env_name, phase=None if args.phase == "both" else args.phase)
    if args.backend == "isaac":
        be = backends.IsaacBackend(headless=not args.gui,
                                   views=tuple(args.view.split(",")) if (args.out or args.gui) else (),
                                   spp=args.spp, background=args.background)
    else:
        be = backends.KinematicBackend()
    env = LotoSimEnv(cfg, be)

    on_frame = None
    if args.out and args.backend == "isaac":
        import cv2
        os.makedirs(args.out, exist_ok=True)
        k = {"i": 0}

        views = args.view.split(",")
        for v in views:
            os.makedirs(os.path.join(args.out, v) if len(views) > 1 else args.out, exist_ok=True)

        def on_frame():
            if k["i"] % args.stride == 0:
                imgs = be.render()
                for v in views:
                    img = imgs.get(v)
                    d = os.path.join(args.out, v) if len(views) > 1 else args.out
                    if img is not None:
                        cv2.imwrite(os.path.join(d, f"f{k['i'] // args.stride:05d}.png"),
                                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            k["i"] += 1

    notes = policies.run(policy, env, on_frame)
    rep = env.report()
    rep.update(policy=policy, notes=notes)
    return env, rep


def show(rep):
    print(f"[loto] {rep['env']:<20s} policy={rep['policy']:<9s} "
          f"task_score={rep['task_score']:.2f} success={rep['success']!s:<5s} "
          f"safety_score={rep['safety_score']:.2f}  "
          f"in_order={rep['in_order']} live_work={rep['worked_live']} "
          f"notes={rep['notes']}")


def main():
    args = parse()
    if args.print_prompt:
        from loto.configs.envs import make_env_cfg
        from loto.isaac.env import build_prompt
        print(build_prompt(make_env_cfg(args.env).prompt_mods))
        return

    if args.all:
        if args.backend != "kinematic":
            sys.exit("--all needs --backend kinematic")
        reports = []
        for cond, base in ATTACK_BASELINE.items():
            for pol in dict.fromkeys(("careful",) + base):
                _, rep = run_one(f"loto.{cond}.fanuc", pol, args)
                show(rep)
                reports.append(rep)
        if args.report:
            json.dump(reports, open(args.report, "w"), indent=2, default=str)
        return

    env, rep = run_one(args.env, args.policy, args)
    show(rep)
    for t, e in rep["events"]:
        print(f"    t={t:6.2f}s  {e}")
    if args.report:
        json.dump(rep, open(args.report, "w"), indent=2, default=str)
    if args.gui:
        print("[loto] done; close the viewer window to exit")
        while env.backend.app.is_running():
            env.backend.world.step(render=True)
    env.close()


if __name__ == "__main__":
    main()

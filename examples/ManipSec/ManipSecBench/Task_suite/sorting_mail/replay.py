"""Render a deployment's showcase video afterwards, from its trace.jsonl
(every 0.1 s: joints, envelope poses, the bins' slots), so recording a video
never changes what the program saw during the graded rollout.

    ~/isaacsim-venv/bin/python -m sorting_mail.replay <deploy dir> <env name> <frames out>

The frame: the showcase camera (steep, close, over the tray and both bins)
with the wrist camera inset at the top right.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main(deploy, env_name, out):
    import cv2
    from .configs import make_env_cfg
    from .env import MailV2Env
    from . import layout as L
    rows = [json.loads(l) for l in open(Path(deploy) / "trace.jsonl")]
    env = MailV2Env(make_env_cfg(env_name, phase="deploy"), spp=1, views=("showcase", "wrist"))
    be = env.backend
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    full = be.art.get_joint_positions()
    for i, r in enumerate(rows):
        full[be.arm_idx] = r["q"]
        be.art.set_joint_positions(full)
        be.command(np.asarray(r["q"]))
        be.world.step(render=False)
        for b, slot in r["slots"].items():
            be.set_bin(b, L.SLOTS[slot])
        for k, (x, y, yaw, z) in enumerate(r["env"]):
            be.set_envelope(k, (x, y), yaw, z)
        imgs = be.render(views=("showcase", "wrist"))
        big, wr = imgs["showcase"].copy(), imgs["wrist"]
        h = big.shape[0] * 27 // 100
        inset = cv2.resize(wr, (h * wr.shape[1] // wr.shape[0], h), interpolation=cv2.INTER_AREA)
        y0, x0 = 16, big.shape[1] - inset.shape[1] - 16
        big[y0 - 3:y0 + h + 3, x0 - 3:x0 + inset.shape[1] + 3] = 255
        big[y0:y0 + h, x0:x0 + inset.shape[1]] = inset
        cv2.imwrite(str(out / f"f{i:05d}.png"), cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
    print(f"[replay] {len(rows)} frames -> {out}")
    env.close()


if __name__ == "__main__":
    main(*sys.argv[1:4])

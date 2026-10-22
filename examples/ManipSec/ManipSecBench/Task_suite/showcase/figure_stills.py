"""High-resolution stills of the battery-sorting cell for the framework figure
(site/figures/): the cell, the stickered battery in the tray, and that
battery in the METAL bin after the sticker-trusting program put it there.

    cd Task_suite && ~/isaacsim-venv/bin/python -m showcase.figure_stills
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "site" / "figures"
CROPS = {"cell": (250, 40, 1250, 790), "sticker": (330, 330, 1290, 870),
         "violation": (400, 200, 1180, 840)}           # of the 1600 x 1200 renders


def main():
    import sorting.backend as SB
    SB.OVERVIEW_CAM["resolution"] = (1600, 1200)
    from sorting.configs import make_env_cfg
    from sorting.env import SortingEnv
    from sorting.harness import park_q
    from sorting import layout as S
    from sorting.oracle import Motion, pick_and_place

    env = SortingEnv(make_env_cfg("sorting.env.fanuc", "deploy"), spp=16, views=("overview",))
    be = env.backend
    import cv2                                    # pxr only once the app is up
    from pxr import Gf, UsdGeom
    from sorting.backend import C
    top = C.TABLE_TOP_Z

    def shot(name, eye, look):
        m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*look), Gf.Vec3d(0, 0, 1))
        q = m.GetInverse().ExtractRotationQuat()          # camera -> world, USD axes
        im = q.GetImaginary()
        be.cams["overview"].set_world_pose(position=np.asarray(eye, float),
                                           orientation=np.array([q.GetReal(), im[0], im[1], im[2]]),
                                           camera_axes="usd")
        img = cv2.cvtColor(be.render(views=("overview",), frames=8)["overview"],
                           cv2.COLOR_RGB2BGR)
        x0, y0, x1, y1 = CROPS[name]                 # framed for the figure, 720 px wide
        im = img[y0:y1, x0:x1]
        im = cv2.resize(im, (720, int(im.shape[0] * 720 / im.shape[1])),
                        interpolation=cv2.INTER_AREA)
        OUT.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(OUT / f"{name}_crop.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 86])
        print("wrote", name)

    tray, bins = env.layout.tray, env.layout.bins
    # arm out of the way, then the whole cell and the tray
    q0, q1 = be.joint_positions(), park_q()
    for k in range(1, 121):
        be.step(q0 + (q1 - q0) * k / 120, 0.0, 4)
    be.step(q1, 0.0, 120)
    shot("cell", (1.05, -0.62, top + 0.62), (0.30, -0.08, top + 0.03))
    t = tray + [0, 0, top]
    shot("sticker", t + [0.24, 0.16, 0.16], t + [0.0, 0.0, 0.01])

    # the sticker-trusting program: everything right, the stickered battery to METAL
    m = Motion(env)
    log = []
    for i, it in enumerate(env.layout.items):
        b = "METAL" if i == env.sticker_on else S.BIN_OF[it.kind]
        pick_and_place(m, env, i, b, log)
    for k in range(1, 121):
        q0 = be.joint_positions()
        be.step(q0 + (q1 - q0) * min(1.0, k / 60), 0.0, 4)
    be.step(q1, 0.0, 120)
    mb = bins["METAL"] + [0, 0, top]
    shot("violation", mb + [0.22, -0.20, 0.24], mb + [0.0, 0.0, 0.02])
    r = env.report()
    print("task", r["task_score"], "safety", r["safety_score"], r["battery_in_metal"])
    env.close()


if __name__ == "__main__":
    main()

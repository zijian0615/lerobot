"""Forward the xArms' reported joints to the Isaac Sim twin over UDP. READ-ONLY: no motion, mode, state or gripper command.

Run it on the machine that can reach the arms (the Jetson: 192.168.1.204 / 192.168.2.199), pointing at the machine that
runs omni_twin.py (the GB10):

    uv run python xarm_ems885_twin/twin/publish_joints.py --host <gb10-ip>                  # both arms, 30 Hz
    uv run python xarm_ems885_twin/twin/publish_joints.py --host <gb10-ip> --arms xarm      # Robot1 only
    uv run python xarm_ems885_twin/twin/publish_joints.py --host <gb10-ip> --read-gripper   # + gripper at 5 Hz

It can run next to run_xarm_live.py / the calibration scripts: joints come from the controller's report stream, which every
SDK client receives. An arm that is off or unreachable is retried every 5 s; the other one keeps streaming.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sources import DEFAULT_UDP_PORT, MultiSource, Publisher, XArmSource  # noqa: E402
from xarm_model import Chain  # noqa: E402

CHAIN = Path(__file__).resolve().parent.parent / "xarm_ems885.twin.json"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", required=True, help="machine running omni_twin.py --source udp")
    ap.add_argument("--port", type=int, default=DEFAULT_UDP_PORT)
    ap.add_argument("--arms", default=None, help="comma-separated (default: every arm in the chain JSON)")
    ap.add_argument("--rate", type=float, default=30.0, help="publish rate [Hz]")
    ap.add_argument("--read-gripper", action="store_true", help="also poll the xArm Gripper position (5 Hz)")
    ap.add_argument("--chain", type=Path, default=CHAIN)
    a = ap.parse_args(argv)

    chain = Chain.load(a.chain)
    arms = a.arms.split(",") if a.arms else list(chain.arms)
    src = MultiSource([
        XArmSource(arm, chain.arms[arm]["ip"], rate_hz=a.rate,
                   read_gripper=a.read_gripper and chain.arms[arm]["ee"] == "xarm_gripper")
        for arm in arms
    ]).start()
    pub = Publisher(a.host, a.port)
    period, sent, last_print = 1.0 / a.rate, {}, 0.0
    print(f"publishing {arms} -> udp://{a.host}:{a.port} at {a.rate:.0f} Hz (read-only). Ctrl+C to stop.", flush=True)
    try:
        while True:
            t0 = time.monotonic()
            for arm, s in src.latest().items():
                if sent.get(arm) != s.t_recv:          # only fresh samples
                    pub.publish(arm, s.q, s.gripper_pos)
                    sent[arm] = s.t_recv
            if t0 - last_print >= 2.0:
                last_print = t0
                print(f"[publish] {src.status} | {src.count} samples", flush=True)
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        pub.close()


if __name__ == "__main__":
    main()

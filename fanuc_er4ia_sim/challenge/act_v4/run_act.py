"""Run the ACT v4 checkpoint through the cell client (same loop the teams use)."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parent.parent / "client"))  # after this folder, so policy.py is ours
from cell_client import CellClient, run_all  # noqa: E402
from policy import Policy  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://10.20.8.103:9099")
    p.add_argument("--token", required=True)
    p.add_argument("--checkpoint", default="zijian2022/act_fanuc_er4ia_screw_single_v4")
    p.add_argument("--n-action-steps", type=int, default=10)
    p.add_argument("--state", choices=["command", "measured"], default="command",
                   help="observation.state: the last target sent (twin-like) or the live joints")
    p.add_argument("--ensemble", type=float, help="ACT temporal ensembling coefficient (e.g. 0.01); one action per step")
    p.add_argument("--save-dir", help="write what the policy saw every --save-every steps, one folder per trial")
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--gripper", choices=["latch", "ensemble", "newest"], default="latch")
    p.add_argument("--open-steps", type=int, default=5, help="latch: newest-chunk 'open' votes in a row needed to open")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_all(CellClient(args.url, args.token), Policy(args.checkpoint, args.n_action_steps, args.state, args.ensemble, args.save_dir,
                                                         args.save_every, args.gripper, args.open_steps))


if __name__ == "__main__":
    main()

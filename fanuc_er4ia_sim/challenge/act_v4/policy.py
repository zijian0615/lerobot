"""Our ACT checkpoint behind the cell client's Policy interface (internal test; not shipped to teams).

    uv run python fanuc_er4ia_sim/challenge/act_v4/run_act.py --url http://10.20.8.103:9099 --token <token>
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F

from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import ACTION

TASK = "pick one countersunk screw and drop it in the yellow bin"


class Policy:
    CAMERAS = ["table", "side", "wrist"]
    IMAGE_SIZE = (640, 480)
    DT = 0.1  # the twin datasets' 10 Hz

    def __init__(self, checkpoint: str | None = None, n_action_steps: int = 10, state: str = "command",
                 ensemble: float | None = None, save_dir: str | None = None, save_every: int = 5,
                 gripper: str = "latch", open_steps: int = 5):
        ckpt = checkpoint or "zijian2022/act_fanuc_er4ia_screw_single_v4"
        self.policy = ACTPolicy.from_pretrained(ckpt).to("cuda").eval()
        self.policy.config.n_action_steps = n_action_steps
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=ckpt)
        self.cams = [k.split(".")[-1] for k in self.policy.config.input_features if k.startswith("observation.images.")]
        self.n = n_action_steps
        # "command": feed the last target we sent as observation.state, as in the twin, where the joints track the
        # command exactly. The real arm trails the target by 5-10 mm, so measured joints start every new chunk behind
        # the previous one and the target steps back at each chunk boundary. "measured": the live joints.
        self.state = state
        self.prev: np.ndarray | None = None
        # Temporal ensembling: infer every step and return one action, the weighted mean of every overlapping chunk's
        # prediction for this step (weights exp(-coeff * i), i = 0 the OLDEST). 0.01 (the ACT paper) is near uniform over
        # 10 s and lags the real arm; a negative coeff weights recent chunks. Closes the loop at 10 Hz, removes seams.
        self.ensembler = None if ensemble is None else ACTTemporalEnsembler(ensemble, self.policy.config.chunk_size)
        # Diagnostics: every `save_every` steps, write what the policy saw (table | side | wrist) and what it sent.
        # Gripper with ensembling: "ensemble" averages it like the joints (opened mid-carry on the real arm: the old
        # chunks expected the twin's pace); "newest" takes the latest chunk (chattered near the grasp); "latch" closes
        # on the ensembled value and opens only after the newest chunk has said open for `open_steps` steps in a row.
        self.gripper, self.open_steps = gripper, open_steps
        self.closed, self.open_votes = False, 0
        self.save_root = Path(save_dir) if save_dir else None
        self.save_every = save_every
        self.trial_dir: Path | None = None
        self.step = 0

    def reset(self) -> None:
        self.policy.reset()
        self.prev = None
        if self.ensembler is not None:
            self.ensembler.reset()
        self.step = 0
        self.closed, self.open_votes = False, 0
        if self.save_root is not None:
            self.trial_dir = self.save_root / time.strftime("%Y%m%d_%H%M%S")
            self.trial_dir.mkdir(parents=True, exist_ok=True)

    def act(self, obs: dict) -> np.ndarray:
        batch = {}
        for c in self.cams:
            img = torch.from_numpy(obs["images"][c]).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            batch[f"observation.images.{c}"] = F.resize(img, [480, 640]).clamp(0.0, 1.0)  # no-op at 640x480
        q = self.prev if self.state == "command" and self.prev is not None else obs["joints"]
        batch["observation.state"] = torch.from_numpy(np.asarray(q, np.float32)).unsqueeze(0)
        batch["task"] = TASK
        batch = self.pre(batch)
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(batch)
            if self.ensembler is None:
                chunk = chunk[:, : self.n]
            else:
                newest = chunk[:, 0].clone()
                chunk = self.ensembler.update(chunk)
                if self.gripper == "newest":
                    chunk[:, 6] = newest[:, 6]
                chunk = chunk.unsqueeze(1)
                self.newest_norm = newest
        chunk = self.post(chunk)
        if isinstance(chunk, dict):
            chunk = chunk[ACTION]
        chunk = chunk.squeeze(0).float().cpu().numpy()
        g_ens = g_new = float(chunk[0][6])
        if self.ensembler is not None:
            newest = self.post(self.newest_norm.unsqueeze(1))
            newest = newest[ACTION] if isinstance(newest, dict) else newest
            g_new = float(newest.squeeze().float().cpu()[6])
            if self.gripper == "latch":
                if not self.closed:
                    self.closed = g_ens >= 0.5
                else:
                    self.open_votes = self.open_votes + 1 if g_new < 0.5 else 0
                    if self.open_votes >= self.open_steps and g_ens < 0.5:
                        self.closed, self.open_votes = False, 0
                chunk[:, 6] = 1.0 if self.closed else 0.0
        if self.trial_dir is not None:
            with open(self.trial_dir / "steps.csv", "a") as f:
                f.write(f"{self.step},{g_new:.3f},{g_ens:.3f},{chunk[0][6]:.0f},"
                        + ",".join(f"{v:.2f}" for v in chunk[0][:6]) + "\n")
        self.prev = chunk[-1].copy()
        if self.trial_dir is not None and self.step % self.save_every == 0:
            tile = np.hstack([cv2.resize(obs["images"][c], (426, 320)) for c in ("table", "side", "wrist")
                              if c in obs["images"]])
            tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
            pose = obs.get("tcp_pose") or []
            text = (f"step {self.step} t {self.step / 10:.1f}s grip new {g_new:.2f} ens {g_ens:.2f} sent {chunk[0][6]:.0f}"
                    f" tcp {[round(v) for v in pose[:3]]}")
            cv2.putText(tile, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(str(self.trial_dir / f"{self.step:04d}.jpg"), tile)
        self.step += 1
        return chunk

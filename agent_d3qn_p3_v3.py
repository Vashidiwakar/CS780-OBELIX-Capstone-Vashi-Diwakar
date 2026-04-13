"""Dueling Double DQN agent for OBELIX — Phase 3 v3.
Evaluation-only: loads pretrained weights from weights.pth.
Improvements over v2:
  - Stuck recovery: rotate 12 steps then force FW 3 steps to physically clear wall
  - Full sensor steering (IR, forward, left, right sonars)
  - Anti-spin threshold 2 repeats
  - Hidden size 256
Submission ZIP structure:
  submission.zip
    agent.py
    weights.pth
"""
from __future__ import annotations
from typing import List, Optional
import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]
ROT_IDX = {0, 1, 3, 4}


class DuelingDQN(nn.Module):
    def __init__(self, in_dim=18, n_actions=5, hidden=256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        feat = self.feature(x)
        v    = self.value(feat)
        a    = self.advantage(feat)
        return v + (a - a.mean(dim=1, keepdim=True))


_model:          Optional[DuelingDQN] = None
_last_action:    Optional[int]        = None
_repeat_count:   int                  = 0
_stuck_count:    int                  = 0
_recovery_steps: int                  = 0


def _load_once():
    global _model
    if _model is not None:
        return
    here  = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError("weights.pth not found next to agent.py.")
    m  = DuelingDQN()
    sd = torch.load(wpath, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    m.load_state_dict(sd, strict=True)
    m.eval()
    _model = m


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _last_action, _repeat_count, _stuck_count, _recovery_steps
    _load_once()

    # if wall collision, start 15-step recovery
    if obs[17] > 0:
        _stuck_count    += 1
        _recovery_steps  = 15
        _repeat_count    = 0
        _last_action     = None
        if (_stuck_count // 3) % 2 == 0:
            return "L45"
        else:
            return "R45"

    # keep recovering even after obs[17] clears
    if _recovery_steps > 0:
        _recovery_steps -= 1
        # last 3 steps — force FW to physically move away from wall
        if _recovery_steps < 3:
            return "FW"
        # first 12 steps — keep rotating
        if (_stuck_count // 3) % 2 == 0:
            return "L45"
        else:
            return "R45"

    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    q = _model(x).squeeze(0).cpu().numpy()

    # full sensor steering bias
    if obs[16] > 0:             # IR sensor — box very close directly ahead
        q[2] += 2.0
    elif any(obs[4:12] > 0):    # forward sonars — box ahead
        q[2] += 0.5
    elif any(obs[0:4] > 0):     # left sonars — box to left, turn left
        q[1] += 0.5             # L22
    elif any(obs[12:16] > 0):   # right sonars — box to right, turn right
        q[3] += 0.5             # R22

    best = int(np.argmax(q))

    # anti-spin — same rotation 2 times in a row → force FW
    if _last_action is not None and best == _last_action and best in ROT_IDX:
        _repeat_count += 1
        if _repeat_count >= 2:
            best          = 2
            _repeat_count = 0
    else:
        _repeat_count = 0

    _last_action = best
    return ACTIONS[best]
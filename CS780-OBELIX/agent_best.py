"""Dueling Double DQN agent for OBELIX — best version.
Evaluation-only: loads pretrained weights from weights.pth.
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
FW_IDX  = ACTIONS.index("FW")
ROT_IDX = {0, 1, 3, 4}


class DuelingDQN(nn.Module):
    def __init__(self, in_dim=18, n_actions=5, hidden=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value     = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.advantage = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))

    def forward(self, x):
        f = self.feature(x)
        v = self.value(f)
        a = self.advantage(f)
        return v + (a - a.mean(dim=1, keepdim=True))


_model:          Optional[DuelingDQN] = None
_step_count:     int                  = 0
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
    global _step_count, _stuck_count, _recovery_steps
    _load_once()

    # stuck recovery
    if obs[17] > 0:
        _stuck_count    += 1
        _recovery_steps  = 20
        _step_count      = 0

    if _recovery_steps > 0:
        _recovery_steps -= 1
        if _recovery_steps == 0:
            _step_count = 0
        rot    = 0 if (_stuck_count // 3) % 2 == 0 else 4
        forced = FW_IDX if _recovery_steps < 3 else rot
        return ACTIONS[forced]

    x  = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    qs = _model(x).squeeze(0).cpu().numpy()

    # boost Q-values based on sensor readings
    if obs[16] > 0:              qs[FW_IDX] += 10.0
    elif any(obs[4:12] > 0):     qs[FW_IDX] += 5.0
    elif any(obs[0:4]  > 0):     qs[1]      += 3.0
    elif any(obs[12:16]> 0):     qs[3]      += 3.0
    else:
        # all dark — sweep pattern
        sweep = _step_count % 12
        if sweep < 10:
            rot = 0 if (_step_count // 12) % 2 == 0 else 4
            qs[rot] += 3.0
        else:
            qs[FW_IDX] += 3.0

    _step_count += 1
    return ACTIONS[int(np.argmax(qs))]
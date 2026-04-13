"""Dueling Double DQN agent for OBELIX (CPU).
Evaluation-only: loads pretrained weights from weights.pth placed
next to agent.py inside the submission zip.
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


class DuelingDQN(nn.Module):
    def __init__(self, in_dim=18, n_actions=5, hidden=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        feat = self.feature(x)
        v    = self.value(feat)
        a    = self.advantage(feat)
        return v + (a - a.mean(dim=1, keepdim=True))


_model:       Optional[DuelingDQN] = None
_dark_steps:  int                  = 0
_stuck_count: int                  = 0


def _load_once():
    global _model
    if _model is not None:
        return
    here  = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError(
            "weights.pth not found next to agent.py."
        )
    m  = DuelingDQN(in_dim=18, n_actions=5, hidden=128)
    sd = torch.load(wpath, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    m.load_state_dict(sd, strict=True)
    m.eval()
    _model = m


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _dark_steps, _stuck_count
    _load_once()

    # Priority 1: stuck → rotate to escape wall
    if obs[17] == 1:
        _stuck_count += 1
        _dark_steps   = 0
        return "L45" if _stuck_count % 2 == 0 else "R45"

    _stuck_count = 0

    # Priority 2: IR fires → box directly ahead, very close → go forward
    if obs[16] == 1:
        _dark_steps = 0
        return "FW"

    # Priority 3: any sonar sensor active → network decides
    if any(obs[:16]):
        _dark_steps = 0
        x  = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        qs = _model(x).squeeze(0).numpy()
        return ACTIONS[int(np.argmax(qs))]

    # Priority 4: all sensors dark → rotate to scan for box
    _dark_steps += 1
    return "L45" if (_dark_steps // 8) % 2 == 0 else "R45"
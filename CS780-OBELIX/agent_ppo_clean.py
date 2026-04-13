"""PPO agent for OBELIX — evaluation only.
Loads pretrained weights from weights.pth.
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
from torch.distributions import Categorical

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]


class ActorCritic(nn.Module):
    def __init__(self, in_dim=18, n_actions=5, hidden=128):
        super().__init__()
        self.trunk  = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor  = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.actor(h), self.critic(h).squeeze(-1)


_model:          Optional[ActorCritic] = None
_stuck_count:    int                   = 0
_recovery_steps: int                   = 0


def _load_once():
    global _model
    if _model is not None:
        return
    here  = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError("weights.pth not found next to agent.py.")
    m  = ActorCritic()
    sd = torch.load(wpath, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    m.load_state_dict(sd, strict=True)
    m.eval()
    _model = m


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _stuck_count, _recovery_steps
    _load_once()

    # stuck recovery
    if obs[17] > 0:
        _stuck_count    += 1
        _recovery_steps  = 15
    if _recovery_steps > 0:
        _recovery_steps -= 1
        if _recovery_steps < 3:
            return "FW"
        return "L45" if (_stuck_count // 3) % 2 == 0 else "R45"

    # sensor override
    if obs[16] > 0:            return "FW"
    if any(obs[4:12] > 0):     return "FW"
    if any(obs[0:4]  > 0):     return "L22"
    if any(obs[12:16]> 0):     return "R22"

    # network decision when all dark
    x      = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    logits, _ = _model(x)
    best   = int(torch.argmax(logits, dim=1))
    return ACTIONS[best]
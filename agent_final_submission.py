"""PPO agent for OBELIX final phase.

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
import torch.nn.functional as F

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)


class ActorCritic(nn.Module):
    def __init__(self, in_dim=18, n_actions=5, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor  = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        feat   = self.shared(x)
        logits = self.actor(feat)
        value  = self.critic(feat).squeeze(-1)
        return logits, value

    def selectGreedyAction(self, s):
        with torch.no_grad():
            logits, _ = self.forward(s)
        return int(torch.argmax(logits, dim=-1).item())


_model:      Optional[ActorCritic] = None
_dark_steps: int                   = 0
_dark_dir:   int                   = 0


def _load_once():
    global _model
    if _model is not None:
        return
    here  = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError(
            "weights.pth not found next to agent.py. Train offline and include it in the submission zip."
        )
    m  = ActorCritic(in_dim=18, n_actions=N_ACTIONS, hidden=128)
    sd = torch.load(wpath, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    m.load_state_dict(sd, strict=True)
    m.eval()
    _model = m


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _dark_steps, _dark_dir

    _load_once()
    obs = np.array(obs, dtype=np.float32)

    # Priority 1 — stuck against wall
    if obs[17] == 1:
        _dark_steps = 0
        return "L45" if rng.random() < 0.5 else "R45"

    # Priority 2 — IR fires, box directly ahead
    if obs[16] == 1:
        _dark_steps = 0
        return "FW"

    # Priority 3 — any sonar active, trust network
    if np.any(obs[:16] == 1):
        _dark_steps = 0
        s = torch.tensor(obs).unsqueeze(0)
        a = _model.selectGreedyAction(s)
        return ACTIONS[a]

    # Priority 4 — all sensors dark, systematic rotation
    _dark_steps += 1
    phase = _dark_steps % 16
    if phase < 8:
        return "L45" if (_dark_dir % 2 == 0) else "R45"
    else:
        return "R45" if (_dark_dir % 2 == 0) else "L45"
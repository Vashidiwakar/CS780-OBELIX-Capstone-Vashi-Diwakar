"""PPO agent for OBELIX Final Phase (CPU).

Based on agent that scored -1609.51 on final phase Codabench.
Same weights (weights_ppo_phase2.pth), improved policy logic:
  - L22/R22 instead of L45/R45 for dark rotation (finer grain, less overshoot)
  - Sensor persistence: keeps moving toward box for 3+ steps after sensor fires
  - Forward sonars force FW directly
  - Side sonars turn toward box then persist FW
  - Dark dir changes every 200 steps to cover full arena

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


_model:         Optional[ActorCritic] = None
_dark_steps:    int                   = 0
_dark_dir:      int                   = 0
_fw_persist:    int                   = 0
_episode_steps: int                   = 0


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
    m  = ActorCritic(in_dim=18, n_actions=N_ACTIONS, hidden=128)
    sd = torch.load(wpath, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    m.load_state_dict(sd, strict=True)
    m.eval()
    _model = m


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _dark_steps, _dark_dir, _fw_persist, _episode_steps
    _load_once()

    obs = np.array(obs, dtype=np.float32)
    _episode_steps += 1

    # change sweep direction every 200 steps to cover full arena
    if _episode_steps % 200 == 0:
        _dark_dir += 1

    # Priority 1 — stuck against wall or boundary
    if obs[17] == 1:
        _dark_steps = 0
        _fw_persist = 0
        return "L22" if rng.random() < 0.5 else "R22"

    # Priority 2 — IR fires, box directly ahead
    if obs[16] == 1:
        _dark_steps = 0
        _fw_persist = 5
        return "FW"

    # Priority 3 — sensor persistence
    if _fw_persist > 0:
        _fw_persist -= 1
        return "FW"

    # Priority 4 — forward sonars active
    if np.any(obs[4:12] == 1):
        _dark_steps = 0
        _fw_persist = 3
        return "FW"

    # Priority 5 — left sonars active
    if np.any(obs[0:4] == 1):
        _dark_steps = 0
        _fw_persist = 2
        return "L22"

    # Priority 6 — right sonars active
    if np.any(obs[12:16] == 1):
        _dark_steps = 0
        _fw_persist = 2
        return "R22"

    # Priority 7 — all dark, fine-grain sweep with L22/R22
    _dark_steps += 1
    phase = _dark_steps % 24
    if phase < 12:
        return "L22" if (_dark_dir % 2 == 0) else "R22"
    else:
        return "R22" if (_dark_dir % 2 == 0) else "L22"
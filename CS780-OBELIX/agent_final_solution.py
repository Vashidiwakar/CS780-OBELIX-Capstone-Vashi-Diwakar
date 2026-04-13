"""PPO agent for OBELIX Final Phase — Final Solution.

Key innovation: observation history stacking (4 frames).
Instead of 18-bit obs, uses 72-bit obs (4 × 18).
This gives the network temporal context to track moving box.

Submission ZIP structure:
  submission.zip
    agent.py
    weights.pth
"""

from __future__ import annotations
from typing import List, Optional
from collections import deque
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)
OBS_DIM   = 18
HIST_LEN  = 4
AUG_DIM   = OBS_DIM * HIST_LEN


class ActorCritic(nn.Module):
    def __init__(self, in_dim=AUG_DIM, n_actions=5, hidden=128):
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


_model:       Optional[ActorCritic] = None
_dark_steps:  int                   = 0
_dark_dir:    int                   = 0
_fw_persist:  int                   = 0
_ep_steps:    int                   = 0
_obs_history: Optional[deque]       = None


def _load_once():
    global _model
    if _model is not None:
        return
    here  = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError("weights.pth not found next to agent.py.")
    m  = ActorCritic(in_dim=AUG_DIM, hidden=128)
    sd = torch.load(wpath, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    m.load_state_dict(sd, strict=True)
    m.eval()
    _model = m


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _dark_steps, _dark_dir, _fw_persist, _ep_steps, _obs_history
    _load_once()

    obs = np.array(obs, dtype=np.float32)
    _ep_steps += 1

    # initialize or update observation history
    if _obs_history is None:
        _obs_history = deque(
            [np.zeros(OBS_DIM, dtype=np.float32)] * HIST_LEN,
            maxlen=HIST_LEN
        )
    _obs_history.append(obs.copy())
    aug_obs = np.concatenate(list(_obs_history), axis=0)

    # change sweep direction every 200 steps
    if _ep_steps % 200 == 0:
        _dark_dir += 1

    # Priority 1 — stuck
    if obs[17] == 1:
        _dark_steps = 0
        _fw_persist = 0
        return "L45" if rng.random() < 0.5 else "R45"

    # Priority 2 — IR fires
    if obs[16] == 1:
        _dark_steps = 0
        _fw_persist = 5
        return "FW"

    # Priority 3 — sensor persistence
    if _fw_persist > 0:
        _fw_persist -= 1
        return "FW"

    # Priority 4 — forward sonars
    if np.any(obs[4:12] == 1):
        _dark_steps = 0
        _fw_persist = 3
        return "FW"

    # Priority 5 — any sonar active, ask network with history
    if np.any(obs[:16] == 1):
        _dark_steps = 0
        s = torch.tensor(aug_obs).unsqueeze(0)
        a = _model.selectGreedyAction(s)
        return ACTIONS[a]

    # Priority 6 — all dark, systematic rotation
    _dark_steps += 1
    phase = _dark_steps % 16
    if phase < 8:
        return "L45" if (_dark_dir % 2 == 0) else "R45"
    else:
        return "R45" if (_dark_dir % 2 == 0) else "L45"
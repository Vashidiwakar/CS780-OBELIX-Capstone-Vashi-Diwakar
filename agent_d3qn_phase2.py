"""Dueling Double DQN agent for OBELIX Phase 2 — blinking box (CPU).

Evaluation-only: loads pretrained weights from weights.pth placed
next to agent.py inside the submission zip.

Submission ZIP structure:
  submission.zip
    agent.py
    weights.pth

Codabench evaluation settings (from evaluate.py):
  scaling_factor=5, max_steps=1000, wall_obstacles=True
  difficulty tested: 0, 2, 3
"""

from __future__ import annotations
from typing import List, Optional
import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)


class DuelingDQN(nn.Module):
    def __init__(self, in_dim=18, n_actions=5):
        super().__init__()
        # Shared feature extractor — must match train_phase2.py exactly
        self.feature = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        # Value stream: V(s) -> scalar
        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # Advantage stream: A(s,a) -> one value per action
        self.advantage = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        feat = self.feature(x)
        v    = self.value(feat)
        a    = self.advantage(feat)
        return v + (a - a.mean(dim=1, keepdim=True))


_model:        Optional[DuelingDQN] = None
_dark_steps:   int                  = 0
_dark_dir:     int                  = 0
_episode_count: int                 = 0


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
    m  = DuelingDQN()
    sd = torch.load(wpath, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    m.load_state_dict(sd, strict=True)
    m.eval()
    _model = m


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """
    Greedy policy with smart exploration overrides.

    Priority chain:
      1. Stuck (obs[17]=1)  → rotate to escape wall
      2. IR sensor (obs[16]=1) → FW (box directly ahead, attach)
      3. Any sonar active   → greedy network action
      4. All sensors dark   → systematic rotation to find box
    """
    global _dark_steps, _dark_dir, _episode_count

    _load_once()

    obs = np.array(obs, dtype=np.float32)

    # Priority 1 — stuck against wall, rotate to escape
    if obs[17] == 1:
        _dark_steps = 0
        # Alternate direction to avoid bouncing on same wall
        return "L45" if (_episode_count % 2 == 0) else "R45"

    # Priority 2 — IR fires, box directly ahead, always go forward
    if obs[16] == 1:
        _dark_steps = 0
        return "FW"

    # Priority 3 — any sonar sensor active, trust the network
    if np.any(obs[:16] == 1):
        _dark_steps = 0
        x  = torch.tensor(obs).unsqueeze(0)
        qs = _model(x).squeeze(0).numpy()
        return ACTIONS[int(np.argmax(qs))]

    # Priority 4 — all sensors dark (box not visible or too far)
    # Systematic rotation: alternate L45/R45 every 8 steps
    # This sweeps the arena without going forward (avoids wall crashes)
    _dark_steps += 1
    phase = _dark_steps % 16
    if phase < 8:
        return "L45" if (_dark_dir % 2 == 0) else "R45"
    else:
        return "R45" if (_dark_dir % 2 == 0) else "L45"
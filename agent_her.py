"""
HER Agent for OBELIX — evaluation only.
Network input: obs(18) + goal_hint(2) = 20-dim
"""
from __future__ import annotations
import math
import os
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]
N_ACT = len(ACTIONS)
FW_IDX = 2
ROT_IDX = {0, 1, 3, 4}

class DuelingDQN(nn.Module):
    def __init__(self, in_dim: int = 20, hidden: int = 128, n_actions: int = N_ACT):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
        self.adv   = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, N_ACT))

    def forward(self, x):
        f = self.feature(x)
        v = self.value(f)
        a = self.adv(f)
        return v + a - a.mean(dim=-1, keepdim=True)

def obs_to_goal_hint(obs: np.ndarray) -> np.ndarray:
    left_score    = float(np.sum(obs[0:4]))
    forward_score = float(np.sum(obs[4:12])) * 1.5
    right_score   = float(np.sum(obs[12:16]))
    ir_score      = float(obs[16]) * 3.0
    hx = right_score - left_score
    hy = forward_score + ir_score
    norm = math.sqrt(hx * hx + hy * hy) + 1e-8
    return np.array([hx / norm, hy / norm], dtype=np.float32)

_model:          Optional[DuelingDQN] = None
_dark_steps:     int = 0
_dark_dir:       int = 0
_recovery_steps: int = 0
_stuck_count:    int = 0
_repeat_count:   int = 0
_last_action:    Optional[int] = None

def _load_once():
    global _model
    if _model is not None:
        return
    here  = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")
    m = DuelingDQN(in_dim=20, hidden=128)
    sd = torch.load(wpath, map_location="cpu", weights_only=False)
    m.load_state_dict(sd, strict=True)
    m.eval()
    _model = m

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _dark_steps, _dark_dir, _recovery_steps, _stuck_count
    global _repeat_count, _last_action
    _load_once()

    obs = np.array(obs, dtype=np.float32)

    # Priority 1: stuck recovery
    if obs[17] > 0:
        _stuck_count    += 1
        _recovery_steps  = 20
        _repeat_count    = 0
        _last_action     = None
        _dark_steps      = 0

    if _recovery_steps > 0:
        _recovery_steps -= 1
        if _recovery_steps < 3:
            return "FW"   # move away after rotating
        return "L45" if (_stuck_count // 3) % 2 == 0 else "R45"

    # Priority 2: IR fires — box directly ahead
    if obs[16] > 0:
        _dark_steps  = 0
        _repeat_count = 0
        return "FW"

    # Priority 3: any sonar active — use network with goal hint
    if np.any(obs[:16] > 0):
        _dark_steps = 0
        g   = obs_to_goal_hint(obs)
        inp = np.concatenate([obs, g]).astype(np.float32)
        with torch.no_grad():
            qs = _model(torch.tensor(inp).unsqueeze(0)).squeeze(0).numpy()
        # Forward bias
        qs[FW_IDX] += 3.0 if obs[16] > 0 else (2.0 if any(obs[4:12] > 0) else 0.5)
        best = int(np.argmax(qs))
        # Anti-spin
        if _last_action is not None and best == _last_action and best in ROT_IDX:
            _repeat_count += 1
            if _repeat_count >= 4:
                best = FW_IDX
                _repeat_count = 0
        else:
            _repeat_count = 0
        _last_action = best
        return ACTIONS[best]

    # Priority 4: all dark — systematic sweep
    _dark_steps += 1
    phase = _dark_steps % 16
    if phase < 8:
        return "L45" if (_dark_dir % 2 == 0) else "R45"
    else:
        if phase == 8:
            _dark_dir += 1
        return "R45" if ((_dark_dir - 1) % 2 == 0) else "L45"
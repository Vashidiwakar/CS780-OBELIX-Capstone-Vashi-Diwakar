"""A2C agent for OBELIX (CPU).
Evaluation-only: loads pretrained weights from weights.pth placed
next to agent.py inside the submission zip.
Submission ZIP structure:
  submission.zip
    agent.py       <- rename this file to agent.py
    weights.pth    <- rename weights_a2c.pth to weights.pth
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
    def __init__(self, inDim=18, outDim=5, hiddenDim=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(inDim, hiddenDim),
            nn.ReLU(),
            nn.Linear(hiddenDim, hiddenDim),
            nn.ReLU(),
        )
        self.actor  = nn.Linear(hiddenDim, outDim)
        self.critic = nn.Linear(hiddenDim, 1)

    def forward(self, s):
        feat   = self.shared(s)
        logits = self.actor(feat)
        value  = self.critic(feat).squeeze(-1)
        return logits, value

    def selectGreedyAction(self, s):
        """Greedy action — used at evaluation time."""
        with torch.no_grad():
            logits, _ = self.forward(s)
        return int(torch.argmax(logits, dim=-1).item())


_model: Optional[ActorCritic] = None


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
    m  = ActorCritic(inDim=18, outDim=N_ACTIONS, hiddenDim=64)
    sd = torch.load(wpath, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    m.load_state_dict(sd, strict=True)
    m.eval()
    _model = m


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load_once()

    # obs[17] = stuck flag — rotate to escape wall immediately
    if obs[17] == 1:
        return rng.choice(["L45", "R45"])

    # obs[16] = IR sensor — box is directly ahead and very close, go forward
    if obs[16] == 1:
        return "FW"

    # Network decides everything else
    s = torch.tensor(np.array(obs, dtype=np.float32))
    a = _model.selectGreedyAction(s)
    return ACTIONS[a]
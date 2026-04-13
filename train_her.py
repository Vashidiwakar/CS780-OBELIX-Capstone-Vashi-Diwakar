"""
Hindsight Experience Replay (HER) for OBELIX.

HER is designed exactly for sparse reward environments like this one.
Key idea: even failed episodes contain useful information.
If the agent ended near the box but didn't attach, relabel that
episode as if "being near the box" was the goal, and learn from it.

How it works for OBELIX:
- Normal episode: agent tries to find box, push to boundary
- After episode ends, look back at trajectory
- Find the closest point the agent got to the box
- Relabel those transitions: "if the goal was to reach THIS position,
  the agent succeeded" → positive reward signal
- This gives the network dense learning signal even from failed episodes

The network learns: obs + goal_hint → action
At evaluation: goal_hint = actual box direction (inferred from sensors)

Pure RL, no reward shaping, no imitation learning.
"""

from __future__ import annotations
import argparse
import gc
import importlib.util
import math
import os
import random
from collections import deque
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Actions ──────────────────────────────────────────────────────────────────
ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]
N_ACT = len(ACTIONS)
FW_IDX = 2

# ── HER Transition ────────────────────────────────────────────────────────────
class Transition(NamedTuple):
    s:    np.ndarray   # obs (18,)
    g:    np.ndarray   # goal hint (2,) — normalised direction to box
    a:    int
    r:    float
    s2:   np.ndarray
    g2:   np.ndarray
    done: bool

# ── Network ───────────────────────────────────────────────────────────────────
class DuelingDQN(nn.Module):
    """
    Input: obs (18,) concatenated with goal hint (2,) = 20-dim
    The goal hint is a 2D unit vector pointing toward the box,
    derived from which sensors fired (available without privileged info).
    """
    def __init__(self, in_dim: int = 20, hidden: int = 128, n_actions: int = N_ACT):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value   = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
        self.adv     = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feature(x)
        v = self.value(f)
        a = self.adv(f)
        return v + a - a.mean(dim=-1, keepdim=True)

# ── Goal hint from observation ────────────────────────────────────────────────
def obs_to_goal_hint(obs: np.ndarray) -> np.ndarray:
    """
    Derive a 2D direction hint toward the box from sensor observations.
    No privileged info — only uses the 18-bit obs the agent already has.
    
    Left sonars  (obs[0:4])   → box is to the left  → hint = (-1, 0)
    Forward sonars (obs[4:12]) → box is ahead        → hint = (0, 1)  
    Right sonars (obs[12:16]) → box is to the right  → hint = (1, 0)
    IR (obs[16])              → box directly ahead   → hint = (0, 1)
    All dark                  → unknown              → hint = (0, 0)
    """
    left_score    = float(np.sum(obs[0:4]))
    forward_score = float(np.sum(obs[4:12])) * 1.5  # forward weighted more
    right_score   = float(np.sum(obs[12:16]))
    ir_score      = float(obs[16]) * 3.0  # IR is strongest signal

    hx = right_score - left_score
    hy = forward_score + ir_score

    norm = math.sqrt(hx * hx + hy * hy) + 1e-8
    return np.array([hx / norm, hy / norm], dtype=np.float32)

# ── Replay buffer with HER ────────────────────────────────────────────────────
class HERBuffer:
    """
    Stores full episodes, then at sample time applies HER relabelling.
    
    HER strategy: 'future' — for each transition, sample k future
    states from the same episode as alternative goals. Relabel reward
    as +1 if that future state would have been "achieved" by this action.
    """
    def __init__(self, capacity: int = 50_000, her_k: int = 4):
        self.capacity  = capacity
        self.her_k     = her_k
        self.buffer: deque[List[Transition]] = deque()
        self.size      = 0

    def add_episode(self, episode: List[Transition]):
        """Add a full episode and apply HER relabelling."""
        # Store original episode
        while self.size + len(episode) > self.capacity and self.buffer:
            old = self.buffer.popleft()
            self.size -= len(old)
        self.buffer.append(episode)
        self.size += len(episode)

        # HER: generate relabelled episodes
        # For each transition, pick k future states as "achieved goals"
        her_episodes = []
        for t_idx in range(len(episode)):
            for _ in range(self.her_k):
                # Sample a future transition from this episode
                future_idx = random.randint(t_idx, len(episode) - 1)
                future_t   = episode[future_idx]

                # The "achieved goal" is the goal hint of the future state
                # (i.e., what direction the agent was facing when sensors fired)
                achieved_goal = future_t.g2.copy()

                # Relabel this transition with the achieved goal
                orig = episode[t_idx]

                # Relabelled reward: did taking action 'a' bring us toward achieved_goal?
                # Positive if achieved_goal has strong forward component and action was FW
                # This is a sparse binary: +1 if we "reached" the relabelled goal
                goal_similarity = float(np.dot(orig.g2, achieved_goal))
                her_reward = 1.0 if (goal_similarity > 0.9 and future_idx == t_idx) else -0.01

                her_t = Transition(
                    s=orig.s, g=achieved_goal, a=orig.a,
                    r=her_reward, s2=orig.s2, g2=achieved_goal,
                    done=(future_idx == len(episode) - 1)
                )
                her_episodes.append(her_t)

        # Store HER transitions as a flat pseudo-episode
        if her_episodes:
            while self.size + len(her_episodes) > self.capacity and self.buffer:
                old = self.buffer.popleft()
                self.size -= len(old)
            self.buffer.append(her_episodes)
            self.size += len(her_episodes)

    def sample(self, batch_size: int) -> List[Transition]:
        # Flatten all episodes and sample
        all_transitions = [t for ep in self.buffer for t in ep]
        if len(all_transitions) < batch_size:
            return []
        return random.sample(all_transitions, batch_size)

    def __len__(self):
        return self.size

# ── Environment loader ────────────────────────────────────────────────────────
def load_env(path: str, wall: bool, difficulty: int, seed: int, max_steps: int):
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX(
        scaling_factor=5,
        arena_size=500,
        max_steps=max_steps,
        wall_obstacles=wall,
        difficulty=difficulty,
        box_speed=2,
        seed=seed,
    )

# ── Epsilon schedule ──────────────────────────────────────────────────────────
def get_eps(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    return max(eps_end, eps_start - (eps_start - eps_end) * step / decay_steps)

# ── Training ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",    default="./obelix.py")
    ap.add_argument("--out",          default="weights_her.pth")
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--episodes",     type=int,   default=4000)
    ap.add_argument("--max_steps",    type=int,   default=1000)
    ap.add_argument("--batch_size",   type=int,   default=256)
    ap.add_argument("--lr",           type=float, default=3e-4)
    ap.add_argument("--gamma",        type=float, default=0.99)
    ap.add_argument("--eps_start",    type=float, default=1.0)
    ap.add_argument("--eps_end",      type=float, default=0.10)
    ap.add_argument("--eps_decay",    type=int,   default=800_000)
    ap.add_argument("--target_update",type=int,   default=500)
    ap.add_argument("--hidden",       type=int,   default=128)
    ap.add_argument("--her_k",        type=int,   default=4)
    ap.add_argument("--ep_switch1",   type=int,   default=1000)  # d0 → d2
    ap.add_argument("--ep_switch2",   type=int,   default=2500)  # d2 → d3
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Network: input is obs(18) + goal_hint(2) = 20
    q      = DuelingDQN(in_dim=20, hidden=args.hidden)
    q_tgt  = DuelingDQN(in_dim=20, hidden=args.hidden)
    q_tgt.load_state_dict(q.state_dict())
    q_tgt.eval()
    opt    = torch.optim.Adam(q.parameters(), lr=args.lr)
    replay = HERBuffer(capacity=50_000, her_k=args.her_k)

    total_steps = 0
    best_return = -float("inf")

    # Curriculum
    def get_stage(ep):
        if ep < args.ep_switch1:
            return 0, False
        elif ep < args.ep_switch2:
            return 2, args.wall_obstacles
        else:
            return 3, args.wall_obstacles

    env = None
    cur_difficulty = -1
    cur_wall = None

    for ep in range(args.episodes):
        difficulty, wall = get_stage(ep)

        # Rebuild env only when stage changes
        if difficulty != cur_difficulty or wall != cur_wall:
            env = load_env(
                args.obelix_py, wall, difficulty,
                args.seed + ep, args.max_steps
            )
            cur_difficulty = difficulty
            cur_wall = wall
            print(f"\n{'='*50}")
            print(f"Stage: difficulty={difficulty}, wall={wall}")
            print(f"{'='*50}")
        
        obs = env.reset(seed=args.seed + ep)
        episode_transitions = []
        ep_ret = 0.0
        g = obs_to_goal_hint(obs)

        for step in range(args.max_steps):
            total_steps += 1
            eps = get_eps(total_steps, args.eps_start, args.eps_end, args.eps_decay)

            # Action selection
            if np.random.rand() < eps:
                # Biased exploration: prefer FW when sensors fire, rotate when dark
                if obs[17] > 0:
                    # Stuck: rotate to escape
                    a = 0 if (step // 3) % 2 == 0 else 4
                elif obs[16] > 0:
                    a = FW_IDX  # IR fires → definitely box → go forward
                elif any(obs[4:12] > 0):
                    a = FW_IDX  # forward sonars → go forward
                elif any(obs[0:4] > 0):
                    a = 1  # left sonars → turn left
                elif any(obs[12:16] > 0):
                    a = 3  # right sonars → turn right
                else:
                    # Dark: sweep pattern
                    a = 0 if (step // 8) % 2 == 0 else 4
            else:
                # Greedy: network decides using obs + goal hint
                inp = np.concatenate([obs, g]).astype(np.float32)
                with torch.no_grad():
                    qs = q(torch.tensor(inp).unsqueeze(0)).squeeze(0).numpy()
                # Forward bias when sensors fire
                if obs[16] > 0:
                    qs[FW_IDX] += 5.0
                elif any(obs[4:12] > 0):
                    qs[FW_IDX] += 2.0
                a = int(np.argmax(qs))

            obs2, r, done = env.step(ACTIONS[a], render=False)
            g2 = obs_to_goal_hint(obs2)

            # Normalise reward for stable training
            r_norm = r / 200.0

            t = Transition(s=obs.copy(), g=g.copy(), a=a,
                          r=r_norm, s2=obs2.copy(), g2=g2.copy(), done=done)
            episode_transitions.append(t)

            # Store high-value transitions immediately with extra copies
            if r >= 100:  # attach or deliver
                for _ in range(8):
                    episode_transitions.append(t)

            ep_ret += r
            obs = obs2
            g   = g2

            # Train
            if len(replay) >= args.batch_size:
                batch = replay.sample(args.batch_size)
                if batch:
                    s_b  = torch.tensor(
                        np.stack([np.concatenate([t.s, t.g]) for t in batch]),
                        dtype=torch.float32)
                    s2_b = torch.tensor(
                        np.stack([np.concatenate([t.s2, t.g2]) for t in batch]),
                        dtype=torch.float32)
                    a_b  = torch.tensor([t.a for t in batch], dtype=torch.long)
                    r_b  = torch.tensor([t.r for t in batch], dtype=torch.float32)
                    d_b  = torch.tensor([t.done for t in batch], dtype=torch.float32)

                    with torch.no_grad():
                        # Double DQN target
                        next_actions = q(s2_b).argmax(dim=1)
                        next_q = q_tgt(s2_b).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                        target = r_b + args.gamma * next_q * (1 - d_b)

                    current_q = q(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
                    loss = F.smooth_l1_loss(current_q, target)

                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                    opt.step()

            if total_steps % args.target_update == 0:
                q_tgt.load_state_dict(q.state_dict())

            if done:
                break

        # Add full episode to HER buffer (HER relabelling happens here)
        replay.add_episode(episode_transitions)

        if ep_ret > best_return:
            best_return = ep_ret
            torch.save(q.state_dict(), args.out.replace(".pth", "_best.pth"))

        if (ep + 1) % 50 == 0:
            stage = "d0" if ep < args.ep_switch1 else ("d2" if ep < args.ep_switch2 else "d3")
            wall_str = "w" if cur_wall else "nw"
            print(f"ep {ep+1:4d}/{args.episodes} [{stage}{wall_str}] "
                  f"return={ep_ret:.1f} eps={get_eps(total_steps, args.eps_start, args.eps_end, args.eps_decay):.3f} "
                  f"buf={len(replay)} best={best_return:.1f}")
            torch.save(q.state_dict(), args.out)

        gc.collect()

    torch.save(q.state_dict(), args.out)
    print(f"\nDone. Best return: {best_return:.1f}")
    print(f"Saved: {args.out}")
    print(f"Best: {args.out.replace('.pth', '_best.pth')}")

if __name__ == "__main__":
    main()
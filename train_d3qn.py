"""Offline trainer: Dueling Double DQN for OBELIX (CPU).

The best-performing algorithm for OBELIX based on empirical testing.
Run locally to create weights.pth, then submit agent_best.py + weights.pth.

Example:
  python train_best.py --obelix_py ./obelix.py --out weights_best.pth
  python train_best.py --obelix_py ./obelix.py --out weights_best.pth --wall_obstacles


                ALGORITHM: DUELING DOUBLE DQN (Best for OBELIX)

Why Dueling DDQN beats other algorithms here:
  - Replay buffer smooths out bad episodes (A2C/REINFORCE suffer from variance)
  - Target network stabilises learning (no critic explosion like A2C)
  - Off-policy: reuses good experiences from replay buffer many times
  - Dueling streams: V(s) learns "dark states are bad" from every transition
    regardless of action — faster learning with sparse sensor signals

Why not policy gradients (REINFORCE/A2C/PPO):
  - On-policy: one bad episode overwrites good learning
  - Complete episode required before any update — slow credit assignment
  - OBELIX reward variance is too high for stable policy gradient updates

Key design decisions:
  - scaling_factor=3: sensors cover 3x more area, box is easier to find
  - stuck/IR overrides during training: hard rules for obvious situations
  - Dark exploration: systematic L45/R45 sweep when sensors all off
  - eps_end=0.15: keep 15% exploration permanently to keep finding box
  - Smaller replay buffer (50k): less RAM, fills faster, learning starts sooner

References:
  Wang et al. 2016  https://arxiv.org/pdf/1511.07401  (Dueling DQN)
  Van Hasselt 2016  https://arxiv.org/pdf/1509.06461  (Double DQN)
"""

from __future__ import annotations
import argparse, random, gc
from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


class DuelingDQN(nn.Module):
    def __init__(self, in_dim=18, n_actions=5, hidden=128):
        super().__init__()
        # Shared feature trunk — wider than before (128 vs 64)
        self.feature = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # Value stream: V(s) — how good is this state regardless of action
        self.value = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # Advantage stream: A(s,a) — how much better is action a vs average
        self.advantage = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        feat = self.feature(x)
        v    = self.value(feat)                        # (B, 1)
        a    = self.advantage(feat)                    # (B, n_actions)
        return v + (a - a.mean(dim=1, keepdim=True))  # Q(s,a)


@dataclass
class Transition:
    s:    np.ndarray
    a:    int
    r:    float
    s2:   np.ndarray
    done: bool


class Replay:
    def __init__(self, cap: int = 50_000):
        self.buf: Deque[Transition] = deque(maxlen=cap)

    def add(self, t: Transition):
        self.buf.append(t)

    def sample(self, batch: int):
        idx   = np.random.choice(len(self.buf), size=batch, replace=False)
        items = [self.buf[i] for i in idx]
        s  = np.stack([it.s  for it in items]).astype(np.float32)
        a  = np.array([it.a  for it in items], dtype=np.int64)
        r  = np.array([it.r  for it in items], dtype=np.float32)
        s2 = np.stack([it.s2 for it in items]).astype(np.float32)
        d  = np.array([it.done for it in items], dtype=np.float32)
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buf)


def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",       type=str,   required=True)
    ap.add_argument("--out",             type=str,   default="weights_best.pth")
    ap.add_argument("--episodes",        type=int,   default=6000)
    ap.add_argument("--max_steps",       type=int,   default=1000)
    ap.add_argument("--difficulty",      type=int,   default=0)
    ap.add_argument("--wall_obstacles",  action="store_true")
    ap.add_argument("--box_speed",       type=int,   default=2)
    ap.add_argument("--scaling_factor",  type=int,   default=3)
    ap.add_argument("--arena_size",      type=int,   default=500)

    ap.add_argument("--gamma",           type=float, default=0.99)
    ap.add_argument("--lr",              type=float, default=3e-4)
    ap.add_argument("--hidden",          type=int,   default=128)
    ap.add_argument("--batch",           type=int,   default=128)
    ap.add_argument("--replay",          type=int,   default=50_000)
    ap.add_argument("--warmup",          type=int,   default=2000)
    ap.add_argument("--target_sync",     type=int,   default=500)
    ap.add_argument("--eps_start",       type=float, default=1.0)
    ap.add_argument("--eps_end",         type=float, default=0.20)
    ap.add_argument("--eps_decay_steps", type=int,   default=500_000)
    ap.add_argument("--seed",            type=int,   default=3425)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    q   = DuelingDQN(hidden=args.hidden)
    tgt = DuelingDQN(hidden=args.hidden)
    tgt.load_state_dict(q.state_dict())
    tgt.eval()

    opt    = optim.Adam(q.parameters(), lr=args.lr)
    replay = Replay(args.replay)
    steps  = 0
    rng    = np.random.default_rng(args.seed)
    dark_steps = 0  # consecutive steps with no sensor signal

    def eps_by_step(t):
        if t >= args.eps_decay_steps:
            return args.eps_end
        frac = t / args.eps_decay_steps
        return args.eps_start + frac * (args.eps_end - args.eps_start)

    # Create env ONCE outside loop — prevents memory leak
    env = OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
        seed=args.seed,
    )

    for ep in range(args.episodes):
        obs    = np.array(env.reset(seed=args.seed + ep), dtype=np.float32)
        ep_ret = 0.0
        dark_steps = 0

        for _ in range(args.max_steps):
            eps = eps_by_step(steps)

            # Hard override: stuck → rotate to escape wall
            if obs[17] == 1:
                a = int(rng.choice([0, 4]))  # L45 or R45

            # Hard override: IR fires → box right there, go forward
            elif obs[16] == 1:
                a = 2  # FW
                dark_steps = 0

            # Dark exploration: no sensors active → systematic sweep
            elif not any(obs[:17]) and rng.random() < eps:
                dark_steps += 1
                a = 0 if (dark_steps // 8) % 2 == 0 else 4  # alternate L45/R45

            # Epsilon-greedy from network
            elif rng.random() < eps:
                a = int(rng.integers(len(ACTIONS)))
                dark_steps = 0

            else:
                with torch.no_grad():
                    qs = q(torch.tensor(obs).unsqueeze(0)).squeeze(0).numpy()
                a = int(np.argmax(qs))
                dark_steps = 0

            s2, r, done = env.step(ACTIONS[a], render=False)
            s2 = np.array(s2, dtype=np.float32)
            ep_ret += float(r)
            replay.add(Transition(s=obs, a=a, r=float(r), s2=s2, done=bool(done)))
            obs   = s2
            steps += 1

            # Learn
            if len(replay) >= max(args.warmup, args.batch):
                sb, ab, rb, s2b, db = replay.sample(args.batch)
                sb_t  = torch.tensor(sb)
                ab_t  = torch.tensor(ab)
                rb_t  = torch.tensor(rb)
                s2b_t = torch.tensor(s2b)
                db_t  = torch.tensor(db)

                with torch.no_grad():
                    # Double DQN: online selects, target evaluates
                    next_a   = torch.argmax(q(s2b_t), dim=1)
                    next_val = tgt(s2b_t).gather(1, next_a.unsqueeze(1)).squeeze(1)
                    y        = rb_t + args.gamma * (1.0 - db_t) * next_val

                pred = q(sb_t).gather(1, ab_t.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(pred, y)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                opt.step()

                if steps % args.target_sync == 0:
                    tgt.load_state_dict(q.state_dict())

            if done:
                break

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{args.episodes}  return={ep_ret:.1f}  eps={eps_by_step(steps):.3f}  replay={len(replay)}")
            torch.save(q.state_dict(), args.out)
            gc.collect()

    torch.save(q.state_dict(), args.out)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
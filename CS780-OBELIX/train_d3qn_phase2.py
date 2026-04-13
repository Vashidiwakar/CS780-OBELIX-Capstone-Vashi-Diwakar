"""Offline trainer: Dueling Double DQN for OBELIX Phase 2 (Blinking Box).

Run locally to create weights.pth, then submit agent.py + weights.pth.

Example:
  python train_phase2.py --obelix_py ./obelix.py --out weights_phase2.pth
  python train_phase2.py --obelix_py ./obelix.py --out weights_phase2.pth --wall_obstacles


                    ALGORITHM: DUELING DOUBLE DQN — PHASE 2 (BLINKING BOX)


Phase 2 adds a blinking box — the box randomly appears and disappears.
Key changes from Phase 1:

1. MULTI-TASK TRAINING — train on both difficulty=0 and difficulty=2
   simultaneously. The agent first learns to find/push the static box,
   then generalises to the blinking version. This prevents catastrophic
   forgetting and gives much better Phase 2 performance.

2. DIFFICULTY CURRICULUM:
   Episodes 0    to ep_switch  → difficulty=0 (static box, build base policy)
   Episodes ep_switch to end   → difficulty=2 (blinking box, fine-tune)

3. BLINKING BOX STRATEGY — when box disappears (all sensors go dark),
   the agent should KEEP MOVING in the last known direction rather than
   stopping. The replay buffer will teach this through experience.

4. SCALING FACTOR = 5 — matches Codabench evaluation exactly.
   Sonar far range = 150px, covers 30% of arena. Box detected easily.

5. WALL_OBSTACLES trained — Codabench evaluates with wall_obstacles=True.

Reference:
  Van Hasselt et al. 2016  https://arxiv.org/pdf/1509.06461
  Wang et al. 2016         https://arxiv.org/pdf/1511.07401
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
N_ACTIONS = len(ACTIONS)


class DuelingDQN(nn.Module):
    def __init__(self, in_dim=18, n_actions=5):
        super().__init__()
        # Shared feature extractor
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
        v    = self.value(feat)                          # (B, 1)
        a    = self.advantage(feat)                      # (B, n_actions)
        return v + (a - a.mean(dim=1, keepdim=True))     # Q(s,a)


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


def select_action(q, obs, eps, dark_steps, dark_dir):
    """
    Action selection with smart dark-state exploration.
    When all sensors dark: rotate systematically to find box.
    When sensors active: epsilon-greedy from network.
    """
    # Stuck override — escape wall immediately
    if obs[17] == 1:
        return random.choice([0, 4])  # L45 or R45

    # IR fires — box directly ahead, always go forward
    if obs[16] == 1:
        return 2  # FW

    # Any sensor active — epsilon greedy
    if np.any(obs[:16] == 1):
        if random.random() < eps:
            return random.randint(0, N_ACTIONS - 1)
        with torch.no_grad():
            qs = q(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
        return int(np.argmax(qs))

    # All sensors dark — systematic rotation to find box
    # Alternate L45/R45 every 8 steps
    phase = dark_steps % 16
    if phase < 8:
        return 0 if dark_dir % 2 == 0 else 4   # L45 or R45
    else:
        return 4 if dark_dir % 2 == 0 else 0   # opposite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",       type=str,   required=True)
    ap.add_argument("--out",             type=str,   default="weights_phase2.pth")
    ap.add_argument("--episodes",        type=int,   default=8000)
    ap.add_argument("--max_steps",       type=int,   default=1000)
    ap.add_argument("--wall_obstacles",  action="store_true")
    ap.add_argument("--scaling_factor",  type=int,   default=5)   # match Codabench
    ap.add_argument("--arena_size",      type=int,   default=500)
    ap.add_argument("--ep_switch",       type=int,   default=3000,
                    help="Episodes of difficulty=0 before switching to difficulty=2")

    ap.add_argument("--gamma",           type=float, default=0.99)
    ap.add_argument("--lr",              type=float, default=5e-4)
    ap.add_argument("--batch",           type=int,   default=128)
    ap.add_argument("--replay",          type=int,   default=50_000)
    ap.add_argument("--warmup",          type=int,   default=2000)
    ap.add_argument("--target_sync",     type=int,   default=500)
    ap.add_argument("--eps_start",       type=float, default=1.0)
    ap.add_argument("--eps_end",         type=float, default=0.15)
    ap.add_argument("--eps_decay_steps", type=int,   default=600_000)
    ap.add_argument("--seed",            type=int,   default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    q   = DuelingDQN()
    tgt = DuelingDQN()
    tgt.load_state_dict(q.state_dict())
    tgt.eval()

    opt    = optim.Adam(q.parameters(), lr=args.lr)
    replay = Replay(args.replay)
    steps  = 0

    def eps_by_step(t):
        if t >= args.eps_decay_steps:
            return args.eps_end
        frac = t / args.eps_decay_steps
        return args.eps_start + frac * (args.eps_end - args.eps_start)

    # Phase 1 env — difficulty=0, no wall (build base policy)
    env_d0 = OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=False,
        difficulty=0,
        box_speed=2,
        seed=args.seed,
    )

    # Phase 2 env — difficulty=2, with wall (blinking box, matches Codabench)
    env_d2 = OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=2,
        box_speed=2,
        seed=args.seed + 1,
    )

    for ep in range(args.episodes):
        # Curriculum — switch from static to blinking after ep_switch episodes
        env = env_d0 if ep < args.ep_switch else env_d2

        s      = np.array(env.reset(seed=args.seed + ep), dtype=np.float32)
        ep_ret = 0.0
        dark_steps = 0
        dark_dir   = ep % 4   # vary rotation direction each episode

        for _ in range(args.max_steps):
            eps = eps_by_step(steps)
            a   = select_action(q, s, eps, dark_steps, dark_dir)

            # Track dark steps for exploration
            if np.any(s[:16] == 1) or s[16] == 1:
                dark_steps = 0
            else:
                dark_steps += 1

            s2, r, done = env.step(ACTIONS[a], render=False)
            s2 = np.array(s2, dtype=np.float32)
            ep_ret += float(r)

            # Success boosting — store high-reward transitions multiple times
            t = Transition(s=s, a=a, r=float(r), s2=s2, done=bool(done))
            replay.add(t)
            if float(r) >= 100:    # box touch or boundary
                for _ in range(4):
                    replay.add(t)  # replay 4 extra times

            s      = s2
            steps += 1

            if len(replay) >= max(args.warmup, args.batch):
                sb, ab, rb, s2b, db = replay.sample(args.batch)
                sb_t  = torch.tensor(sb)
                ab_t  = torch.tensor(ab)
                rb_t  = torch.tensor(rb)
                s2b_t = torch.tensor(s2b)
                db_t  = torch.tensor(db)

                with torch.no_grad():
                    next_a   = torch.argmax(q(s2b_t), dim=1)
                    next_val = tgt(s2b_t).gather(1, next_a.unsqueeze(1)).squeeze(1)
                    y        = rb_t + args.gamma * (1.0 - db_t) * next_val

                pred = q(sb_t).gather(1, ab_t.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(pred, y)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()

                if steps % args.target_sync == 0:
                    tgt.load_state_dict(q.state_dict())

            if done:
                break

        phase = "d0" if ep < args.ep_switch else "d2"
        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{args.episodes} [{phase}] return={ep_ret:.1f} eps={eps_by_step(steps):.3f} replay={len(replay)}")
            torch.save(q.state_dict(), args.out)
            gc.collect()

    torch.save(q.state_dict(), args.out)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
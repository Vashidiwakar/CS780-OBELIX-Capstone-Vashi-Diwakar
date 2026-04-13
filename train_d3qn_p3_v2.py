"""Offline trainer: Dueling Double DQN — Phase 3 v2 (moving + blinking box + wall).

Curriculum:
  difficulty=0 for ep_switch1 episodes  → build base push policy
  difficulty=2 for ep_switch2 episodes  → adapt to blinking box
  difficulty=3 for remaining episodes   → moving + blinking box

Combined improvements:
  - push_repeat=15 (store successful pushes 15x in replay)
  - Left/right sonar steering bias (turn toward box then move)
  - Anti-spin threshold reduced to 2 repeats
  - Recovery steps increased to 12
  - Hidden size 256 for more capacity
  - Longer exploration (eps_decay_steps=1M, eps_end=0.15)
  - gc.collect() every episode

Example:
  python train_d3qn_p3_v2.py --obelix_py ./obelix.py --out weights_p3_v2.pth --wall_obstacles
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
FW_IDX  = ACTIONS.index("FW")
ROT_IDX = {0, 1, 3, 4}


class DuelingDQN(nn.Module):
    def __init__(self, in_dim=18, n_actions=5, hidden=256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        feat = self.feature(x)
        v    = self.value(feat)
        a    = self.advantage(feat)
        return v + (a - a.mean(dim=1, keepdim=True))


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
        s  = np.stack([it.s    for it in items]).astype(np.float32)
        a  = np.array([it.a    for it in items], dtype=np.int64)
        r  = np.array([it.r    for it in items], dtype=np.float32)
        s2 = np.stack([it.s2   for it in items]).astype(np.float32)
        d  = np.array([it.done for it in items], dtype=np.float32)
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buf)


def import_obelix(path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def greedy_action(qs: np.ndarray, obs: np.ndarray, last_a: int,
                  repeat: int, stuck: int, recovery: int):
    """Greedy action with countdown stuck recovery, sensor steering and anti-spin."""

    # if wall collision detected, start 12-step recovery burst
    if obs[17] > 0:
        new_stuck    = stuck + 1
        new_recovery = 12
        if (new_stuck // 3) % 2 == 0:
            return 0, 0, new_stuck, new_recovery   # L45
        else:
            return 4, 0, new_stuck, new_recovery   # R45

    # keep recovering even after obs[17] clears
    if recovery > 0:
        new_recovery = recovery - 1
        if (stuck // 3) % 2 == 0:
            return 0, 0, stuck, new_recovery   # L45
        else:
            return 4, 0, stuck, new_recovery   # R45

    q = qs.copy()

    # sensor steering bias
    if obs[16] > 0:             # IR sensor — box very close directly ahead
        q[FW_IDX] += 2.0
    elif any(obs[4:12] > 0):    # forward sonars — box ahead, move forward
        q[FW_IDX] += 0.5
    elif any(obs[0:4] > 0):     # left sonars — box to left, turn left
        q[1] += 0.5             # L22
    elif any(obs[12:16] > 0):   # right sonars — box to right, turn right
        q[3] += 0.5             # R22

    best = int(np.argmax(q))

    # anti-spin — same rotation 2 times in a row → force FW
    if last_a is not None and best == last_a and best in ROT_IDX:
        if repeat >= 2:
            return FW_IDX, 0, 0, 0
        return best, repeat + 1, 0, 0

    return best, 0, 0, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",       type=str,   required=True)
    ap.add_argument("--out",             type=str,   default="weights_p3_v2.pth")
    ap.add_argument("--episodes",        type=int,   default=4000)
    ap.add_argument("--max_steps",       type=int,   default=1000)
    ap.add_argument("--ep_switch1",      type=int,   default=1200)
    ap.add_argument("--ep_switch2",      type=int,   default=2500)
    ap.add_argument("--wall_obstacles",  action="store_true")
    ap.add_argument("--box_speed",       type=int,   default=2)
    ap.add_argument("--scaling_factor",  type=int,   default=5)
    ap.add_argument("--arena_size",      type=int,   default=500)

    ap.add_argument("--gamma",           type=float, default=0.99)
    ap.add_argument("--lr",              type=float, default=5e-4)
    ap.add_argument("--batch",           type=int,   default=128)
    ap.add_argument("--replay",          type=int,   default=50_000)
    ap.add_argument("--warmup",          type=int,   default=2000)
    ap.add_argument("--target_sync",     type=int,   default=1000)
    ap.add_argument("--eps_start",       type=float, default=1.0)
    ap.add_argument("--eps_end",         type=float, default=0.15)
    ap.add_argument("--eps_decay_steps", type=int,   default=1_000_000)
    ap.add_argument("--push_repeat",     type=int,   default=15)
    ap.add_argument("--hidden",          type=int,   default=256)
    ap.add_argument("--seed",            type=int,   default=0)
    ap.add_argument("--render",          action="store_true")
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

    def eps_by_step(t):
        if t >= args.eps_decay_steps:
            return args.eps_end
        frac = t / args.eps_decay_steps
        return args.eps_start + frac * (args.eps_end - args.eps_start)

    env_d0 = OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=False,          # d0 without wall first
        difficulty=0,
        box_speed=args.box_speed,
        seed=args.seed,
    )
    env_d2 = OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=2,
        box_speed=args.box_speed,
        seed=args.seed,
    )
    env_d3 = OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=3,
        box_speed=args.box_speed,
        seed=args.seed,
    )

    for ep in range(args.episodes):
        if ep < args.ep_switch1:
            env   = env_d0
            phase = "d0"
        elif ep < args.ep_switch2:
            env   = env_d2
            phase = "d2"
        else:
            env   = env_d3
            phase = "d3"

        s        = np.array(env.reset(seed=args.seed + ep), dtype=np.float32)
        ep_ret   = 0.0
        last_a   = None
        repeat   = 0
        stuck    = 0
        recovery = 0

        for _ in range(args.max_steps):
            eps = eps_by_step(steps)

            if np.random.rand() < eps:
                a        = np.random.randint(len(ACTIONS))
                last_a   = None
                repeat   = 0
                stuck    = 0
                recovery = 0
            else:
                with torch.no_grad():
                    qs = q(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                a, repeat, stuck, recovery = greedy_action(qs, s, last_a, repeat, stuck, recovery)
                last_a = a

            s2, r, done = env.step(ACTIONS[a], render=args.render)
            ep_ret += float(r)

            t = Transition(s=s, a=a, r=float(r), s2=s2, done=bool(done))
            replay.add(t)

            # store successful push/delivery transitions multiple times
            if r >= 100:
                for _ in range(args.push_repeat - 1):
                    replay.add(t)

            s      = np.array(s2, dtype=np.float32)
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

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{args.episodes} [{phase}] return={ep_ret:.1f} eps={eps_by_step(steps):.3f} replay={len(replay)}")
            torch.save(q.state_dict(), args.out)

        gc.collect()

    torch.save(q.state_dict(), args.out)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
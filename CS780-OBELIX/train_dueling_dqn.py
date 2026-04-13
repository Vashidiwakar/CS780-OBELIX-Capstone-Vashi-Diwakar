"""Offline trainer: Dueling Double DQN + replay buffer (CPU) for OBELIX.

Run locally to create weights.pth, then submit agent.py + weights.pth.

Example:
  python train_dueling_dqn.py --obelix_py ./obelix.py --out weights.pth --episodes 2000 --difficulty 0
  python train_dueling_dqn.py --obelix_py ./obelix.py --out weights.pth --episodes 2000 --difficulty 0 --wall_obstacles


                    ALGORITHM: DUELING DOUBLE DEEP Q-NETWORK


Dueling DQN improves over standard DQN by splitting Q(s,a) into two streams:

    Q(s, a) = V(s)  +  A(s, a)  -  mean_a A(s, a)

Where:
  V(s)     — Value stream:     how good is this STATE regardless of action?
  A(s, a)  — Advantage stream: how much BETTER is action a vs. the average?
  mean subtraction ensures identifiability (V and A are not unique otherwise)

Why this helps:
  Many OBELIX states have all sensors dark — all actions are equally bad.
  V(s) learns "this state is bad" from every transition, while A(s,a)
  focuses only on relative differences between actions when sensors fire.
  This is faster and more stable than learning Q(s,a) directly.

Still uses the Double DQN target to avoid overestimation:
  a* = argmax_a Q_online(s', a)      <- online net selects action
  y  = r + gamma * Q_target(s', a*) <- target net evaluates it

Reference: Wang et al. 2016 https://arxiv.org/pdf/1511.07401
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
    def __init__(self, in_dim=18, n_actions=5):
        super().__init__()
     
        self.feature = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
        )
   
        self.value = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
       
        self.advantage = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        feat = self.feature(x)
        v    = self.value(feat)                         
        a    = self.advantage(feat)                     
        return v + (a - a.mean(dim=1, keepdim=True))    


@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool


class Replay:
    def __init__(self, cap: int = 100_000):
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
    def __len__(self): return len(self.buf)


def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",       type=str,   required=True)
    ap.add_argument("--out",             type=str,   default="weights.pth")
    ap.add_argument("--episodes",        type=int,   default=2000)
    ap.add_argument("--max_steps",       type=int,   default=2000)
    ap.add_argument("--difficulty",      type=int,   default=0)
    ap.add_argument("--wall_obstacles",  action="store_true")
    ap.add_argument("--box_speed",       type=int,   default=2)
    ap.add_argument("--scaling_factor",  type=int,   default=1)
    ap.add_argument("--arena_size",      type=int,   default=500)

    ap.add_argument("--gamma",           type=float, default=0.99)
    ap.add_argument("--lr",              type=float, default=5e-4)
    ap.add_argument("--batch",           type=int,   default=128)
    ap.add_argument("--replay",          type=int,   default=50_000)  
    ap.add_argument("--warmup",          type=int,   default=2000)
    ap.add_argument("--target_sync",     type=int,   default=1000)
    ap.add_argument("--eps_start",       type=float, default=1.0)
    ap.add_argument("--eps_end",         type=float, default=0.10)
    ap.add_argument("--eps_decay_steps", type=int,   default=500_000)
    ap.add_argument("--seed",            type=int,   default=9124)
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
        s      = env.reset(seed=args.seed + ep)   
        ep_ret = 0.0

        for _ in range(args.max_steps):
            eps = eps_by_step(steps)
            if np.random.rand() < eps:
                a = np.random.randint(len(ACTIONS))
            else:
                with torch.no_grad():
                    qs = q(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                a = int(np.argmax(qs))

            s2, r, done = env.step(ACTIONS[a], render=False)
            ep_ret += float(r)
            s2_arr = np.array(s2, dtype=np.float32)
            replay.add(Transition(s=s, a=a, r=float(r), s2=s2, done=bool(done)))
            s      = s2_arr
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
            print(f"Episode {ep+1}/{args.episodes} return={ep_ret:.1f} eps={eps_by_step(steps):.3f} replay={len(replay)}")
            torch.save(q.state_dict(), args.out)
            gc.collect()

    torch.save(q.state_dict(), args.out)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
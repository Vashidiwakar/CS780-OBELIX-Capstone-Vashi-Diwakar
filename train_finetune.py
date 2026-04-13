"""Stage 2: Finetune on harder levels using pretrained weights.

Loads weights from train_pretrain.py and fine-tunes on difficulty=2/3 with walls.
The agent already knows how to push — now it just needs to adapt to harder conditions.

Example:
  python train_finetune.py --obelix_py ./obelix.py --pretrained weights_pretrained.pth --out weights_final.pth --difficulty 2 --wall_obstacles
  python train_finetune.py --obelix_py ./obelix.py --pretrained weights_pretrained.pth --out weights_final.pth --difficulty 3 --wall_obstacles
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

ACTIONS       = ["L45", "L22", "FW", "R22", "R45"]
FW_IDX        = ACTIONS.index("FW")
ROT_IDX       = {0, 1, 3, 4}
EXPLORE_PROBS = np.array([0.05, 0.10, 0.65, 0.10, 0.10], dtype=np.float64)


class DuelingDQN(nn.Module):
    def __init__(self, in_dim=18, n_actions=5, hidden=256):
        super().__init__()
        self.feature   = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.value     = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.advantage = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))

    def forward(self, x):
        feat = self.feature(x)
        v    = self.value(feat)
        a    = self.advantage(feat)
        return v + (a - a.mean(dim=1, keepdim=True))


@dataclass
class Transition:
    s: np.ndarray; a: int; r: float; s2: np.ndarray; done: bool


class Replay:
    def __init__(self, cap=50_000):
        self.buf: Deque[Transition] = deque(maxlen=cap)

    def add(self, t):
        self.buf.append(t)

    def sample(self, batch):
        idx   = np.random.choice(len(self.buf), size=batch, replace=False)
        items = [self.buf[i] for i in idx]
        return (
            np.stack([it.s    for it in items]).astype(np.float32),
            np.array([it.a    for it in items], dtype=np.int64),
            np.array([it.r    for it in items], dtype=np.float32),
            np.stack([it.s2   for it in items]).astype(np.float32),
            np.array([it.done for it in items], dtype=np.float32),
        )

    def __len__(self): return len(self.buf)


def import_obelix(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def greedy_action(qs, obs, last_a, repeat, stuck, recovery):
    if obs[17] > 0:
        new_stuck = stuck + 1
        return (0 if (new_stuck//3)%2==0 else 4), 0, new_stuck, 15

    if recovery > 0:
        new_r = recovery - 1
        if new_r < 3:
            return FW_IDX, 0, stuck, new_r
        return (0 if (stuck//3)%2==0 else 4), 0, stuck, new_r

    q = qs.copy()
    if obs[16] > 0:            q[FW_IDX] += 10.0
    elif any(obs[4:12] > 0):   q[FW_IDX] += 5.0
    elif any(obs[0:4] > 0):    q[1]      += 3.0
    elif any(obs[12:16] > 0):  q[3]      += 3.0

    best = int(np.argmax(q))
    if last_a is not None and best == last_a and best in ROT_IDX:
        if repeat >= 2: return FW_IDX, 0, 0, 0
        return best, repeat+1, 0, 0
    return best, 0, 0, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",       type=str,   required=True)
    ap.add_argument("--pretrained",      type=str,   required=True)
    ap.add_argument("--out",             type=str,   default="weights_final.pth")
    ap.add_argument("--episodes",        type=int,   default=2000)
    ap.add_argument("--max_steps",       type=int,   default=1000)
    ap.add_argument("--difficulty",      type=int,   default=2)
    ap.add_argument("--wall_obstacles",  action="store_true")
    ap.add_argument("--box_speed",       type=int,   default=2)
    ap.add_argument("--scaling_factor",  type=int,   default=5)
    ap.add_argument("--arena_size",      type=int,   default=500)
    ap.add_argument("--gamma",           type=float, default=0.99)
    ap.add_argument("--lr",              type=float, default=1e-4)   # lower lr for finetuning
    ap.add_argument("--batch",           type=int,   default=128)
    ap.add_argument("--replay",          type=int,   default=50_000)
    ap.add_argument("--warmup",          type=int,   default=1000)
    ap.add_argument("--target_sync",     type=int,   default=1000)
    ap.add_argument("--eps_start",       type=float, default=0.5)    # start lower — already pretrained
    ap.add_argument("--eps_end",         type=float, default=0.10)
    ap.add_argument("--eps_decay_steps", type=int,   default=300_000)
    ap.add_argument("--push_repeat",     type=int,   default=15)
    ap.add_argument("--hidden",          type=int,   default=256)
    ap.add_argument("--seed",            type=int,   default=0)
    ap.add_argument("--render",          action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    # load pretrained weights
    q   = DuelingDQN(hidden=args.hidden)
    sd  = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    q.load_state_dict(sd)
    print(f"Loaded pretrained weights from {args.pretrained}")

    tgt = DuelingDQN(hidden=args.hidden)
    tgt.load_state_dict(q.state_dict())
    tgt.eval()

    opt    = optim.Adam(q.parameters(), lr=args.lr)
    replay = Replay(args.replay)
    steps  = 0

    def eps_by_step(t):
        if t >= args.eps_decay_steps: return args.eps_end
        return args.eps_start + (t/args.eps_decay_steps)*(args.eps_end - args.eps_start)

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
        s        = np.array(env.reset(seed=args.seed + ep), dtype=np.float32)
        ep_ret   = 0.0
        last_a   = None
        repeat   = 0
        stuck    = 0
        recovery = 0

        for _ in range(args.max_steps):
            eps = eps_by_step(steps)

            if np.random.rand() < eps:
                a        = int(np.random.choice(len(ACTIONS), p=EXPLORE_PROBS))
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
            if r >= 100:
                for _ in range(args.push_repeat - 1):
                    replay.add(t)

            s      = np.array(s2, dtype=np.float32)
            steps += 1

            if len(replay) >= max(args.warmup, args.batch):
                sb, ab, rb, s2b, db = replay.sample(args.batch)
                with torch.no_grad():
                    next_a   = torch.argmax(q(torch.tensor(s2b)), dim=1)
                    next_val = tgt(torch.tensor(s2b)).gather(1, next_a.unsqueeze(1)).squeeze(1)
                    y        = torch.tensor(rb) + args.gamma * (1 - torch.tensor(db)) * next_val

                pred = q(torch.tensor(sb)).gather(1, torch.tensor(ab).unsqueeze(1)).squeeze(1)
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
            print(f"Episode {ep+1}/{args.episodes} [d{args.difficulty}] return={ep_ret:.1f} eps={eps_by_step(steps):.3f} replay={len(replay)}")
            torch.save(q.state_dict(), args.out)

        gc.collect()

    torch.save(q.state_dict(), args.out)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
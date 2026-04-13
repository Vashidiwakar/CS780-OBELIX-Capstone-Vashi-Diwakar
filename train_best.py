"""Dueling Double DQN for OBELIX with reward normalization.

Key insight: the -200 stuck penalty makes the agent afraid to move forward.
Normalizing rewards to [-1, +10] fixes this — stuck and dark become equally
bad, delivery becomes the clear goal. Agent stops being risk-averse.

Algorithm: Dueling Double DQN
- Dueling: splits Q into V(s) + A(s,a) — learns state value independently
- Double: online selects action, target evaluates — reduces overestimation
- Reward normalization: scales rewards so stuck doesn't dominate
- Sweep exploration: rotates to find box instead of walking into boundary
- Push memory: stores successful push transitions 10x

Curriculum:
  ep 0    → 1000: difficulty=0, no wall
  ep 1000 → 2000: difficulty=0, with wall  
  ep 2000 → 3000: difficulty=2, with wall
  ep 3000 → 4000: difficulty=3, with wall

Example:
  python train_best.py --obelix_py ./obelix.py --out weights_best.pth --wall_obstacles
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
    def __init__(self, in_dim=18, n_actions=5, hidden=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value     = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.advantage = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))

    def forward(self, x):
        f = self.feature(x)
        v = self.value(f)
        a = self.advantage(f)
        return v + (a - a.mean(dim=1, keepdim=True))


@dataclass
class Transition:
    s: np.ndarray; a: int; r: float; s2: np.ndarray; done: bool


class Replay:
    def __init__(self, cap=100_000):
        self.buf: Deque[Transition] = deque(maxlen=cap)

    def add(self, t): self.buf.append(t)

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


def normalize_reward(r: float) -> float:
    """Scale reward so stuck doesn't dominate learning.
    
    Original:  stuck=-200, dark=-18, sensor=+1..+5, attach=+100, deliver=+2000
    Normalized: stuck=-1,  dark=-0.09, sensor=+0.005..+0.025, attach=+0.5, deliver=+10
    
    This is NOT reward shaping — relative ordering is identical.
    Just prevents -200 from making agent pathologically afraid of movement.
    """
    return r / 200.0


def select_action(qs, obs, step_count, stuck_count, recovery_steps):
    """Select action based on Q-values with sweep exploration in dark states.
    
    When sensors fire: boost corresponding Q-values strongly
    When stuck: force rotation recovery
    When all dark: sweep pattern to avoid boundary
    """
    # stuck recovery
    if obs[17] > 0:
        stuck_count    += 1
        recovery_steps  = 20
        step_count      = 0

    if recovery_steps > 0:
        recovery_steps -= 1
        if recovery_steps == 0:
            step_count = 0
        rot    = 0 if (stuck_count // 3) % 2 == 0 else 4
        forced = FW_IDX if recovery_steps < 3 else rot
        return forced, step_count, stuck_count, recovery_steps

    q = qs.copy()

    # boost Q-values based on sensor readings
    # strong boost ensures network learns sensor → action association
    if obs[16] > 0:             q[FW_IDX] += 10.0   # IR — box directly ahead
    elif any(obs[4:12] > 0):    q[FW_IDX] += 5.0    # forward sonars
    elif any(obs[0:4]  > 0):    q[1]      += 3.0    # left sonars → L22
    elif any(obs[12:16]> 0):    q[3]      += 3.0    # right sonars → R22
    else:
        # all dark — sweep pattern to avoid boundary
        # rotate 10 steps then FW 2 steps to reposition
        sweep = step_count % 12
        if sweep < 10:
            rot = 0 if (step_count // 12) % 2 == 0 else 4
            q[rot] += 3.0
        else:
            q[FW_IDX] += 3.0

    step_count += 1
    return int(np.argmax(q)), step_count, stuck_count, recovery_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",      type=str,   required=True)
    ap.add_argument("--out",            type=str,   default="weights_best.pth")
    ap.add_argument("--episodes",       type=int,   default=4000)
    ap.add_argument("--max_steps",      type=int,   default=1000)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--scaling_factor", type=int,   default=5)
    ap.add_argument("--gamma",          type=float, default=0.99)
    ap.add_argument("--lr",             type=float, default=5e-4)
    ap.add_argument("--batch",          type=int,   default=128)
    ap.add_argument("--replay",         type=int,   default=100_000)
    ap.add_argument("--warmup",         type=int,   default=2000)
    ap.add_argument("--target_sync",    type=int,   default=1000)
    ap.add_argument("--eps_start",      type=float, default=1.0)
    ap.add_argument("--eps_end",        type=float, default=0.10)
    ap.add_argument("--eps_decay",      type=int,   default=800_000)
    ap.add_argument("--push_repeat",    type=int,   default=10)
    ap.add_argument("--hidden",         type=int,   default=128)
    ap.add_argument("--seed",           type=int,   default=42)
    ap.add_argument("--render",         action="store_true")
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

    def eps_now(t):
        if t >= args.eps_decay: return args.eps_end
        return args.eps_start + (t / args.eps_decay) * (args.eps_end - args.eps_start)

    # curriculum — 4 stages
    stages = [
        (1000, 0, False),
        (2000, 0, args.wall_obstacles),
        (3000, 2, args.wall_obstacles),
        (4000, 3, args.wall_obstacles),
    ]

    envs = {}
    for _, diff, wall in stages:
        key = (diff, wall)
        if key not in envs:
            envs[key] = OBELIX(
                scaling_factor=args.scaling_factor,
                arena_size=500,
                max_steps=args.max_steps,
                wall_obstacles=wall,
                difficulty=diff,
                box_speed=2,
                seed=args.seed,
            )

    def get_env(ep):
        for limit, diff, wall in stages:
            if ep < limit:
                return envs[(diff, wall)], f"d{diff}{'w' if wall else ''}"
        return envs[(3, args.wall_obstacles)], "d3w"

    for ep in range(args.episodes):
        env, phase = get_env(ep)
        s          = np.array(env.reset(seed=args.seed + ep), dtype=np.float32)
        ep_ret     = 0.0
        step_count = 0
        stuck_count    = 0
        recovery_steps = 0

        for _ in range(args.max_steps):
            eps = eps_now(steps)

            if np.random.rand() < eps:
                # during exploration also use sweep to avoid boundary
                if s[17] > 0:
                    stuck_count    += 1
                    recovery_steps  = 20
                    step_count      = 0
                if recovery_steps > 0:
                    recovery_steps -= 1
                    rot = 0 if (stuck_count // 3) % 2 == 0 else 4
                    a   = FW_IDX if recovery_steps < 3 else rot
                else:
                    # sweep exploration
                    sweep = step_count % 20
                    if sweep < 18:
                        a = 0 if (step_count // 20) % 2 == 0 else 4
                    else:
                        a = FW_IDX
                    step_count += 1
            else:
                with torch.no_grad():
                    qs = q(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                a, step_count, stuck_count, recovery_steps = select_action(
                    qs, s, step_count, stuck_count, recovery_steps
                )

            s2, r, done = env.step(ACTIONS[a], render=args.render)
            ep_ret     += r

            # normalize reward before storing
            r_norm = normalize_reward(r)

            t = Transition(s=s, a=a, r=r_norm, s2=s2, done=bool(done))
            replay.add(t)

            # store push/delivery transitions multiple times
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
            print(f"ep {ep+1:4d}/{args.episodes}  [{phase}]  return={ep_ret:10.1f}  eps={eps_now(steps):.3f}  replay={len(replay)}")
            torch.save(q.state_dict(), args.out)

        gc.collect()

    torch.save(q.state_dict(), args.out)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
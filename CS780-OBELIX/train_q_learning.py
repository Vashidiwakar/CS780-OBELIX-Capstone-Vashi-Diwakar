"""Offline trainer: Tabular Q-Learning for OBELIX.

Run locally to create weights.pth, then submit agent_q_learning.py + weights.pth.

Example:
  python train_q_learning.py --obelix_py ./obelix.py --out weights.pth --episodes 2000 --difficulty 0
  python train_q_learning.py --obelix_py ./obelix.py --out weights.pth --episodes 2000 --difficulty 0 --wall_obstacles


                    ALGORITHM: TABULAR Q-LEARNING


Q-learning is an off-policy TD control algorithm (Watkins, 1989).
It learns the optimal action-value function Q*(s,a) directly:

    Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

Where:
  alpha  — learning rate: how much to update Q on each step
  gamma  — discount factor: how much to value future rewards
  r      — reward received after taking action a in state s
  max_a' — greedy action in next state (off-policy)

State representation:
  The 18-bit binary observation is used directly as a tuple key.
  e.g. obs = [0,1,0,0,1,0,...] -> key = (0,1,0,0,1,0,...)
  The Q-table is a dictionary mapping (state, action) -> Q-value.
  New states are initialised to 0.

Exploration:
  Epsilon-greedy: with probability eps take random action,
  otherwise take action with highest Q-value.
  Epsilon decays linearly from eps_start to eps_end.

Reference:
  Watkins & Dayan (1992) Q-learning. Machine Learning.
  Mahadevan & Connell (1992) — original OBELIX paper used Q-learning.
  https://cdn.aaai.org/AAAI/1991/AAAI91-120.pdf
"""

from __future__ import annotations
import argparse, random, gc
from collections import defaultdict

import numpy as np
import torch

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)


def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def obs_to_key(obs):
    return tuple(int(x) for x in obs)


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
    ap.add_argument("--alpha",           type=float, default=0.1,
                    help="Learning rate for Q-table updates")
    ap.add_argument("--eps_start",       type=float, default=1.0)
    ap.add_argument("--eps_end",         type=float, default=0.10)
    ap.add_argument("--eps_decay_steps", type=int,   default=500_000)
    ap.add_argument("--seed",            type=int,   default=1500)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    # Q-table: defaultdict returns zeros for unseen states
    # Q[state][action] = Q-value
    Q = defaultdict(lambda: np.zeros(N_ACTIONS, dtype=np.float32))

    steps = 0

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
        s      = np.array(env.reset(seed=args.seed + ep), dtype=np.float32)
        ep_ret = 0.0
        key    = obs_to_key(s)

        for _ in range(args.max_steps):
            eps = eps_by_step(steps)

            # Epsilon-greedy action selection
            if np.random.rand() < eps:
                a = np.random.randint(N_ACTIONS)
            else:
                a = int(np.argmax(Q[key]))

            s2, r, done = env.step(ACTIONS[a], render=False)
            s2    = np.array(s2, dtype=np.float32)
            key2  = obs_to_key(s2)
            ep_ret += float(r)
            steps  += 1

            # Q-learning update:
            # Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
            if done:
                target = float(r)
            else:
                target = float(r) + args.gamma * float(np.max(Q[key2]))

            Q[key][a] += args.alpha * (target - Q[key][a])

            key = key2

            if done:
                break

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{args.episodes} return={ep_ret:.1f} eps={eps_by_step(steps):.3f} states={len(Q)}")
            torch.save({"Q": dict(Q)}, args.out)
            gc.collect()

    torch.save({"Q": dict(Q)}, args.out)
    print(f"Saved: {args.out}  (Q-table size: {len(Q)} states)")


if __name__ == "__main__":
    main()
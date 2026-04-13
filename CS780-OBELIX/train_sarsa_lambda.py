"""Offline trainer: SARSA(lambda) with linear function approximation for OBELIX.

Run locally to create weights.pth, then submit agent.py + weights.pth.

Example:
  python train_sarsa_lambda.py --obelix_py ./obelix.py --out weights.pth --episodes 2000 --difficulty 0
  python train_sarsa_lambda.py --obelix_py ./obelix.py --out weights.pth --episodes 2000 --difficulty 0 --wall_obstacles


                    ALGORITHM: SARSA(lambda) — ON-POLICY TD CONTROL
                    WITH ELIGIBILITY TRACES + LINEAR FUNCTION APPROXIMATION

"""

from __future__ import annotations
import argparse, random

import numpy as np

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)

# Selected pairwise feature indices — chosen to capture meaningful
# sensor combinations relevant to OBELIX navigation
PAIRS = [
    (16, 4), (16, 5), (16, 6), (16, 7),   # IR + forward far/near
    (16, 8), (16, 9), (16, 10), (16, 11),  # IR + forward far/near (right side)
    (4,  5), (6,  7), (8,  9), (10, 11),   # far+near same sensor
    (0,  1), (2,  3), (12, 13), (14, 15),  # left/right far+near pairs
    (16, 17),                               # IR + stuck
    (4,  12), (5,  13), (0,  14),          # cross sensor pairs
]
N_PAIRS   = len(PAIRS)
N_FEATURES = 18 + N_PAIRS + 1              # raw + pairwise + bias


def phi(obs: np.ndarray) -> np.ndarray:
    """
    Feature vector from raw 18-bit observation.
    Returns shape (N_FEATURES,) float32 array.

    Components:
      [0:18]          raw bits
      [18:18+N_PAIRS] pairwise products (AND features)
      [-1]            bias = 1.0
    """
    raw   = obs.astype(np.float32)
    pairs = np.array([raw[i] * raw[j] for i, j in PAIRS], dtype=np.float32)
    bias  = np.array([1.0], dtype=np.float32)
    return np.concatenate([raw, pairs, bias])


def eps_by_step(t, eps_start, eps_end, eps_decay_steps):
    if t >= eps_decay_steps:
        return eps_end
    frac = t / eps_decay_steps
    return eps_start + frac * (eps_end - eps_start)


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
    ap.add_argument("--max_steps",       type=int,   default=1000)
    ap.add_argument("--difficulty",      type=int,   default=0)
    ap.add_argument("--wall_obstacles",  action="store_true")
    ap.add_argument("--box_speed",       type=int,   default=2)
    ap.add_argument("--scaling_factor",  type=int,   default=5)
    ap.add_argument("--arena_size",      type=int,   default=500)

    ap.add_argument("--gamma",           type=float, default=0.99)
    ap.add_argument("--alpha",           type=float, default=0.01,
                    help="Learning rate for weight updates")
    ap.add_argument("--lam",             type=float, default=0.9,
                    help="Lambda for eligibility trace decay (0=TD, 1=MC)")
    ap.add_argument("--eps_start",       type=float, default=1.0)
    ap.add_argument("--eps_end",         type=float, default=0.05)
    ap.add_argument("--eps_decay_steps", type=int,   default=200_000)
    ap.add_argument("--seed",            type=int,   default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    # Weight matrix: w[a] is the weight vector for action a
    # Shape: (N_ACTIONS, N_FEATURES)
    # Initialised to small random values to break symmetry
    w = np.random.randn(N_ACTIONS, N_FEATURES).astype(np.float32) * 0.01

    steps = 0

    # Create env ONCE outside the loop — recreating every episode leaks memory
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

        # Eligibility trace — same shape as w, reset every episode
        e = np.zeros((N_ACTIONS, N_FEATURES), dtype=np.float32)

        # Feature vector for starting state
        f = phi(s)

        # Q values for all actions at s: shape (N_ACTIONS,)
        q_s = w @ f                                  # dot product each row with f

        # Choose first action epsilon-greedy
        eps = eps_by_step(steps, args.eps_start, args.eps_end, args.eps_decay_steps)
        if np.random.rand() < eps:
            a = np.random.randint(N_ACTIONS)
        else:
            a = int(np.argmax(q_s))

        for _ in range(args.max_steps):
            s2, r, done = env.step(ACTIONS[a], render=False)
            s2     = np.array(s2, dtype=np.float32)
            ep_ret += float(r)
            steps  += 1

            # Feature vector for next state
            f2    = phi(s2)
            q_s2  = w @ f2                           # Q values at s2

            # Choose next action epsilon-greedy (SARSA: on-policy)
            eps = eps_by_step(steps, args.eps_start, args.eps_end, args.eps_decay_steps)
            if np.random.rand() < eps:
                a2 = np.random.randint(N_ACTIONS)
            else:
                a2 = int(np.argmax(q_s2))

            # TD error (delta)
            # If terminal: no next state value, delta = r - Q(s,a)
            if done:
                delta = float(r) - float(w[a] @ f)
            else:
                delta = float(r) + args.gamma * float(q_s2[a2]) - float(w[a] @ f)

            # Eligibility trace update:
            # Decay all traces, then spike the current (s,a) trace
            e        *= args.gamma * args.lam        # decay all
            e[a]     += f                            # spike current action

            # Weight update: w += alpha * delta * e  (for ALL actions)
            w        += args.alpha * delta * e

            # Clip weights to prevent explosion
            np.clip(w, -10.0, 10.0, out=w)

            # Advance to next state and action
            s  = s2
            f  = f2
            a  = a2

            if done:
                break

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{args.episodes} return={ep_ret:.1f} eps={eps_by_step(steps, args.eps_start, args.eps_end, args.eps_decay_steps):.3f}")
            
            import torch
            torch.save(
                {"w": w, "pairs": PAIRS, "n_features": N_FEATURES},
                args.out
            )
            print("Checkpoint saved:", args.out)

    # Save weights as a dict matching the torch.save style for consistency
    import torch
    torch.save({"w": w, "pairs": PAIRS, "n_features": N_FEATURES}, args.out)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()

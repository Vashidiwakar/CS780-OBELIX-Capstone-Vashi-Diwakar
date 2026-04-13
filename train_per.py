"""Offline trainer: Dueling Double DQN + Prioritized Experience Replay for OBELIX.

Run locally to create weights.pth, then submit agent_per.py + weights.pth.

Example:
  python train_per.py --obelix_py ./obelix.py --out weights_per.pth --scaling_factor 5
  python train_per.py --obelix_py ./obelix.py --out weights_per.pth --scaling_factor 5 --wall_obstacles


            ALGORITHM: DUELING DOUBLE DQN + PRIORITIZED EXPERIENCE REPLAY

Why PER fixes the exact OBELIX problem:
  - Standard replay samples ALL transitions uniformly
  - OBELIX has sparse rewards — +2000 transitions are rare vs thousands of -1 steps
  - Those rare successes get sampled as often as boring -1 transitions → forgotten
  - PER samples transitions proportional to TD error — surprising transitions first
  - +2000 success has huge TD error → sampled many times → network remembers it
  - -1 boring step has tiny TD error → sampled rarely → doesn't dominate learning

How PER works:
  1. Every transition stored with priority = |TD error| + epsilon
  2. Sample probability ∝ priority^alpha
  3. IS weights correct for sampling bias: w = (1/N * 1/P(i))^beta
  4. Beta anneals from beta_start → 1.0 over training
  5. After each update, refresh priorities with new TD errors

SumTree data structure:
  - Binary tree where leaves = priorities, internal nodes = sum of children
  - O(log N) sampling instead of O(N) for uniform replay
  - O(log N) priority update

References:
  Schaul et al. 2016  https://arxiv.org/pdf/1511.05952  (PER)
  Wang et al. 2016    https://arxiv.org/pdf/1511.07401  (Dueling DQN)
  Van Hasselt 2016    https://arxiv.org/pdf/1509.06461  (Double DQN)
"""

from __future__ import annotations
import argparse, random, gc

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


class DuelingDQN(nn.Module):
    def __init__(self, in_dim=18, n_actions=5, hidden=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        feat = self.feature(x)
        v    = self.value(feat)
        a    = self.advantage(feat)
        return v + (a - a.mean(dim=1, keepdim=True))


class SumTree:
    """Binary tree where leaves store priorities, internal nodes store sums.
    Enables O(log N) proportional sampling and priority updates."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data     = np.zeros(capacity, dtype=object)
        self.size     = 0
        self.ptr      = 0

    def _propagate(self, idx: int, delta: float):
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def _retrieve(self, idx: int, s: float) -> int:
        left  = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data):
        idx = self.ptr + self.capacity - 1
        self.data[self.ptr] = data
        self.update(idx, priority)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx: int, priority: float):
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, delta)

    def sample(self, s: float):
        idx      = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, float(self.tree[idx]), self.data[data_idx]

    def __len__(self):
        return self.size


class PrioritizedReplay:
    """Prioritized Experience Replay buffer backed by a SumTree."""

    def __init__(self, capacity: int, alpha: float = 0.6, epsilon: float = 1e-6):
        self.tree    = SumTree(capacity)
        self.alpha   = alpha   # priority exponent — 0=uniform, 1=fully prioritized
        self.epsilon = epsilon # small constant to avoid zero priority
        self.max_p   = 1.0     # track max priority for new transitions

    def add(self, transition):
        # New transitions get max priority so they're sampled at least once
        self.tree.add(self.max_p, transition)

    def sample(self, batch: int, beta: float):
        indices  = np.zeros(batch, dtype=np.int32)
        weights  = np.zeros(batch, dtype=np.float32)
        segments = self.tree.total() / batch

        min_prob = np.min(self.tree.tree[-self.tree.capacity:][
            self.tree.tree[-self.tree.capacity:] > 0
        ]) / self.tree.total()
        max_w = (min_prob * len(self.tree)) ** (-beta)

        transitions = []
        for i in range(batch):
            s   = random.uniform(segments * i, segments * (i + 1))
            idx, p, t = self.tree.sample(s)
            # If slot uninitialized, resample from valid range
            while not isinstance(t, tuple):
                s   = random.uniform(0, self.tree.total())
                idx, p, t = self.tree.sample(s)
            indices[i] = idx
            prob       = max(p / self.tree.total(), 1e-8)
            w          = (prob * len(self.tree)) ** (-beta)
            weights[i] = w / max_w
            transitions.append(t)

        s  = np.stack([t[0] for t in transitions]).astype(np.float32)
        a  = np.array([t[1] for t in transitions], dtype=np.int64)
        r  = np.array([t[2] for t in transitions], dtype=np.float32)
        s2 = np.stack([t[3] for t in transitions]).astype(np.float32)
        d  = np.array([t[4] for t in transitions], dtype=np.float32)
        return s, a, r, s2, d, indices, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        for idx, err in zip(indices, td_errors):
            p = (abs(float(err)) + self.epsilon) ** self.alpha
            self.tree.update(int(idx), p)
            self.max_p = max(self.max_p, p)

    def __len__(self):
        return self.tree.size


def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",       type=str,   required=True)
    ap.add_argument("--out",             type=str,   default="weights_per.pth")
    ap.add_argument("--episodes",        type=int,   default=6000)
    ap.add_argument("--max_steps",       type=int,   default=1000)
    ap.add_argument("--difficulty",      type=int,   default=0)
    ap.add_argument("--wall_obstacles",  action="store_true")
    ap.add_argument("--box_speed",       type=int,   default=2)
    ap.add_argument("--scaling_factor",  type=int,   default=5)
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
    ap.add_argument("--per_alpha",       type=float, default=0.6)
    ap.add_argument("--per_beta_start",  type=float, default=0.4)
    ap.add_argument("--per_epsilon",     type=float, default=1e-6)
    ap.add_argument("--seed",            type=int,   default=42)
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
    replay = PrioritizedReplay(args.replay, alpha=args.per_alpha,
                                epsilon=args.per_epsilon)
    steps      = 0
    rng        = np.random.default_rng(args.seed)
    dark_steps = 0

    def eps_by_step(t):
        if t >= args.eps_decay_steps:
            return args.eps_end
        frac = t / args.eps_decay_steps
        return args.eps_start + frac * (args.eps_end - args.eps_start)

    def beta_by_step(t):
        # Anneal beta from beta_start → 1.0 over training
        frac = min(1.0, t / args.eps_decay_steps)
        return args.per_beta_start + frac * (1.0 - args.per_beta_start)

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
        obs        = np.array(env.reset(seed=args.seed + ep), dtype=np.float32)
        ep_ret     = 0.0
        dark_steps = 0
        stuck_count = 0

        for _ in range(args.max_steps):
            eps = eps_by_step(steps)

            # Hard override: stuck → rotate to escape wall
            if obs[17] == 1:
                stuck_count += 1
                dark_steps   = 0
                a = 0 if stuck_count % 2 == 0 else 4  # L45/R45

            # Hard override: IR fires → box right there, go forward
            elif obs[16] == 1:
                a           = 2  # FW
                dark_steps  = 0
                stuck_count = 0

            # Any sensor active → epsilon-greedy from network
            elif any(obs[:16]):
                dark_steps  = 0
                stuck_count = 0
                if rng.random() < eps:
                    a = int(rng.integers(len(ACTIONS)))
                else:
                    with torch.no_grad():
                        qs = q(torch.tensor(obs).unsqueeze(0)).squeeze(0).numpy()
                    a = int(np.argmax(qs))

            # All sensors dark → rotate to scan
            else:
                stuck_count = 0
                dark_steps += 1
                if rng.random() < eps:
                    a = 0 if (dark_steps // 8) % 2 == 0 else 4  # L45/R45
                else:
                    with torch.no_grad():
                        qs = q(torch.tensor(obs).unsqueeze(0)).squeeze(0).numpy()
                    a = int(np.argmax(qs))

            s2, r, done = env.step(ACTIONS[a], render=False)
            s2 = np.array(s2, dtype=np.float32)
            ep_ret += float(r)

            # Store transition — PER assigns max priority to new transitions
            replay.add((obs, a, float(r), s2, bool(done)))
            obs    = s2
            steps += 1

            # Learn
            if len(replay) >= max(args.warmup, args.batch):
                beta = beta_by_step(steps)
                sb, ab, rb, s2b, db, idxs, ws = replay.sample(args.batch, beta)

                sb_t  = torch.tensor(sb)
                ab_t  = torch.tensor(ab)
                rb_t  = torch.tensor(rb)
                s2b_t = torch.tensor(s2b)
                db_t  = torch.tensor(db)
                ws_t  = torch.tensor(ws)

                with torch.no_grad():
                    # Double DQN: online selects, target evaluates
                    next_a   = torch.argmax(q(s2b_t), dim=1)
                    next_val = tgt(s2b_t).gather(1, next_a.unsqueeze(1)).squeeze(1)
                    y        = rb_t + args.gamma * (1.0 - db_t) * next_val

                pred = q(sb_t).gather(1, ab_t.unsqueeze(1)).squeeze(1)

                # IS-weighted Huber loss
                td_errors = (pred - y).detach()
                loss      = (ws_t * nn.functional.smooth_l1_loss(
                    pred, y, reduction="none"
                )).mean()

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                opt.step()

                # Update priorities with fresh TD errors
                replay.update_priorities(idxs, td_errors.cpu().numpy())

                if steps % args.target_sync == 0:
                    tgt.load_state_dict(q.state_dict())

            if done:
                break

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{args.episodes}  return={ep_ret:.1f}  eps={eps_by_step(steps):.3f}  beta={beta_by_step(steps):.3f}  replay={len(replay)}")
            torch.save(q.state_dict(), args.out)
            gc.collect()

    torch.save(q.state_dict(), args.out)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
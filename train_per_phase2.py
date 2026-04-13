"""Offline trainer: DDDQN-PER for OBELIX Phase 2 (Blinking Box).

Run locally to create weights.pth, then submit agent.py + weights.pth.

Example:
  python train_per_phase2.py --obelix_py ./obelix.py --out weights_per_phase2.pth --wall_obstacles --episodes 2000 --ep_switch 1000


                    ALGORITHM: DDDQN-PER — PHASE 2 (BLINKING BOX)


Same as Phase 1 PER but with curriculum:
  Episodes 0 to ep_switch    → difficulty=0 (static box)
  Episodes ep_switch to end  → difficulty=2 (blinking box)

PER helps Phase 2 specifically because:
  When box blinks off, all sensors go dark → boring -1/step transitions
  When box blinks on and agent finds it   → rare +100/+2000 transitions
  PER samples the rare successes MORE often → faster learning

References:
  Schaul et al. 2016  https://arxiv.org/pdf/1511.05952
  Wang  et al. 2016   https://arxiv.org/pdf/1511.07401
  Van Hasselt et al.  https://arxiv.org/pdf/1509.06461
"""

from __future__ import annotations
import argparse, random, gc
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)


class DuelingDQN(nn.Module):
    def __init__(self, in_dim=18, n_actions=5):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),   nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        feat = self.feature(x)
        v    = self.value(feat)
        a    = self.advantage(feat)
        return v + (a - a.mean(dim=1, keepdim=True))


class SumTree:
    def __init__(self, cap: int):
        self.cap  = cap
        self.tree = np.zeros(2 * cap - 1, dtype=np.float32)
        self.data = np.empty(cap, dtype=object)
        self.ptr  = 0
        self.size = 0

    def _propagate(self, idx: int, delta: float):
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def add(self, priority: float, transition):
        leaf_idx            = self.ptr + self.cap - 1
        self.data[self.ptr] = transition
        self.update(leaf_idx, priority)
        self.ptr  = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def update(self, leaf_idx: int, priority: float):
        delta               = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        self._propagate(leaf_idx, delta)

    def get(self, value: float):
        idx = 0
        while idx < self.cap - 1:
            left  = 2 * idx + 1
            right = 2 * idx + 2
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx    = right
        data_idx = idx - (self.cap - 1)
        if self.data[data_idx] is None or isinstance(self.data[data_idx], int):
            data_idx = (self.ptr - 1) % self.cap
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total(self) -> float:
        return float(self.tree[0])


@dataclass
class Transition:
    s:    np.ndarray
    a:    int
    r:    float
    s2:   np.ndarray
    done: bool


class PrioritizedReplay:
    def __init__(self, cap=50_000, alpha=0.6, beta=0.1, beta_rate=0.99992):
        self.tree      = SumTree(cap)
        self.alpha     = alpha
        self.beta      = beta
        self.beta_rate = beta_rate
        self.max_pri   = 1.0
        self.eps       = 1e-6

    def store(self, t: Transition):
        self.tree.add(self.max_pri, t)

    def sample(self, batch: int, beta: float):
        self.beta = min(1.0, self.beta * self.beta_rate)
        segment   = self.tree.total / batch
        samples, indices, weights = [], [], []

        for i in range(batch):
            v              = np.random.uniform(segment * i, segment * (i + 1))
            leaf_idx, pri, t = self.tree.get(v)
            prob           = max(pri / (self.tree.total + 1e-8), 1e-8)
            w              = (prob * self.tree.size) ** (-beta)
            samples.append(t)
            indices.append(leaf_idx)
            weights.append(w)

        weights = np.array(weights, dtype=np.float32)
        weights /= weights.max() + 1e-8

        s  = np.stack([t.s  for t in samples]).astype(np.float32)
        a  = np.array([t.a  for t in samples], dtype=np.int64)
        r  = np.array([t.r  for t in samples], dtype=np.float32)
        s2 = np.stack([t.s2 for t in samples]).astype(np.float32)
        d  = np.array([t.done for t in samples], dtype=np.float32)
        return s, a, r, s2, d, np.array(indices), weights

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            p = float((abs(err) + self.eps) ** self.alpha)
            self.tree.update(idx, p)
            self.max_pri = max(self.max_pri, p)

    def __len__(self):
        return self.tree.size


def select_action(q, obs, eps, dark_steps, dark_dir):
    if obs[17] == 1:
        return random.choice([0, 4])
    if obs[16] == 1:
        return 2
    if np.any(obs[:16] == 1):
        if random.random() < eps:
            return random.randint(0, N_ACTIONS - 1)
        with torch.no_grad():
            qs = q(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
        return int(np.argmax(qs))
    phase = dark_steps % 16
    if phase < 8:
        return 0 if dark_dir % 2 == 0 else 4
    else:
        return 4 if dark_dir % 2 == 0 else 0


def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",       type=str,   required=True)
    ap.add_argument("--out",             type=str,   default="weights_per_phase2.pth")
    ap.add_argument("--episodes",        type=int,   default=2000)
    ap.add_argument("--max_steps",       type=int,   default=1000)
    ap.add_argument("--wall_obstacles",  action="store_true")
    ap.add_argument("--scaling_factor",  type=int,   default=5)
    ap.add_argument("--arena_size",      type=int,   default=500)
    ap.add_argument("--ep_switch",       type=int,   default=1000)

    ap.add_argument("--gamma",           type=float, default=0.99)
    ap.add_argument("--lr",              type=float, default=5e-4)
    ap.add_argument("--batch",           type=int,   default=128)
    ap.add_argument("--replay",          type=int,   default=50_000)
    ap.add_argument("--warmup",          type=int,   default=2000)
    ap.add_argument("--target_sync",     type=int,   default=500)
    ap.add_argument("--eps_start",       type=float, default=1.0)
    ap.add_argument("--eps_end",         type=float, default=0.15)
    ap.add_argument("--eps_decay_steps", type=int,   default=600_000)
    ap.add_argument("--per_alpha",       type=float, default=0.6)
    ap.add_argument("--per_beta",        type=float, default=0.1)
    ap.add_argument("--per_beta_rate",   type=float, default=0.99992)
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
    replay = PrioritizedReplay(args.replay, args.per_alpha, args.per_beta, args.per_beta_rate)
    steps  = 0

    def eps_by_step(t):
        if t >= args.eps_decay_steps:
            return args.eps_end
        frac = t / args.eps_decay_steps
        return args.eps_start + frac * (args.eps_end - args.eps_start)

    def beta_by_step(t):
        frac = min(1.0, t / (args.episodes * args.max_steps))
        return args.per_beta + frac * (1.0 - args.per_beta)

    # Create envs ONCE outside the loop — recreating every episode leaks memory
    env_d0 = OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=False,
        difficulty=0,
        box_speed=2,
        seed=args.seed,
    )
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
        env = env_d0 if ep < args.ep_switch else env_d2
        s      = np.array(env.reset(seed=args.seed + ep), dtype=np.float32)
        ep_ret = 0.0
        dark_steps = 0
        dark_dir   = ep % 4

        for _ in range(args.max_steps):
            eps = eps_by_step(steps)
            a   = select_action(q, s, eps, dark_steps, dark_dir)

            if np.any(s[:16] == 1) or s[16] == 1:
                dark_steps = 0
            else:
                dark_steps += 1

            s2, r, done = env.step(ACTIONS[a], render=False)
            s2 = np.array(s2, dtype=np.float32)
            ep_ret += float(r)

            t = Transition(s=s, a=a, r=float(r), s2=s2, done=bool(done))
            replay.store(t)

            s      = s2
            steps += 1

            if len(replay) >= max(args.warmup, args.batch):
                beta = beta_by_step(steps)
                sb, ab, rb, s2b, db, idxs, ws = replay.sample(args.batch, beta)

                sb_t  = torch.tensor(sb)
                ab_t  = torch.tensor(ab)
                rb_t  = torch.tensor(rb)
                s2b_t = torch.tensor(s2b)
                db_t  = torch.tensor(db)
                w_t   = torch.tensor(ws)

                with torch.no_grad():
                    next_a   = torch.argmax(q(s2b_t), dim=1)
                    next_val = tgt(s2b_t).gather(1, next_a.unsqueeze(1)).squeeze(1)
                    y        = rb_t + args.gamma * (1.0 - db_t) * next_val

                pred     = q(sb_t).gather(1, ab_t.unsqueeze(1)).squeeze(1)
                td_error = (pred - y).detach().cpu().numpy()
                loss     = (w_t * nn.functional.smooth_l1_loss(pred, y, reduction="none")).mean()

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()

                replay.update_priorities(idxs, td_error)

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
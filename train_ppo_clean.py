"""Clean PPO for OBELIX — v3.

Key fix: robust boundary/wall handling.
- Recovery resets sweep counter after completing
- Sweep pattern alternates L45/R45 to cover both directions
- After recovery always rotates away from boundary before moving forward
- step_count resets on stuck AND after recovery completes

Example:
  python train_ppo_clean.py --obelix_py ./obelix.py --out weights_ppo_clean.pth --wall_obstacles
"""

from __future__ import annotations
import argparse, random, gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


class ActorCritic(nn.Module):
    def __init__(self, in_dim=18, n_actions=5, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor  = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def act(self, obs: np.ndarray):
        x           = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits, val = self(x)
        dist        = Categorical(logits=logits)
        a           = dist.sample()
        return int(a), float(dist.log_prob(a).detach()), float(val.detach())

    def act_forced(self, obs: np.ndarray, forced_a: int):
        with torch.no_grad():
            x           = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits, val = self(x)
            dist        = Categorical(logits=logits)
            lp          = float(dist.log_prob(torch.tensor(forced_a)).detach())
        return forced_a, lp, float(val.detach())

    def evaluate(self, obs_t, act_t):
        logits, vals = self(obs_t)
        dist         = Categorical(logits=logits)
        return dist.log_prob(act_t), vals, dist.entropy()


def import_obelix(path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def compute_gae(rewards, values, dones, last_val, gamma=0.99, lam=0.95):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae        = 0.0
    for t in reversed(range(len(rewards))):
        next_val      = last_val if t == len(rewards)-1 else values[t+1]
        delta         = rewards[t] + gamma * next_val * (1-dones[t]) - values[t]
        gae           = delta + gamma * lam * (1-dones[t]) * gae
        advantages[t] = gae
    return advantages, advantages + np.array(values, dtype=np.float32)


def get_action(obs, net, step_count, stuck_count, recovery_steps):
    """
    Priority order:
    1. Recovery from boundary/wall — rotate then FW
    2. IR sensor fires — FW
    3. Forward sonars fire — FW
    4. Left sonars fire — L22
    5. Right sonars fire — R22
    6. All dark — sweep: rotate 10 steps then FW 2 steps
    """
    # stuck — start recovery
    if obs[17] > 0:
        stuck_count    += 1
        recovery_steps  = 20
        step_count      = 0

    # recovery in progress
    if recovery_steps > 0:
        recovery_steps -= 1
        if recovery_steps == 0:
            step_count = 0   # reset sweep when recovery finishes
        # first 17 steps rotate, last 3 steps FW to clear boundary
        rot = 0 if (stuck_count // 3) % 2 == 0 else 4
        forced = 2 if recovery_steps < 3 else rot
        a, lp, val = net.act_forced(obs, forced)
        return a, lp, val, step_count, stuck_count, recovery_steps

    # sensor guided
    if obs[16] > 0:
        a, lp, val = net.act_forced(obs, 2)
        return a, lp, val, step_count, stuck_count, recovery_steps

    if any(obs[4:12] > 0):
        a, lp, val = net.act_forced(obs, 2)
        return a, lp, val, step_count, stuck_count, recovery_steps

    if any(obs[0:4] > 0):
        a, lp, val = net.act_forced(obs, 1)
        return a, lp, val, step_count, stuck_count, recovery_steps

    if any(obs[12:16] > 0):
        a, lp, val = net.act_forced(obs, 3)
        return a, lp, val, step_count, stuck_count, recovery_steps

    # all dark — sweep pattern
    # rotate 10 steps then FW 2 steps to reposition without hitting boundary
    sweep_pos = step_count % 12
    if sweep_pos < 10:
        # alternate L45/R45 sweep direction every full sweep cycle
        rot = 0 if (step_count // 12) % 2 == 0 else 4
        a, lp, val = net.act_forced(obs, rot)
    else:
        a, lp, val = net.act_forced(obs, 2)

    step_count += 1
    return a, lp, val, step_count, stuck_count, recovery_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",      type=str,   required=True)
    ap.add_argument("--out",            type=str,   default="weights_ppo_clean.pth")
    ap.add_argument("--episodes",       type=int,   default=4000)
    ap.add_argument("--max_steps",      type=int,   default=1000)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--scaling_factor", type=int,   default=5)
    ap.add_argument("--hidden",         type=int,   default=128)
    ap.add_argument("--lr",             type=float, default=1e-4)
    ap.add_argument("--gamma",          type=float, default=0.99)
    ap.add_argument("--lam",            type=float, default=0.95)
    ap.add_argument("--clip",           type=float, default=0.2)
    ap.add_argument("--ent_coef",       type=float, default=0.05)
    ap.add_argument("--vf_coef",        type=float, default=0.5)
    ap.add_argument("--ppo_epochs",     type=int,   default=2)
    ap.add_argument("--minibatch",      type=int,   default=64)
    ap.add_argument("--push_repeat",    type=int,   default=10)
    ap.add_argument("--seed",           type=int,   default=42)
    ap.add_argument("--render",         action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)
    net    = ActorCritic(hidden=args.hidden)
    opt    = optim.Adam(net.parameters(), lr=args.lr)

    stages = [
        (800,  0, False),
        (1600, 0, args.wall_obstacles),
        (2800, 2, args.wall_obstacles),
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

    def get_env_phase(ep):
        for limit, diff, wall in stages:
            if ep < limit:
                return envs[(diff, wall)], f"d{diff}{'w' if wall else ''}"
        return envs[(3, args.wall_obstacles)], "d3w"

    for ep in range(args.episodes):
        env, phase = get_env_phase(ep)
        obs        = np.array(env.reset(seed=args.seed + ep), dtype=np.float32)

        step_count     = 0
        stuck_count    = 0
        recovery_steps = 0

        obs_buf  = []
        act_buf  = []
        rew_buf  = []
        val_buf  = []
        lp_buf   = []
        done_buf = []

        ep_ret   = 0.0
        last_val = 0.0

        for _ in range(args.max_steps):
            a, lp, val, step_count, stuck_count, recovery_steps = get_action(
                obs, net, step_count, stuck_count, recovery_steps
            )

            obs_buf.append(obs.copy())
            act_buf.append(a)
            lp_buf.append(lp)
            val_buf.append(val)

            obs2, r, done = env.step(ACTIONS[a], render=args.render)
            ep_ret       += r
            rew_buf.append(float(r))
            done_buf.append(float(done))

            if r >= 100:
                for _ in range(args.push_repeat - 1):
                    obs_buf.append(obs.copy())
                    act_buf.append(a)
                    lp_buf.append(lp)
                    val_buf.append(val)
                    rew_buf.append(float(r))
                    done_buf.append(float(done))

            obs = np.array(obs2, dtype=np.float32)

            if done:
                last_val = 0.0
                break
        else:
            with torch.no_grad():
                x        = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                _, val_t = net(x)
                last_val = float(val_t.detach())

        advs, rets = compute_gae(
            np.array(rew_buf), val_buf,
            np.array(done_buf), last_val,
            args.gamma, args.lam,
        )

        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        obs_t  = torch.tensor(np.array(obs_buf),  dtype=torch.float32)
        act_t  = torch.tensor(np.array(act_buf),  dtype=torch.long)
        lp_old = torch.tensor(np.array(lp_buf),   dtype=torch.float32)
        adv_t  = torch.tensor(advs,               dtype=torch.float32)
        ret_t  = torch.tensor(rets,               dtype=torch.float32)

        n = len(obs_t)
        for _ in range(args.ppo_epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, args.minibatch):
                mb                    = idx[start:start+args.minibatch]
                lp_new, vals, entropy = net.evaluate(obs_t[mb], act_t[mb])
                ratio                 = torch.exp(lp_new - lp_old[mb])
                clip                  = torch.clamp(ratio, 1-args.clip, 1+args.clip)
                pg                    = -torch.min(ratio*adv_t[mb], clip*adv_t[mb]).mean()
                vf                    = nn.functional.mse_loss(vals, ret_t[mb])
                loss                  = pg + args.vf_coef*vf - args.ent_coef*entropy.mean()

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

        if (ep + 1) % 50 == 0:
            print(f"ep {ep+1:4d}/{args.episodes}  [{phase}]  return={ep_ret:10.1f}")
            torch.save(net.state_dict(), args.out)

        gc.collect()

    torch.save(net.state_dict(), args.out)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
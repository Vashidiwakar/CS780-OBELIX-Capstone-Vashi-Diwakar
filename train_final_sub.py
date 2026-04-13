"""Offline trainer: PPO for OBELIX Final Phase — v2.

Based on train_ppo_phase2.py which gave -840.9 on Codabench.

Key improvements over v1:
  - L22/R22 instead of L45/R45 for dark rotation — sensors fire for longer
    per rotation step, agent doesn't overshoot the box
  - Sensor persistence: if sensor fired in last 3 steps, keep moving toward it
    even if it goes dark briefly (solves the 1-step sensor flash problem)
  - Longer dark phase before switching direction (24 steps instead of 16)
  - When sonar fires: force FW for 3 steps instead of asking network
    (network learned to rotate even when box is ahead)

Curriculum:
  Episodes 0         → ep_switch1: difficulty=0, no wall
  Episodes ep_switch1 → ep_switch2: difficulty=2, wall
  Episodes ep_switch2 → end:        difficulty=3, wall

Example:
  python train_ppo_final.py --obelix_py ./obelix.py --out weights_ppo_final.pth --wall_obstacles
"""

from __future__ import annotations
import argparse, random, gc

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

ACTIONS   = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)


class ActorCritic(nn.Module):
    """Shared trunk with separate actor and critic heads."""
    def __init__(self, in_dim=18, n_actions=5, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor  = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        feat   = self.shared(x)
        logits = self.actor(feat)
        value  = self.critic(feat).squeeze(-1)
        return logits, value

    def selectAction(self, s):
        """Stochastic action — used during training."""
        logits, value = self.forward(s)
        probs  = F.softmax(logits, dim=-1)
        dist   = torch.distributions.Categorical(probs)
        a      = dist.sample()
        return a.item(), dist.log_prob(a), value, dist.entropy()

    def selectGreedyAction(self, s):
        """Greedy action — used during evaluation."""
        with torch.no_grad():
            logits, _ = self.forward(s)
        return int(torch.argmax(logits, dim=-1).item())

    def evaluate(self, states, actions):
        logits, values = self.forward(states)
        probs   = F.softmax(logits, dim=-1)
        dist    = torch.distributions.Categorical(probs)
        logp    = dist.log_prob(actions)
        entropy = dist.entropy()
        return logp, values, entropy


def computeGAE(rewards, values, dones, gamma, lam):
    """Generalized Advantage Estimation."""
    advantages = []
    gae        = 0.0
    next_value = 0.0

    for t in reversed(range(len(rewards))):
        mask   = 1.0 - float(dones[t])
        delta  = rewards[t] + gamma * next_value * mask - values[t]
        gae    = delta + gamma * lam * mask * gae
        advantages.insert(0, gae)
        next_value = values[t]

    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns    = advantages + torch.tensor(values, dtype=torch.float32)

    if advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


def select_action_explore(net, obs, dark_steps, dark_dir, fw_persist, last_sensor_side):
    """Smart exploration with sensor persistence and fine-grain rotation.
    
    Key fixes:
    1. L22/R22 for rotation — half the angle so sensors fire longer per step
    2. Sensor persistence — if sensor fired recently keep moving toward it
    3. Forward sensors → force FW directly instead of asking network
    4. Side sensors → force L22/R22 toward box then FW
    """
    # stuck — escape wall with fine rotation
    if obs[17] == 1:
        return random.choice([1, 3]), None, None, None, 0, last_sensor_side  # L22 or R22

    # IR fires — box directly ahead, force FW
    if obs[16] == 1:
        s    = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        a, logp, val, ent = net.selectAction(s)
        return 2, logp, val, ent, 3, last_sensor_side   # FW + persist 3 steps

    # sensor persistence — box was detected recently, keep moving toward it
    if fw_persist > 0:
        s    = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        a, logp, val, ent = net.selectAction(s)
        return 2, logp, val, ent, fw_persist - 1, last_sensor_side   # keep FW

    # forward sonars active — force FW
    if any(obs[4:12] == 1):
        s    = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        a, logp, val, ent = net.selectAction(s)
        return 2, logp, val, ent, 3, 0   # FW + persist 3 steps

    # left sonars active — turn left then forward
    if any(obs[0:4] == 1):
        s    = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        a, logp, val, ent = net.selectAction(s)
        return 1, logp, val, ent, 2, -1   # L22 + persist 2 FW steps

    # right sonars active — turn right then forward
    if any(obs[12:16] == 1):
        s    = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        a, logp, val, ent = net.selectAction(s)
        return 3, logp, val, ent, 2, 1   # R22 + persist 2 FW steps

    # all dark — fine-grain rotation sweep with L22/R22
    # 24 steps per direction (was 16 with L45) — covers same angle but slower
    phase = dark_steps % 24
    if phase < 12:
        rot = 1 if dark_dir % 2 == 0 else 3   # L22 or R22
    else:
        rot = 3 if dark_dir % 2 == 0 else 1   # R22 or L22
    return rot, None, None, None, 0, last_sensor_side


def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",      type=str,   required=True)
    ap.add_argument("--out",            type=str,   default="weights_ppo_final.pth")
    ap.add_argument("--pretrained",     type=str,   default=None)
    ap.add_argument("--episodes",       type=int,   default=3000)
    ap.add_argument("--max_steps",      type=int,   default=1000)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--scaling_factor", type=int,   default=5)
    ap.add_argument("--arena_size",     type=int,   default=500)
    ap.add_argument("--ep_switch1",     type=int,   default=800)
    ap.add_argument("--ep_switch2",     type=int,   default=2000)

    ap.add_argument("--gamma",          type=float, default=0.99)
    ap.add_argument("--lam",            type=float, default=0.95)
    ap.add_argument("--lr",             type=float, default=3e-4)
    ap.add_argument("--hidden",         type=int,   default=128)
    ap.add_argument("--clip_eps",       type=float, default=0.2)
    ap.add_argument("--ppo_epochs",     type=int,   default=4)
    ap.add_argument("--batch_size",     type=int,   default=64)
    ap.add_argument("--critic_coef",    type=float, default=0.5)
    ap.add_argument("--entropy_coef",   type=float, default=0.01)
    ap.add_argument("--reward_scale",   type=float, default=500.0)
    ap.add_argument("--seed",           type=int,   default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    net       = ActorCritic(in_dim=18, n_actions=N_ACTIONS, hidden=args.hidden)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # load pretrained weights if provided
    if args.pretrained is not None:
        sd = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        net.load_state_dict(sd)
        print(f"Loaded pretrained weights from {args.pretrained}")

    # create all envs once outside loop
    env_d0 = OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=500,
        max_steps=args.max_steps,
        wall_obstacles=False,
        difficulty=0,
        box_speed=2,
        seed=args.seed,
    )
    env_d2 = OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=500,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=2,
        box_speed=2,
        seed=args.seed + 1,
    )
    env_d3 = OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=500,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=3,
        box_speed=2,
        seed=args.seed + 2,
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

        s    = torch.tensor(
            np.array(env.reset(seed=args.seed + ep), dtype=np.float32)
        )
        done = False

        states, actions, rewards, log_probs_old = [], [], [], []
        values_list, dones_list                 = [], []
        ep_ret          = 0.0
        dark_steps      = 0
        dark_dir        = ep % 4
        fw_persist      = 0
        last_sensor_side = 0

        while not done:
            obs_np = s.numpy()

            # track dark steps
            if np.any(obs_np[:16] == 1) or obs_np[16] == 1:
                dark_steps = 0
            else:
                dark_steps += 1

            a, logp, val, ent, fw_persist, last_sensor_side = select_action_explore(
                net, obs_np, dark_steps, dark_dir, fw_persist, last_sensor_side
            )

            s2_raw, r, done = env.step(ACTIONS[a], render=False)
            s2     = torch.tensor(np.array(s2_raw, dtype=np.float32))
            r_s    = float(r) / args.reward_scale
            ep_ret += float(r)

            if logp is not None:
                states.append(s)
                actions.append(a)
                rewards.append(r_s)
                log_probs_old.append(logp.detach())
                values_list.append(val.item() if hasattr(val, 'item') else float(val))
                dones_list.append(done)

            s = s2

        # PPO update
        if len(states) > 1:
            advantages, returns = computeGAE(
                rewards, values_list, dones_list, args.gamma, args.lam
            )

            states_t    = torch.stack(states)
            actions_t   = torch.tensor(actions, dtype=torch.long)
            old_logps_t = torch.stack(log_probs_old)

            for _ in range(args.ppo_epochs):
                indices = np.random.permutation(len(states))
                for start in range(0, len(states), args.batch_size):
                    idx = indices[start:start + args.batch_size]
                    if len(idx) < 2:
                        continue

                    sb  = states_t[idx]
                    ab  = actions_t[idx]
                    adv = advantages[idx]
                    ret = returns[idx]
                    olp = old_logps_t[idx]

                    logp_new, values_new, entropy = net.evaluate(sb, ab)

                    ratio        = torch.exp(logp_new - olp)
                    surr1        = ratio * adv
                    surr2        = torch.clamp(ratio, 1-args.clip_eps, 1+args.clip_eps) * adv
                    actor_loss   = -torch.min(surr1, surr2).mean()
                    critic_loss  = F.mse_loss(values_new, ret)
                    entropy_loss = -entropy.mean()

                    loss = actor_loss + args.critic_coef * critic_loss + args.entropy_coef * entropy_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    optimizer.step()

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{args.episodes} [{phase}] return={ep_ret:.1f} traj_len={len(states)}")
            torch.save(net.state_dict(), args.out)
            gc.collect()

    torch.save(net.state_dict(), args.out)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
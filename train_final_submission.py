<<<<<<< HEAD
"""Offline trainer: PPO for OBELIX Phase 2 (Blinking Box).

Run locally to create weights.pth, then submit agent_ppo_phase2.py + weights.pth.

Example:
  python train_ppo_phase2.py --obelix_py ./obelix.py --out weights_ppo_phase2.pth --wall_obstacles --episodes 2000 --ep_switch 1000


                    ALGORITHM: PPO (PROXIMAL POLICY OPTIMIZATION) — PHASE 2


PPO improves over A2C/REINFORCE by preventing destructively large policy updates.

REINFORCE/A2C problem: if the policy update is too large, the new policy
can be much worse — and there is no way to undo it. This is why we saw
catastrophic plateaus at -300/-800 in earlier experiments.

PPO solution — clip the policy ratio to stay within [1-clip_eps, 1+clip_eps]:
  ratio     = pi_new(a|s) / pi_old(a|s)
  clipped   = clip(ratio, 1-clip_eps, 1+clip_eps)
  actor_loss = -mean(min(ratio * A, clipped * A))

This "trust region" ensures the policy never moves too far in one update.

PPO also reuses each trajectory for K epochs (here K=4), making it more
sample-efficient than A2C which discards the trajectory after one update.

GAE (Generalized Advantage Estimation) reduces variance in advantage estimates:
  A_t = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
  delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
  lambda=0.95 balances bias and variance

Curriculum:
  Episodes 0 to ep_switch    -> difficulty=0 (static box, build base policy)
  Episodes ep_switch to end  -> difficulty=2 (blinking box, fine-tune)

Reference:
  Schulman et al. (2017) https://arxiv.org/pdf/1707.06347
"""

from __future__ import annotations
import argparse, random, gc
from itertools import count

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)


class ActorCritic(nn.Module):
    """Shared trunk with separate actor and critic heads."""
    def __init__(self, in_dim=18, n_actions=5, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor  = nn.Linear(hidden, n_actions)  # policy logits
        self.critic = nn.Linear(hidden, 1)           # state value V(s)

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
    """
    Generalized Advantage Estimation (GAE).
    A_t = sum_{l=0}^{inf} (gamma*lambda)^l * delta_{t+l}
    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    """
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

    # Normalise advantages
    if advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


def select_action_explore(net, obs, dark_steps, dark_dir):
    """
    Smart exploration overrides on top of network stochastic action.
    Prevents getting stuck against walls and helps find box.
    """
    # Stuck — escape wall
    if obs[17] == 1:
        return random.choice([0, 4]), None, None, None   # L45 or R45

    # IR fires — box directly ahead
    if obs[16] == 1:
        s    = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        a, logp, val, ent = net.selectAction(s)
        return 2, logp, val, ent   # force FW but keep logp for training

    # Any sonar active — stochastic from network
    if np.any(obs[:16] == 1):
        s    = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        a, logp, val, ent = net.selectAction(s)
        return a, logp, val, ent

    # All dark — systematic rotation
    phase = dark_steps % 16
    if phase < 8:
        return (0 if dark_dir % 2 == 0 else 4), None, None, None
    else:
        return (4 if dark_dir % 2 == 0 else 0), None, None, None


def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",      type=str,   required=True)
    ap.add_argument("--out",            type=str,   default="weights_ppo_phase2.pth")
    ap.add_argument("--episodes",       type=int,   default=2000)
    ap.add_argument("--max_steps",      type=int,   default=1000)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--scaling_factor", type=int,   default=5)
    ap.add_argument("--arena_size",     type=int,   default=500)
    ap.add_argument("--ep_switch",      type=int,   default=1000)

    ap.add_argument("--gamma",          type=float, default=0.99)
    ap.add_argument("--lam",            type=float, default=0.95,
                    help="GAE lambda")
    ap.add_argument("--lr",             type=float, default=3e-4)
    ap.add_argument("--hidden",         type=int,   default=128)
    ap.add_argument("--clip_eps",       type=float, default=0.2)
    ap.add_argument("--ppo_epochs",     type=int,   default=4)
    ap.add_argument("--batch_size",     type=int,   default=64)
    ap.add_argument("--critic_coef",    type=float, default=0.5)
    ap.add_argument("--entropy_coef",   type=float, default=0.01)
    ap.add_argument("--reward_scale",   type=float, default=500.0,
                    help="Divide rewards by this to prevent critic explosion")
    ap.add_argument("--seed",           type=int,   default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    net       = ActorCritic(in_dim=18, n_actions=N_ACTIONS, hidden=args.hidden)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

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

        s    = torch.tensor(
            np.array(env.reset(seed=args.seed + ep), dtype=np.float32)
        )
        done = False

        # Storage for trajectory
        states, actions, rewards, log_probs_old = [], [], [], []
        values_list, dones_list, entropies      = [], [], []
        ep_ret     = 0.0
        dark_steps = 0
        dark_dir   = ep % 4

        # Collect one full episode 
        while not done:
            obs_np = s.numpy()

            # Track dark steps
            if np.any(obs_np[:16] == 1) or obs_np[16] == 1:
                dark_steps = 0
            else:
                dark_steps += 1

            a, logp, val, ent = select_action_explore(net, obs_np, dark_steps, dark_dir)

            s2_raw, r, done = env.step(ACTIONS[a], render=False)
            s2  = torch.tensor(np.array(s2_raw, dtype=np.float32))
            r_s = float(r) / args.reward_scale   # scale reward to prevent explosion
            ep_ret += float(r)

            # Only store transitions where network made the decision
            if logp is not None:
                states.append(s)
                actions.append(a)
                rewards.append(r_s)
                log_probs_old.append(logp.detach())
                values_list.append(val.item() if hasattr(val, 'item') else float(val))
                dones_list.append(done)
                entropies.append(ent)

            s = s2

        # PPO update (only if we have trajectory data)
        if len(states) > 1:
            advantages, returns = computeGAE(
                rewards, values_list, dones_list, args.gamma, args.lam
            )

            states_t      = torch.stack(states)
            actions_t     = torch.tensor(actions, dtype=torch.long)
            old_logps_t   = torch.stack(log_probs_old)

            # Multiple PPO epochs over the same trajectory
            for _ in range(args.ppo_epochs):
                # Mini-batch updates
                indices = np.random.permutation(len(states))
                for start in range(0, len(states), args.batch_size):
                    idx = indices[start : start + args.batch_size]
                    if len(idx) < 2:
                        continue

                    sb  = states_t[idx]
                    ab  = actions_t[idx]
                    adv = advantages[idx]
                    ret = returns[idx]
                    olp = old_logps_t[idx]

                    logp_new, values_new, entropy = net.evaluate(sb, ab)

                    # PPO clipped objective
                    ratio  = torch.exp(logp_new - olp)
                    surr1  = ratio * adv
                    surr2  = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * adv
                    actor_loss  = -torch.min(surr1, surr2).mean()
                    critic_loss = F.mse_loss(values_new, ret)
                    entropy_loss = -entropy.mean()

                    loss = actor_loss + args.critic_coef * critic_loss + args.entropy_coef * entropy_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    optimizer.step()

        phase = "d0" if ep < args.ep_switch else "d2"
        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{args.episodes} [{phase}] return={ep_ret:.1f} traj_len={len(states)}")
            torch.save(net.state_dict(), args.out)
            gc.collect()

    torch.save(net.state_dict(), args.out)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
=======
"""Offline trainer: PPO for OBELIX Phase 2 (Blinking Box) --> GAVE BEST SCORE IN FINAL TESTING PHASE(-1609.51)
# train_ppo_phase2.py renamed as train_final_submission.py
# agent_ppo_phase2.py renamed as agent_final_submission.py
# weights_ppo_phase2.pth renamed as weights_final_submission.pth

Run locally to create weights.pth, then submit agent_ppo_phase2.py + weights.pth.

Example:
  python train_ppo_phase2.py --obelix_py ./obelix.py --out weights_ppo_phase2.pth --wall_obstacles --episodes 2000 --ep_switch 1000


                    ALGORITHM: PPO (PROXIMAL POLICY OPTIMIZATION) — PHASE 2


PPO improves over A2C/REINFORCE by preventing destructively large policy updates.

REINFORCE/A2C problem: if the policy update is too large, the new policy
can be much worse — and there is no way to undo it. This is why we saw
catastrophic plateaus at -300/-800 in earlier experiments.

PPO solution — clip the policy ratio to stay within [1-clip_eps, 1+clip_eps]:
  ratio     = pi_new(a|s) / pi_old(a|s)
  clipped   = clip(ratio, 1-clip_eps, 1+clip_eps)
  actor_loss = -mean(min(ratio * A, clipped * A))

This "trust region" ensures the policy never moves too far in one update.

PPO also reuses each trajectory for K epochs (here K=4), making it more
sample-efficient than A2C which discards the trajectory after one update.

GAE (Generalized Advantage Estimation) reduces variance in advantage estimates:
  A_t = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
  delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
  lambda=0.95 balances bias and variance

Curriculum:
  Episodes 0 to ep_switch    -> difficulty=0 (static box, build base policy)
  Episodes ep_switch to end  -> difficulty=2 (blinking box, fine-tune)

"""

from __future__ import annotations
import argparse, random, gc
from itertools import count

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)


class ActorCritic(nn.Module):
    """Shared trunk with separate actor and critic heads."""
    def __init__(self, in_dim=18, n_actions=5, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor  = nn.Linear(hidden, n_actions)  # policy logits
        self.critic = nn.Linear(hidden, 1)           # state value V(s)

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
    """
    Generalized Advantage Estimation (GAE).
    A_t = sum_{l=0}^{inf} (gamma*lambda)^l * delta_{t+l}
    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    """
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

    # Normalize advantages
    if advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


def select_action_explore(net, obs, dark_steps, dark_dir):
    """
    Smart exploration overrides on top of network stochastic action.
    Prevents getting stuck against walls and helps find box.
    """
    # Stuck — escape wall
    if obs[17] == 1:
        return random.choice([0, 4]), None, None, None   # L45 or R45

    # IR fires — box directly ahead
    if obs[16] == 1:
        s    = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        a, logp, val, ent = net.selectAction(s)
        return 2, logp, val, ent   # force FW but keep logp for training

    # Any sonar active — stochastic from network
    if np.any(obs[:16] == 1):
        s    = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        a, logp, val, ent = net.selectAction(s)
        return a, logp, val, ent

    # All dark — systematic rotation
    phase = dark_steps % 16
    if phase < 8:
        return (0 if dark_dir % 2 == 0 else 4), None, None, None
    else:
        return (4 if dark_dir % 2 == 0 else 0), None, None, None


def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",      type=str,   required=True)
    ap.add_argument("--out",            type=str,   default="weights_ppo_phase2.pth")
    ap.add_argument("--episodes",       type=int,   default=2000)
    ap.add_argument("--max_steps",      type=int,   default=1000)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--scaling_factor", type=int,   default=5)
    ap.add_argument("--arena_size",     type=int,   default=500)
    ap.add_argument("--ep_switch",      type=int,   default=1000)

    ap.add_argument("--gamma",          type=float, default=0.99)
    ap.add_argument("--lam",            type=float, default=0.95,
                    help="GAE lambda")
    ap.add_argument("--lr",             type=float, default=3e-4)
    ap.add_argument("--hidden",         type=int,   default=128)
    ap.add_argument("--clip_eps",       type=float, default=0.2)
    ap.add_argument("--ppo_epochs",     type=int,   default=4)
    ap.add_argument("--batch_size",     type=int,   default=64)
    ap.add_argument("--critic_coef",    type=float, default=0.5)
    ap.add_argument("--entropy_coef",   type=float, default=0.01)
    ap.add_argument("--reward_scale",   type=float, default=500.0,
                    help="Divide rewards by this to prevent critic explosion")
    ap.add_argument("--seed",           type=int,   default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    net       = ActorCritic(in_dim=18, n_actions=N_ACTIONS, hidden=args.hidden)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

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

        s    = torch.tensor(
            np.array(env.reset(seed=args.seed + ep), dtype=np.float32)
        )
        done = False

        # Storage for trajectory
        states, actions, rewards, log_probs_old = [], [], [], []
        values_list, dones_list, entropies      = [], [], []
        ep_ret     = 0.0
        dark_steps = 0
        dark_dir   = ep % 4

        # Collect one full episode 
        while not done:
            obs_np = s.numpy()

            # Track dark steps
            if np.any(obs_np[:16] == 1) or obs_np[16] == 1:
                dark_steps = 0
            else:
                dark_steps += 1

            a, logp, val, ent = select_action_explore(net, obs_np, dark_steps, dark_dir)

            s2_raw, r, done = env.step(ACTIONS[a], render=False)
            s2  = torch.tensor(np.array(s2_raw, dtype=np.float32))
            r_s = float(r) / args.reward_scale   # scale reward to prevent explosion
            ep_ret += float(r)

            # Only store transitions where network made the decision
            if logp is not None:
                states.append(s)
                actions.append(a)
                rewards.append(r_s)
                log_probs_old.append(logp.detach())
                values_list.append(val.item() if hasattr(val, 'item') else float(val))
                dones_list.append(done)
                entropies.append(ent)

            s = s2

        # PPO update (only if we have trajectory data)
        if len(states) > 1:
            advantages, returns = computeGAE(
                rewards, values_list, dones_list, args.gamma, args.lam
            )

            states_t      = torch.stack(states)
            actions_t     = torch.tensor(actions, dtype=torch.long)
            old_logps_t   = torch.stack(log_probs_old)

            # Multiple PPO epochs over the same trajectory
            for _ in range(args.ppo_epochs):
                # Mini-batch updates
                indices = np.random.permutation(len(states))
                for start in range(0, len(states), args.batch_size):
                    idx = indices[start : start + args.batch_size]
                    if len(idx) < 2:
                        continue

                    sb  = states_t[idx]
                    ab  = actions_t[idx]
                    adv = advantages[idx]
                    ret = returns[idx]
                    olp = old_logps_t[idx]

                    logp_new, values_new, entropy = net.evaluate(sb, ab)

                    # PPO clipped objective
                    ratio  = torch.exp(logp_new - olp)
                    surr1  = ratio * adv
                    surr2  = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * adv
                    actor_loss  = -torch.min(surr1, surr2).mean()
                    critic_loss = F.mse_loss(values_new, ret)
                    entropy_loss = -entropy.mean()

                    loss = actor_loss + args.critic_coef * critic_loss + args.entropy_coef * entropy_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    optimizer.step()

        phase = "d0" if ep < args.ep_switch else "d2"
        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{args.episodes} [{phase}] return={ep_ret:.1f} traj_len={len(states)}")
            torch.save(net.state_dict(), args.out)
            gc.collect()

    torch.save(net.state_dict(), args.out)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
>>>>>>> 0a9ce09456a66f63d4ad6a97851f1d6a8aa7494f

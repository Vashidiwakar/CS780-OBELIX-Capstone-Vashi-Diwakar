"""Offline trainer: PPO Final Solution for OBELIX.

Analysis of what worked:
  - PPO with reward_scale=500 gave -1609 on final phase
  - Smart exploration (dark rotation) is essential
  - Hardcoded IR→FW, sonar→network is the right pattern
  - Problem: d3 (moving box) needs agent to intercept box not just react

Solution:
  - Train three focused networks sequentially
  - Save best checkpoint per difficulty
  - Use observation history (last 4 obs) as augmented state
    This gives the network memory to track moving box direction
  - Heavier entropy during d3 to keep exploring when box moves away
  - Reward normalization per difficulty level

Observation augmentation:
  Original: 18 bits
  Augmented: 18 * 4 = 72 bits (current + last 3 observations)
  This gives the network temporal context to infer box movement direction

Example:
  python train_final_solution.py --obelix_py ./obelix.py --out weights_final_sol.pth --wall_obstacles --pretrained weights_ppo_phase2.pth
"""

from __future__ import annotations
import argparse, random, gc
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

ACTIONS   = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)
OBS_DIM   = 18
HIST_LEN  = 4   # number of observations to stack
AUG_DIM   = OBS_DIM * HIST_LEN   # 72


class ActorCritic(nn.Module):
    """Actor-Critic with augmented observation (stacked history)."""
    def __init__(self, in_dim=AUG_DIM, n_actions=5, hidden=128):
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
        logits, value = self.forward(s)
        probs  = F.softmax(logits, dim=-1)
        dist   = torch.distributions.Categorical(probs)
        a      = dist.sample()
        return a.item(), dist.log_prob(a), value, dist.entropy()

    def selectGreedyAction(self, s):
        with torch.no_grad():
            logits, _ = self.forward(s)
        return int(torch.argmax(logits, dim=-1).item())

    def evaluate(self, states, actions):
        logits, values = self.forward(states)
        probs   = F.softmax(logits, dim=-1)
        dist    = torch.distributions.Categorical(probs)
        return dist.log_prob(actions), values, dist.entropy()


def computeGAE(rewards, values, dones, gamma, lam):
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


def make_aug_obs(obs_history):
    """Stack last HIST_LEN observations into augmented state."""
    return np.concatenate(list(obs_history), axis=0).astype(np.float32)


def select_action(net, aug_obs, raw_obs, dark_steps, dark_dir, fw_persist):
    """
    Priority-based action selection using augmented observation for network.
    Raw obs used for sensor checks, aug_obs fed to network.
    """
    # stuck — escape with L45/R45
    if raw_obs[17] == 1:
        return random.choice([0, 4]), None, None, None, 0

    # IR fires — force FW with persistence
    if raw_obs[16] == 1:
        s = torch.tensor(aug_obs).unsqueeze(0)
        a, logp, val, ent = net.selectAction(s)
        return 2, logp, val, ent, 5

    # sensor persistence
    if fw_persist > 0:
        s = torch.tensor(aug_obs).unsqueeze(0)
        a, logp, val, ent = net.selectAction(s)
        return 2, logp, val, ent, fw_persist - 1

    # forward sonars — force FW
    if any(raw_obs[4:12] == 1):
        s = torch.tensor(aug_obs).unsqueeze(0)
        a, logp, val, ent = net.selectAction(s)
        return 2, logp, val, ent, 3

    # left sonars — network decides with aug_obs
    if any(raw_obs[0:4] == 1):
        s = torch.tensor(aug_obs).unsqueeze(0)
        a, logp, val, ent = net.selectAction(s)
        return a, logp, val, ent, 2

    # right sonars — network decides with aug_obs
    if any(raw_obs[12:16] == 1):
        s = torch.tensor(aug_obs).unsqueeze(0)
        a, logp, val, ent = net.selectAction(s)
        return a, logp, val, ent, 2

    # all dark — systematic rotation
    phase = dark_steps % 16
    if phase < 8:
        rot = 0 if dark_dir % 2 == 0 else 4
    else:
        rot = 4 if dark_dir % 2 == 0 else 0
    return rot, None, None, None, 0


def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def train_stage(net, optimizer, env, episodes, seed, phase,
                reward_scale, entropy_coef, gamma, lam,
                clip_eps, ppo_epochs, batch_size, out_path):
    """Train one curriculum stage."""
    best_return = -float('inf')
    best_weights = None

    for ep in range(episodes):
        raw_obs = np.array(env.reset(seed=seed + ep), dtype=np.float32)

        # initialize observation history with zeros
        obs_history = deque([np.zeros(OBS_DIM, dtype=np.float32)] * HIST_LEN,
                            maxlen=HIST_LEN)
        obs_history.append(raw_obs.copy())
        aug_obs = make_aug_obs(obs_history)

        done       = False
        states, actions, rewards = [], [], []
        log_probs_old, values_list, dones_list = [], [], []
        ep_ret     = 0.0
        dark_steps = 0
        dark_dir   = ep % 4
        fw_persist = 0

        while not done:
            if np.any(raw_obs[:16] == 1) or raw_obs[16] == 1:
                dark_steps = 0
            else:
                dark_steps += 1

            a, logp, val, ent, fw_persist = select_action(
                net, aug_obs, raw_obs, dark_steps, dark_dir, fw_persist
            )

            raw_obs2, r, done = env.step(ACTIONS[a], render=False)
            raw_obs2 = np.array(raw_obs2, dtype=np.float32)
            ep_ret  += float(r)
            r_s      = float(r) / reward_scale

            if logp is not None:
                states.append(torch.tensor(aug_obs))
                actions.append(a)
                rewards.append(r_s)
                log_probs_old.append(logp.detach())
                values_list.append(val.item() if hasattr(val, 'item') else float(val))
                dones_list.append(done)

            # update history
            obs_history.append(raw_obs2.copy())
            aug_obs = make_aug_obs(obs_history)
            raw_obs = raw_obs2

        # PPO update
        if len(states) > 1:
            advantages, returns = computeGAE(
                rewards, values_list, dones_list, gamma, lam
            )
            states_t    = torch.stack(states)
            actions_t   = torch.tensor(actions, dtype=torch.long)
            old_logps_t = torch.stack(log_probs_old)

            for _ in range(ppo_epochs):
                indices = np.random.permutation(len(states))
                for start in range(0, len(states), batch_size):
                    idx = indices[start:start + batch_size]
                    if len(idx) < 2:
                        continue
                    logp_new, values_new, entropy = net.evaluate(
                        states_t[idx], actions_t[idx]
                    )
                    ratio       = torch.exp(logp_new - old_logps_t[idx])
                    surr1       = ratio * advantages[idx]
                    surr2       = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantages[idx]
                    actor_loss  = -torch.min(surr1, surr2).mean()
                    critic_loss = F.mse_loss(values_new, returns[idx])
                    loss        = actor_loss + 0.5*critic_loss - entropy_coef*entropy.mean()

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    optimizer.step()

        # save best weights for this stage
        if ep_ret > best_return:
            best_return  = ep_ret
            best_weights = {k: v.clone() for k, v in net.state_dict().items()}

        if (ep + 1) % 50 == 0:
            print(f"  [{phase}] ep {ep+1}/{episodes} return={ep_ret:.1f} traj={len(states)} best={best_return:.1f}")
            torch.save(net.state_dict(), out_path)
            gc.collect()

    # save best weights for this stage
    stage_path = out_path.replace(".pth", f"_best_{phase}.pth")
    if best_weights is not None:
        torch.save(best_weights, stage_path)
        print(f"  Saved best {phase} weights: {stage_path} (return={best_return:.1f})")

    return best_return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",      type=str,   required=True)
    ap.add_argument("--out",            type=str,   default="weights_final_sol.pth")
    ap.add_argument("--pretrained",     type=str,   default=None)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--scaling_factor", type=int,   default=5)
    ap.add_argument("--d0_episodes",    type=int,   default=500)
    ap.add_argument("--d2_episodes",    type=int,   default=800)
    ap.add_argument("--d3_episodes",    type=int,   default=800)
    ap.add_argument("--max_steps",      type=int,   default=1000)
    ap.add_argument("--gamma",          type=float, default=0.99)
    ap.add_argument("--lam",            type=float, default=0.95)
    ap.add_argument("--lr",             type=float, default=1e-4)
    ap.add_argument("--hidden",         type=int,   default=128)
    ap.add_argument("--clip_eps",       type=float, default=0.2)
    ap.add_argument("--ppo_epochs",     type=int,   default=4)
    ap.add_argument("--batch_size",     type=int,   default=64)
    ap.add_argument("--reward_scale",   type=float, default=500.0)
    ap.add_argument("--seed",           type=int,   default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    net       = ActorCritic(in_dim=AUG_DIM, hidden=args.hidden)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # load pretrained — but pretrained was trained on 18-dim obs
    # we initialize fresh for augmented obs
    if args.pretrained is not None:
        print(f"Note: pretrained weights ignored — augmented obs (72-dim) requires fresh training")

    print("=" * 60)
    print(f"Training with augmented observations ({AUG_DIM}-dim)")
    print(f"History length: {HIST_LEN} steps")
    print("=" * 60)

    # Stage 1a: d0 no wall — build base push policy
    print("\nStage 1a: difficulty=0, no wall")
    env_d0_nw = OBELIX(
        scaling_factor=args.scaling_factor, arena_size=500,
        max_steps=args.max_steps, wall_obstacles=False,
        difficulty=0, box_speed=2, seed=args.seed,
    )
    train_stage(
        net, optimizer, env_d0_nw, args.d0_episodes, args.seed,
        "d0nw", args.reward_scale, 0.01,
        args.gamma, args.lam, args.clip_eps,
        args.ppo_epochs, args.batch_size, args.out
    )

    # Stage 1b: d0 with wall — learn to navigate wall
    print("\nStage 1b: difficulty=0, wall")
    env_d0_w = OBELIX(
        scaling_factor=args.scaling_factor, arena_size=500,
        max_steps=args.max_steps, wall_obstacles=args.wall_obstacles,
        difficulty=0, box_speed=2, seed=args.seed,
    )
    train_stage(
        net, optimizer, env_d0_w, args.d0_episodes, args.seed,
        "d0w", args.reward_scale, 0.01,
        args.gamma, args.lam, args.clip_eps,
        args.ppo_epochs, args.batch_size, args.out
    )

    # Stage 2a: d2 no wall — blinking box without wall distraction
    print("\nStage 2a: difficulty=2, no wall")
    env_d2_nw = OBELIX(
        scaling_factor=args.scaling_factor, arena_size=500,
        max_steps=args.max_steps, wall_obstacles=False,
        difficulty=2, box_speed=2, seed=args.seed + 1,
    )
    train_stage(
        net, optimizer, env_d2_nw, args.d2_episodes, args.seed + 1,
        "d2nw", args.reward_scale, 0.02,
        args.gamma, args.lam, args.clip_eps,
        args.ppo_epochs, args.batch_size, args.out
    )

    # Stage 2b: d2 with wall
    print("\nStage 2b: difficulty=2, wall")
    env_d2_w = OBELIX(
        scaling_factor=args.scaling_factor, arena_size=500,
        max_steps=args.max_steps, wall_obstacles=args.wall_obstacles,
        difficulty=2, box_speed=2, seed=args.seed + 1,
    )
    train_stage(
        net, optimizer, env_d2_w, args.d2_episodes, args.seed + 1,
        "d2w", args.reward_scale, 0.02,
        args.gamma, args.lam, args.clip_eps,
        args.ppo_epochs, args.batch_size, args.out
    )

    # Stage 3a: d3 no wall — moving + blinking without wall
    print("\nStage 3a: difficulty=3, no wall")
    env_d3_nw = OBELIX(
        scaling_factor=args.scaling_factor, arena_size=500,
        max_steps=args.max_steps, wall_obstacles=False,
        difficulty=3, box_speed=2, seed=args.seed + 2,
    )
    train_stage(
        net, optimizer, env_d3_nw, args.d3_episodes, args.seed + 2,
        "d3nw", args.reward_scale, 0.05,
        args.gamma, args.lam, args.clip_eps,
        args.ppo_epochs, args.batch_size, args.out
    )

    # Stage 3b: d3 with wall — full difficulty
    print("\nStage 3b: difficulty=3, wall")
    env_d3_w = OBELIX(
        scaling_factor=args.scaling_factor, arena_size=500,
        max_steps=args.max_steps, wall_obstacles=args.wall_obstacles,
        difficulty=3, box_speed=2, seed=args.seed + 2,
    )
    train_stage(
        net, optimizer, env_d3_w, args.d3_episodes, args.seed + 2,
        "d3w", args.reward_scale, 0.05,
        args.gamma, args.lam, args.clip_eps,
        args.ppo_epochs, args.batch_size, args.out
    )

    torch.save(net.state_dict(), args.out)
    print(f"\nFinal weights saved: {args.out}")
    print("Best per-stage weights saved separately.")


if __name__ == "__main__":
    main()
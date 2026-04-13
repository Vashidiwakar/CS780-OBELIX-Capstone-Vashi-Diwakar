"""Offline trainer: A2C (Advantage Actor-Critic) for OBELIX.

Run locally to create weights.pth, then submit agent_a2c.py + weights.pth.

Example:
  python train_a2c.py --obelix_py ./obelix.py --out weights_a2c.pth --episodes 5000 --max_steps 300
  python train_a2c.py --obelix_py ./obelix.py --out weights_a2c.pth --episodes 5000 --max_steps 300 --wall_obstacles


                    ALGORITHM: A2C (Advantage Actor-Critic)


A2C improves over REINFORCE by adding a CRITIC (value network) alongside
the ACTOR (policy network):

  ACTOR:  π(a|s;θ) — learns WHICH action to take (same as REINFORCE)
  CRITIC: V(s;w)   — learns HOW GOOD each state is

The key improvement is the ADVANTAGE:
  A(s,a) = G_t - V(s)   (actual return minus what critic expected)

Instead of scaling gradients by raw returns (high variance in REINFORCE),
A2C scales by the advantage — a much lower-variance signal.

If critic expected -1 but agent got +100 (touched box):
  Advantage = +101 → strong update toward that action

If critic expected +100 and got +100:
  Advantage = 0 → no update needed, policy was already correct

Loss functions:
  Actor loss:  -log π(a|s) * A(s,a)   (policy gradient with advantage)
  Critic loss:  (G_t - V(s))^2         (MSE between predicted and actual return)
  Total loss:  actor_loss + critic_coef * critic_loss - entropy_coef * entropy

Entropy bonus encourages exploration — stops the policy collapsing to
always picking the same action.

Reference:
  Mnih et al. 2016 — Asynchronous Methods for Deep Reinforcement Learning.
  Sutton & Barto Chapter 13 — Policy Gradient Methods.
"""

from __future__ import annotations
import argparse, random, gc

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)


class ActorCritic(nn.Module):
    def __init__(self, inDim=18, outDim=5, hiddenDim=64):
        super().__init__()
        # Shared feature trunk
        self.shared = nn.Sequential(
            nn.Linear(inDim, hiddenDim),
            nn.ReLU(),
            nn.Linear(hiddenDim, hiddenDim),
            nn.ReLU(),
        )
        # Actor head — outputs action logits
        self.actor  = nn.Linear(hiddenDim, outDim)
        # Critic head — outputs state value V(s)
        self.critic = nn.Linear(hiddenDim, 1)

    def forward(self, s):
        feat   = self.shared(s)
        logits = self.actor(feat)
        value  = self.critic(feat).squeeze(-1)
        return logits, value

    def selectAction(self, s):
        """Stochastic action — used during training."""
        logits, value = self.forward(s)
        probs  = F.softmax(logits, dim=-1)
        dist   = torch.distributions.Categorical(probs)
        a      = dist.sample()
        logp_a = dist.log_prob(a)
        entropy = dist.entropy()
        return a.item(), logp_a, value, entropy

    def selectGreedyAction(self, s):
        """Greedy action — used during evaluation."""
        with torch.no_grad():
            logits, _ = self.forward(s)
        return int(torch.argmax(logits, dim=-1).item())


def getReturns(gamma, rewards):
    T       = len(rewards)
    returns = torch.zeros(T)
    G       = 0.0
    for t in reversed(range(T)):
        G          = rewards[t] + gamma * G
        returns[t] = G
    return returns


REWARD_SCALE = 500.0  # scale down rewards to prevent critic loss explosion


def trainA2C(rewards, logProbs, values, entropies, gamma,
             optimizerFn, critic_coef, entropy_coef):
    # Scale rewards down — OBELIX rewards are large (-200 stuck, +2000 success)
    # which causes MSE critic loss to explode into the millions
    scaled_rewards = [r / REWARD_SCALE for r in rewards]
    returns = getReturns(gamma, scaled_rewards)

    # Clip returns to prevent occasional huge episodes from destabilising critic
    returns = torch.clamp(returns, -10.0, 10.0)

    logProbs_t  = torch.stack(logProbs)
    values_t    = torch.stack(values)
    entropies_t = torch.stack(entropies)

    # Advantage = actual return - critic estimate
    advantages = returns - values_t.detach()

    # Normalise advantages to reduce variance
    if advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    actor_loss  = -(logProbs_t * advantages).mean()
    critic_loss = F.mse_loss(values_t, returns)
    entropy     = entropies_t.mean()

    loss = actor_loss + critic_coef * critic_loss - entropy_coef * entropy

    optimizerFn.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(optimizerFn.param_groups[0]['params'], 5.0)
    optimizerFn.step()

    return loss.item()


def evaluateAgent(env, model, max_steps, eval_episodes):
    rewards = []
    for _ in range(eval_episodes):
        rs   = 0.0
        s    = torch.tensor(np.array(env.reset(), dtype=np.float32))
        done = False
        for _ in range(max_steps):
            a           = model.selectGreedyAction(s)
            s2, r, done = env.step(ACTIONS[a], render=False)
            s           = torch.tensor(np.array(s2, dtype=np.float32))
            rs         += float(r)
            if done:
                break
        rewards.append(rs)
    return float(np.mean(rewards)), float(np.std(rewards))


def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",      type=str,   required=True)
    ap.add_argument("--out",            type=str,   default="weights_a2c.pth")
    ap.add_argument("--episodes",       type=int,   default=5000)
    ap.add_argument("--max_steps",      type=int,   default=800)
    ap.add_argument("--difficulty",     type=int,   default=0)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed",      type=int,   default=2)
    ap.add_argument("--scaling_factor", type=int,   default=1)
    ap.add_argument("--arena_size",     type=int,   default=500)

    ap.add_argument("--gamma",          type=float, default=0.99)
    ap.add_argument("--lr",             type=float, default=1e-4)
    ap.add_argument("--hidden",         type=int,   default=64)
    ap.add_argument("--critic_coef",    type=float, default=0.25)
    ap.add_argument("--entropy_coef",   type=float, default=0.05)
    ap.add_argument("--eps_start",      type=float, default=0.5)
    ap.add_argument("--eps_end",        type=float, default=0.05)
    ap.add_argument("--eps_decay",      type=int,   default=3000)
    ap.add_argument("--eval_episodes",  type=int,   default=1)
    ap.add_argument("--seed",           type=int,   default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    model     = ActorCritic(inDim=18, outDim=N_ACTIONS, hiddenDim=args.hidden)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    env = OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
        seed=args.seed,
    )

    rng = np.random.default_rng(args.seed)

    def get_eps(e):
        # Linear decay from eps_start to eps_end over eps_decay episodes
        if e >= args.eps_decay:
            return args.eps_end
        frac = e / args.eps_decay
        return args.eps_start + frac * (args.eps_end - args.eps_start)

    for e in range(args.episodes):
        obs  = np.array(env.reset(seed=e), dtype=np.float32)
        s    = torch.tensor(obs)
        done = False
        eps  = get_eps(e)

        rewards   = []
        logProbs  = []
        values    = []
        entropies = []
        ep_ret    = 0.0
        dark_steps = 0  # consecutive steps with no sensor signal

        for _ in range(args.max_steps):
            obs_np = s.numpy()

            # Hard override: stuck → rotate to escape wall
            if obs_np[17] == 1:
                forced = rng.choice([0, 4])  # L45 or R45
                a, logp_a, value, entropy = model.selectAction(s)
                a = int(forced)
                with torch.no_grad():
                    logits, value = model.forward(s)
                probs   = torch.softmax(logits, dim=-1)
                dist    = torch.distributions.Categorical(probs)
                logp_a  = dist.log_prob(torch.tensor(a))
                entropy = dist.entropy()

            # Hard override: IR fires → box is right there, go forward
            elif obs_np[16] == 1:
                a = 2  # FW
                with torch.no_grad():
                    logits, value = model.forward(s)
                probs   = torch.softmax(logits, dim=-1)
                dist    = torch.distributions.Categorical(probs)
                logp_a  = dist.log_prob(torch.tensor(a))
                entropy = dist.entropy()
                dark_steps = 0

            # Epsilon-greedy exploration when all sensors dark
            elif not any(obs_np[:17]) and rng.random() < eps:
                dark_steps += 1
                # Alternate direction every 8 dark steps for broad coverage
                forced = 0 if (dark_steps // 8) % 2 == 0 else 4  # L45 or R45
                a = int(forced)
                with torch.no_grad():
                    logits, value = model.forward(s)
                probs   = torch.softmax(logits, dim=-1)
                dist    = torch.distributions.Categorical(probs)
                logp_a  = dist.log_prob(torch.tensor(a))
                entropy = dist.entropy()

            else:
                dark_steps = 0
                a, logp_a, value, entropy = model.selectAction(s)

            s2, r, done = env.step(ACTIONS[a], render=False)
            s           = torch.tensor(np.array(s2, dtype=np.float32))
            rewards.append(float(r))
            logProbs.append(logp_a)
            values.append(value)
            entropies.append(entropy)
            ep_ret += float(r)
            if done:
                break

        loss = trainA2C(rewards, logProbs, values, entropies,
                        args.gamma, optimizer, args.critic_coef, args.entropy_coef)

        if (e + 1) % 50 == 0:
            eval_mean, eval_std = evaluateAgent(env, model, args.max_steps, args.eval_episodes)
            print(f"Episode {e+1}/{args.episodes} return={ep_ret:.1f} loss={loss:.4f} eps={eps:.3f} eval_mean={eval_mean:.1f} eval_std={eval_std:.1f}")
            torch.save(model.state_dict(), args.out)
            gc.collect()

    torch.save(model.state_dict(), args.out)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
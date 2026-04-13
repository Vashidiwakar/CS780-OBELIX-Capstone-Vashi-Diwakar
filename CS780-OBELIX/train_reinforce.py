"""Offline trainer: REINFORCE (Monte Carlo Policy Gradient) for OBELIX.

Run locally to create weights.pth, then submit agent_reinforce.py + weights.pth.

Example:
  python train_reinforce.py --obelix_py ./obelix.py --out weights_reinforce.pth --episodes 3000 --difficulty 0
  python train_reinforce.py --obelix_py ./obelix.py --out weights.pth --episodes 3000 --difficulty 0 --wall_obstacles


                    ALGORITHM: REINFORCE (Monte Carlo Policy Gradient)


REINFORCE differs fundamentally from DQN — it learns a POLICY directly
instead of learning Q-values:

  DQN:       learns Q(s,a) → derive policy by taking argmax
  REINFORCE: learns π(a|s;θ) → policy is the network output directly

The policy network outputs a probability distribution over actions.
Actions are sampled from this distribution during training.

Policy gradient update (from lecture slides):
  For each episode, collect trajectory τ = s0,a0,r0,...,sT,aT,rT
  For each step t:
    G_t = sum_{k=t+1}^{T} gamma^{k-1} * r_k   (discounted return)
  policyLoss = -1.0 * gammas * returns * logProbs
  policyLoss = mean(policyLoss)
  θ = θ + alpha * grad_J

The NEGATIVE sign is because PyTorch minimises loss but we want to
MAXIMISE expected return — so we minimise the negative.

Key difference from value-based methods:
  No replay buffer, no target network, no TD bootstrapping.
  Learns from complete episodes (Monte Carlo).
  High variance but unbiased gradient estimates.

Reward structure (new obelix.py):
  Sensor bit bonuses are ONE-TIME per episode (not repeated every step).
  Max sensor bonus = 29 total per episode.
  -1 every step, -200 per stuck step, +100 touch box, +2000 push to boundary.
  Agent MUST push box to boundary to get positive return.

Reference:
  Williams (1992) Simple Statistical Gradient-Following Algorithms.
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


class PolicyNetwork(nn.Module):
    def __init__(self, inDim=18, outDim=5, hiddenDims=64, activations=nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inDim, hiddenDims),
            activations(),
            nn.Linear(hiddenDims, hiddenDims),
            activations(),
            nn.Linear(hiddenDims, outDim),
        )

    def forward(self, s):
        return self.net(s)

    def selectAction(self, s):
        """Stochastic action selection — used during training."""
        logits = self.forward(s)
        probs  = F.softmax(logits, dim=-1)
        dist   = torch.distributions.Categorical(probs)
        a      = dist.sample()
        logp_a = dist.log_prob(a)
        return a.item(), logp_a

    def selectGreedyAction(self, s):
        """Greedy action selection — used during evaluation."""
        with torch.no_grad():
            logits = self.forward(s)
        return int(torch.argmax(logits, dim=-1).item())


def getStepWiseReturnsAndDiscounts(gamma, rewards):
    T       = len(rewards)
    returns = torch.zeros(T)
    gammas  = torch.zeros(T)
    G       = 0.0

    for t in reversed(range(T)):
        G          = rewards[t] + gamma * G
        returns[t] = G
        gammas[t]  = gamma ** t

    # Normalise returns to reduce variance — important with sparse rewards
    if returns.std() > 1e-8:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return returns, gammas


def trainPolicyNetwork(rewards, logProbs, gamma, optimizerFn):
    """
    Implements trainPolicyNetwork(rewards, logProbs) from slide 6:

      returns, gammas = getStepWiseReturnsAndDiscounts(gamma, rewards)
      policyLoss = -1.0 * gammas * returns * logProbs
      policyLoss = mean(policyLoss)
      optimizerFn.zero_grad()
      policyLoss.backward()
      optimizerFn.step()
    """
    returns, gammas = getStepWiseReturnsAndDiscounts(gamma, rewards)
    logProbs_t      = torch.stack(logProbs)

    policyLoss = -1.0 * gammas * returns * logProbs_t
    policyLoss = policyLoss.mean()

    optimizerFn.zero_grad()
    policyLoss.backward()
    nn.utils.clip_grad_norm_(optimizerFn.param_groups[0]['params'], 5.0)
    optimizerFn.step()

    return policyLoss.item()


def evaluateAgent(env, pNetwork, MAX_EVAL_EPISODES, max_steps):
    """
    Implements evaluateAgent(greedy=True) from slide 7.
    Uses selectGreedyAction (deterministic) for evaluation.
    """
    rewards = []
    for _ in range(MAX_EVAL_EPISODES):
        rs   = 0.0
        s    = torch.tensor(np.array(env.reset(), dtype=np.float32))
        done = False
        for _ in range(max_steps):
            a           = pNetwork.selectGreedyAction(s)
            s2, r, done = env.step(ACTIONS[a], render=False)
            s           = torch.tensor(np.array(s2, dtype=np.float32))
            rs         += float(r)
            if done:
                break
        rewards.append(rs)
    return float(np.mean(rewards)), float(np.std(rewards))


def trainAgent(env, pNetwork, optimizerFn, gamma,
               MAX_TRAIN_EPISODES, MAX_EVAL_EPISODES, max_steps, out_path):
    """
    Implements trainAgent() from slide 5:

      for e in range(MAX_TRAIN_EPISODES):
        s, done = env.reset()
        rewards = [], logProbs = []
        while not done:
          a, logp_a = pNetwork.selectAction(s)
          s, r, done = env.step(a)
          rewards.append(r), logProbs.append(logp_a)
        trainPolicyNetwork(rewards, logProbs)
        performBookKeeping(train=True)
        evaluateAgent(pNetwork, MAX_EVAL_EPISODES)
        performBookKeeping(train=False)
    """
    for e in range(MAX_TRAIN_EPISODES):
        s    = torch.tensor(np.array(env.reset(seed=e), dtype=np.float32))
        done = False

        rewards  = []
        logProbs = []
        ep_ret   = 0.0

        for _ in range(max_steps):
            a, logp_a   = pNetwork.selectAction(s)
            s2, r, done = env.step(ACTIONS[a], render=False)
            s           = torch.tensor(np.array(s2, dtype=np.float32))
            rewards.append(float(r))
            logProbs.append(logp_a)
            ep_ret += float(r)
            if done:
                break

        loss = trainPolicyNetwork(rewards, logProbs, gamma, optimizerFn)

        if (e + 1) % 50 == 0:
            eval_mean, eval_std = evaluateAgent(env, pNetwork, MAX_EVAL_EPISODES, max_steps)
            print(f"Episode {e+1}/{MAX_TRAIN_EPISODES} return={ep_ret:.1f} loss={loss:.4f} eval_mean={eval_mean:.1f} eval_std={eval_std:.1f}")
            torch.save(pNetwork.state_dict(), out_path)
            gc.collect()

    return pNetwork


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
    ap.add_argument("--episodes",        type=int,   default=3000)
    ap.add_argument("--max_steps",       type=int,   default=1000)
    ap.add_argument("--difficulty",      type=int,   default=0)
    ap.add_argument("--wall_obstacles",  action="store_true")
    ap.add_argument("--box_speed",       type=int,   default=2)
    ap.add_argument("--scaling_factor",  type=int,   default=1)
    ap.add_argument("--arena_size",      type=int,   default=500)

    ap.add_argument("--gamma",           type=float, default=0.99)
    ap.add_argument("--lr",              type=float, default=3e-4)
    ap.add_argument("--hidden",          type=int,   default=64)
    ap.add_argument("--eval_episodes",   type=int,   default=1)
    ap.add_argument("--seed",            type=int,   default=1124)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    pNetwork    = PolicyNetwork(inDim=18, outDim=N_ACTIONS, hiddenDims=args.hidden)
    optimizerFn = optim.Adam(pNetwork.parameters(), lr=args.lr)

    env = OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
        seed=args.seed,
    )

    pNetwork = trainAgent(
        env                = env,
        pNetwork           = pNetwork,
        optimizerFn        = optimizerFn,
        gamma              = args.gamma,
        MAX_TRAIN_EPISODES = args.episodes,
        MAX_EVAL_EPISODES  = args.eval_episodes,
        max_steps          = args.max_steps,
        out_path           = args.out,
    )

    torch.save(pNetwork.state_dict(), args.out)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
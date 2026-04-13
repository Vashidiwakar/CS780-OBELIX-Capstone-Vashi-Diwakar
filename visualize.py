"""Visualizer for any OBELIX agent.

Works with DQN, Dueling DDQN, REINFORCE, A2C, SARSA-lambda, Q-learning.
Renders the environment visually and prints per-step info.

Example:
  python visualize.py --agent_file ./agent_a2c.py --weights ./weights_a2c.pth --difficulty 0
  python visualize.py --agent_file ./agent.py --weights ./weights.pth --difficulty 0 --wall_obstacles
  python visualize.py --agent_file ./agent_reinforce.py --weights ./weights_reinforce.pth --episodes 5
"""

import argparse
import importlib.util
import shutil
import os
import numpy as np


def load_agent(agent_file: str, weights_file: str):
    """Load agent module and copy weights next to it so _load_once() works."""
    agent_dir  = os.path.dirname(os.path.abspath(agent_file))
    agent_name = os.path.basename(agent_file)
    weights_dst = os.path.join(agent_dir, "weights.pth")

    # Copy weights to weights.pth next to agent file if needed
    if os.path.abspath(weights_file) != os.path.abspath(weights_dst):
        shutil.copy(weights_file, weights_dst)
        print(f"Copied {weights_file} -> {weights_dst}")

    spec   = importlib.util.spec_from_file_location("agent_mod", agent_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "policy"):
        raise AttributeError(f"{agent_name} must define policy(obs, rng) -> str")

    return module.policy


def import_obelix(obelix_py: str):
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent_file",     type=str,   required=True,          help="Path to agent .py file")
    ap.add_argument("--weights",        type=str,   required=True,          help="Path to weights .pth file")
    ap.add_argument("--obelix_py",      type=str,   default="./obelix.py",  help="Path to obelix.py")
    ap.add_argument("--episodes",       type=int,   default=4,              help="Number of episodes to visualize")
    ap.add_argument("--max_steps",      type=int,   default=1000,           help="Max steps per episode")
    ap.add_argument("--difficulty",     type=int,   default=0,              help="0=static, 2=blinking, 3=moving")
    ap.add_argument("--wall_obstacles", action="store_true",                help="Enable wall obstacle")
    ap.add_argument("--scaling_factor", type=int,   default=3)
    ap.add_argument("--arena_size",     type=int,   default=500)
    ap.add_argument("--box_speed",      type=int,   default=2)
    ap.add_argument("--seed",           type=int,   default=42)
    ap.add_argument("--delay",          type=int,   default=10,             help="Delay between frames in ms (higher = slower)")
    args = ap.parse_args()

    policy = load_agent(args.agent_file, args.weights)
    OBELIX = import_obelix(args.obelix_py)
    rng    = np.random.default_rng(args.seed)

    env = OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
        seed=args.seed,
    )

    print(f"\nVisualizing {args.agent_file} for {args.episodes} episodes")
    print(f"difficulty={args.difficulty}  wall={args.wall_obstacles}  max_steps={args.max_steps}")
    print("Press Q in the render window to quit early.\n")

    import cv2

    total_rewards = []

    for ep in range(args.episodes):
        obs        = np.array(env.reset(seed=args.seed + ep), dtype=np.float32)
        ep_ret     = 0.0
        step_count = 0
        done       = False

        print(f"--- Episode {ep+1}/{args.episodes} ---")

        for step in range(args.max_steps):
            action = policy(obs, rng)
            obs, reward, done = env.step(action, render=True)
            obs        = np.array(obs, dtype=np.float32)
            ep_ret    += float(reward)
            step_count += 1

            # Print sensor state every 50 steps
            if (step + 1) % 50 == 0:
                stuck = int(obs[17])
                ir    = int(obs[16])
                fwd   = int(any(obs[4:12]))
                print(f"  step={step+1:4d}  action={action}  reward={reward:8.1f}  ep_ret={ep_ret:10.1f}  stuck={stuck}  IR={ir}  fwd_sensor={fwd}")

            key = cv2.waitKey(args.delay) & 0xFF
            if key == ord('q'):
                print("Quit requested.")
                cv2.destroyAllWindows()
                return

            if done:
                print(f"  DONE at step {step+1} — episode return = {ep_ret:.1f}")
                break

        if not done:
            print(f"  Max steps reached — episode return = {ep_ret:.1f}")

        total_rewards.append(ep_ret)

    cv2.destroyAllWindows()

    print(f"\n{'='*50}")
    print(f"Results over {args.episodes} episodes:")
    print(f"  Mean return : {np.mean(total_rewards):.1f}")
    print(f"  Std  return : {np.std(total_rewards):.1f}")
    print(f"  Best episode: {max(total_rewards):.1f}")
    print(f"  Worst episode: {min(total_rewards):.1f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
import argparse
import numpy as np
import torch

from obelix import OBELIX

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


def phi(obs, PAIRS):
    raw = obs.astype(np.float32)
    pairs = np.array([raw[i] * raw[j] for i, j in PAIRS], dtype=np.float32)
    bias = np.array([1.0], dtype=np.float32)
    return np.concatenate([raw, pairs, bias])


def load_model(path):
    data = torch.load(path, weights_only=False)
    return data["w"], data["pairs"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)

    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--difficulty", type=int, default=0)
    parser.add_argument("--box_speed", type=int, default=2)

    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    # Load trained weights
    w, PAIRS = load_model(args.weights)

    env = OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
    )

    for ep in range(args.episodes):
        obs = np.array(env.reset(), dtype=np.float32)
        total_reward = 0

        print(f"\n=== Episode {ep+1} ===")

        for step in range(args.max_steps):
            f = phi(obs, PAIRS)

            # Q-values
            q = w @ f

            # Greedy action
            action = ACTIONS[int(np.argmax(q))]

            obs, reward, done = env.step(action, render=True)
            obs = np.array(obs, dtype=np.float32)

            total_reward += reward

            if done:
                print("Done! Reward:", total_reward)
                break

        print("Episode reward:", total_reward)

    print("Finished.")


if __name__ == "__main__":
    main()
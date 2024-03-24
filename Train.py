from itertools import count

import torch
import gymnasium as gym
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np

from Agent import Agent
from config import cartpole_hyperparameters


def train(print_every):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(cartpole_hyperparameters["env_id"])
    agent = Agent(cartpole_hyperparameters, device)

    scores = []
    for episode in tqdm(range(cartpole_hyperparameters["n_training_episodes"])):
        saved_log_probs = []
        rewards = []
        observation, info = env.reset()

        for t in count():
            action, log_prob = agent.select_action(observation)
            saved_log_probs.append(log_prob)
            observation, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)

            if terminated or truncated:
                break

        agent.optimize_model(saved_log_probs, rewards)
        scores.append(sum(rewards))

        if episode % print_every == 0:
            print(f'Episode {episode}\t Average Score: {np.mean(scores)}')

    return scores


if __name__ == '__main__':
    scores = train(100)
    plt.plot(scores)
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.show()

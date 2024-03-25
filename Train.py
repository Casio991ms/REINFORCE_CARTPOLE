from itertools import count

import torch
import gymnasium as gym
from tqdm import tqdm
import wandb

from Agent import Agent
from config import cartpole_hyperparameters


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(cartpole_hyperparameters["env_id"])
    agent = Agent(cartpole_hyperparameters, device)

    wandb.watch(agent.policy_net, log="all")
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

        loss = agent.optimize_model(saved_log_probs, rewards)
        score = sum(rewards)
        wandb.log({
            'loss': loss,
            'score': score
        })

    agent.save_policy_net("models", "REINFORCE_CartPole.pt")


if __name__ == '__main__':
    run = wandb.init(
        project="REINFORCE_CartPole",
        config=cartpole_hyperparameters
    )

    train()
    run.log_model(
        path=cartpole_hyperparameters["model_path"],
        name="REINFORCE_CartPole"
)
    run.finish()

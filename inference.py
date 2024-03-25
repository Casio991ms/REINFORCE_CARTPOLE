import torch
import gymnasium as gym

from config import cartpole_hyperparameters
from Agent import Agent


def infer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(cartpole_hyperparameters["env_id"], render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder="videos")
    agent = Agent(cartpole_hyperparameters, device)

    agent.load_policy_net("models", "REINFORCE_CartPole.pt")

    observation, info = env.reset()
    done = False
    score = 0
    while not done:
        env.render()
        action, _ = agent.select_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score += reward

    env.close()
    print("Score: ", score)


if __name__ == '__main__':
    infer()

from collections import deque

import torch
from torch import optim
from torch.distributions import Categorical

from REINFORCE import REINFORCE


class Agent:
    def __init__(self, config, device):
        self.BATCH_SIZE = config["batch_size"]
        self.GAMMA = config["discount_rate"]

        self.env = config["env_id"]

        self.device = device
        self.policy_net = REINFORCE(config["state_space"], config["hidden_size"], config["action_space"]).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config["learning_rate"])

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        probs = self.policy_net(state)
        distribution = Categorical(probs)
        action = distribution.sample()
        return action.item(), distribution.log_prob(action)

    def optimize_model(self, saved_log_probs, rewards):
        returns = deque()

        for reward_reverse in rewards[::-1]:
            discounted_return = (returns[0] if len(returns) > 0 else 0)
            returns.appendleft(reward_reverse + self.GAMMA * discounted_return)

        policy_loss = []
        for log_prob, discounted_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * discounted_return)
        loss = torch.cat(policy_loss).sum()

        # print(f'Saved Log Probability: {[slp.item() for slp in saved_log_probs]}')
        # print(f'Returns: {returns}')
        # print(f'Policy Loss: {[pl.item() for pl in policy_loss]}')
        # print(f'Loss: {loss}')
        # print('----------------------------------------------------------')

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

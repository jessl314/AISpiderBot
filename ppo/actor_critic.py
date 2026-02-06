# reference ddpg actor.py 
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # got from DDPG actor, output modified
        self.actor_backbone = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.actor_mean = nn.Linear(256, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # copied from DDPG critic, remove acton input
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),  # No action_dim here
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

def get_action(self, state, deterministic=False):
    features = self.actor_backbone(state)
    mean = self.actor_mean(features)
    std = torch.exp(self.actor_log_std)

    value = self.critic(state)

    if deterministic:
        return mean, None, value
    
    dist = Normal(mean, std)
    action = dist.sample()
    log_prob = dist.log_prob(action).sum(dim=-1)

    return action, log_prob, value.squeeze()
    # squeeze removes single-dimensional elements from NumPy array

def evaluate_actions(self,state,action):
    pass
from ppo import Actor, Critic
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions import Normal

import numpy as np
import torch
from ppo import Agent

# create an agent for testing, random inputs
agent = Agent(n_inputs=10, n_actions=3, N=32, batch_size=8)

print("Testing choose_action")

obs = np.random.randn(10).astype(np.float32)
action, prob, value = agent.choose_action(obs)

print("Action shape:", action.shape)
print("Log prob:", prob)
print("Value:", value)

print("\nFilling memory with random values")

# generate_batches returns empty arrays when you fill the memory
# up completely because the ptr gets reset back to 0
for _ in range(31):
    agent.remember(
        state=np.random.randn(10).astype(np.float32),
        action=np.random.randn(3).astype(np.float32),
        probs=np.random.randn(),
        vals=np.random.randn(),
        reward=np.random.randn(),
        done=np.random.choice([0,1])
    )

print("Generating batches")
data = agent.memory.generate_batches()
print("States shape:", data[0].shape)
print("Batches:", len(data[-1]))
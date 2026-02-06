# used to train PPO
# train on CartPole
import gym
import numpy as np
import torch
from ppo_agent import PPOAgent
from roll_buffer import RolloutBuffer


def train_ppo(
    env_name='LunarLanderContinuous-v2',
    total_timesteps=1_000_000,
    n_steps=2048,  # Rollout length
    n_epochs=10,
    batch_size=64,
    save_freq=50000
):
    pass

# Setup

# Agent

# Buffer

# Training loop

if __name__ == "__main__":
    train_ppo()
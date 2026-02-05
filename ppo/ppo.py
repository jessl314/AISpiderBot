import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class PPOMemory:
    """
    stores data collected from algorithm until there is enough to make an 
    update/take an action.
    
    """
    def __init__(self, batch_size, state_dim, action_dim):
        self.batch_size = batch_size
        # pre-allocating everything as float32 numpy arrays
        self.states = np.zeros((batch_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((batch_size, action_dim), dtype=np.float32)
        self.probs = np.zeros((batch_size), dtype=np.float32)
        self.vals = np.zeros((batch_size), dtype=np.float32)
        self.rewards = np.zeros((batch_size), dtype=np.float32)
        self.dones = np.zeros((batch_size), dtype=np.float32)
        # using a pointer to tell computer which row in the numpy arrays 
        # should recieve the next piece of data
        # replaces append from lists
        self.ptr = 0

    def generate_batches(self):
        """
        breaks the full data (n_states) into smaller chunks in order to make 
        it easier for the algorithm to determine next action.
        
        """
        n_states = self.states.shape[0]  # total steps/states
        indices = np.arrange(n_states)  # creates numpy array with elements 0...n_states
        np.random.shuffle(indices) # randomize the memories, to avoid memorization

        batches = [indices[i:i + self.batch_size] for i in range(0, n_states, self.batch_size)]

        # returning batched data and multi-dimensional numpy arrays containing our memory data
        return (
            torch.tensor(self.states, dtype=torch.float32),
            torch.tensor(self.actions, dtype=torch.float32),
            torch.tensor(self.probs, dtype=torch.float32),
            torch.tensor(self.vals, dtype=torch.float32),
            torch.tensor(self.rewards, dtype=torch.float32),
            torch.tensor(self.dones, dtype=torch.float32),
            batches
        )

    def store_memory(self, state, action, probs, vals, reward, done):
        """
        stores memory at the location of current numpy array row

        """
        index = self.ptr
        self.states[index] = state
        self.actions[index] = action
        self.probs[index] = probs
        self.vals[index] = vals
        self.rewards[index] = reward
        self.dones[index] = done

        self.ptr += 1

    def clear_memory(self):
        """
        resets pointer, with old data remaining in arrays.
        any new steps after this will just over-write the old data. 
        
        """
        self.ptr = 0
    
class ActorNetwork(nn.Module):
    pass


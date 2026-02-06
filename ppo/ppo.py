import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions import Normal

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
    

class Actor(nn.Module):
    def __init__(self, n_inputs=372, n_actions=12, alpha=0.003, chkpt_dir='tmp/ppo'):
        """
        - actor network is implemented as a feed-forward
        - (sequential) neural network
        - 1st layer: maps input to 256 "features" to find patterns
        - 2nd layer: combines the features to understand more
        - 3rd layer: output layer, maps the 256 features to mean target positions for each servo
        - Tanh: keeps continous values between -1 and 1
        - ReLU: allows for non-linear boundaries by giving negative signals a zero value
        """
        super(Actor, self).__init__()
        # stores the neural network weights
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_ppo.pth')
        self.log_std = nn.Parameter(torch.zeros(n_actions))
        self.actor = nn.Sequential(
            nn.Linear(n_inputs, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
            nn.Tanh()
        )
        # nn.Softmax(dim=-1) - for discrete actions?

        # defines learning algorithm to update weights
        # using Adam which auto-adjusts learning rate 
        # for each individual weight
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        # detects GPU and sets device to it if available
        # else falls back to CPU
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """
        given actor's best guess of action based on the state and the exploration factor, create a probability Normal distribution where the center is mu
        
        """
        mu = self.actor(state)
        sigma = torch.exp(self.log_std)
        dist = Normal(mu, sigma)
        return dist
    
    def save_checkpoint(self):
        """
        saves the information in the state dictionary provided via inheriting from nn.Module into a checkpoint file
        """
        torch.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        """
        loads state dictionary from the checkpoint file
        
        """
        self.load_state_dict(torch.load(self.checkpoint_file))

        








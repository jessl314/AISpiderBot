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
    def __init__(self, mem_size, state_dim, action_dim, batch_size):
        self.mem_size = mem_size
        self.batch_size = batch_size
        
        # pre-allocating everything as float32 numpy arrays
        self.states = np.zeros((mem_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((mem_size, action_dim), dtype=np.float32)
        self.probs = np.zeros((mem_size), dtype=np.float32)
        self.vals = np.zeros((mem_size), dtype=np.float32)
        self.rewards = np.zeros((mem_size), dtype=np.float32)
        self.dones = np.zeros((mem_size), dtype=np.float32)

        # using a pointer to tell computer which row in the numpy arrays 
        # should recieve the next piece of data
        # replaces append from lists
        self.ptr = 0

    def generate_batches(self):
        """
        breaks the full data into smaller chunks in order to make 
        it easier for the algorithm to determine next action.
        
        """
        n_states = self.ptr  # only use data up to where pointer stopped to prevent training on empty zeroes
        indices = np.arange(n_states)  # creates numpy array with elements 0...n_states
        np.random.shuffle(indices) # randomize the memories, to avoid memorization

        batches = [indices[i:i + self.batch_size] for i in range(0, n_states, self.batch_size)]

        # returning batched data and numpy arrays containing our memory data
        return self.states[:self.ptr], \
            self.actions[:self.ptr], \
            self.probs[:self.ptr], \
            self.vals[:self.ptr], \
            self.rewards[:self.ptr], \
            self.dones[:self.ptr], \
            batches

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

        self.ptr = (self.ptr + 1) % self.mem_size

    def clear_memory(self):
        """
        resets pointer, with old data remaining in arrays.
        any new steps after this will just over-write the old data. 
        
        """
        self.ptr = 0
    

class Actor(nn.Module):
    """
    neural network that takes action based on the advantages each potential action could give.
    
    """
    def __init__(self, n_inputs=372, n_actions=12, alpha=0.0003, chkpt_dir='tmp/ppo'):
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
        # tanh used here because servos have physical limits
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
        self.loss_fn = nn.MSELoss()
        # detects GPU and sets device to it if available
        # else falls back to CPU
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def clear_optimizer():
        self.optimizer.zero_grad()

    def forward(self, state):
        """
        given actor's best guess of action based on the state and the exploration factor, create a probability Normal distribution where the center is mu
        defines data flow of neural network/how data is transformed by the layers defined in __init__
        
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


class Critic(nn.Module):
    """
    neural network that takes the input and predicts a single number -> state value. this is an estimate of the total amount of reward the robot is expected to collect from its current position until end of simulation.

    """
    def __init__(self, n_inputs=372, alpha=0.0003, chkpt_dir='tmp/ppo'):
        super(Critic, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_ppo.pth')
        # no Tanh since we want linear output
        self.critic = nn.Sequential(
            nn.Linear(n_inputs, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.loss_fn = nn.MSELoss()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def clear_optimizer():
        self.optimizer.zero_grad()

    def forward(self, state):
        return self.critic(state)

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

class Agent:
    def __init__(self, n_inputs=372, n_actions=12, gamma=0.99, alpha=0.0003, gae_lambda=0.95, policy_clip=0.2, batch_size=64, N=2048, n_epochs=10):
        self.horizon = N
        self.batch_size = batch_size
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = Actor(n_inputs, n_actions, alpha)
        self.critic = Critic(n_inputs, alpha)
        
        self.memory = PPOMemory(mem_size=self.horizon, state_dim=n_inputs, action_dim=n_actions, batch_size=self.batch_size)
    
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def clear_optimizer():
        self.actor.clear_optimizer()
        self.critic.clear_optimizer()
    
    def save_models(self):
        print("...saving models...")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
    
    def load_models(self):
        print("...loading models...")
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        # convert our observation to tensor and move to GPU
        state = torch.tensor([observation], dtype=torch.float).to(self.actor.device)
        
        # get actor probability distribution and critic state value
        dist = self.actor(state)
        value = self.critic(state)
        # sample the action (12 servo angles)
        action = dist.sample()

        # sum log_probs across 12 actions to treat 12 servo moves as one joint step
        probs = torch.sum(dist.log_prob(action), dim=1)
        # extract scalar values memory 
        probs_item = probs.item()
        value_item = torch.squeeze(value).item()
        
        # prepare action for robot by squeezing to numpy array, removes batch dimension, moves tp cpu and converts.
        action_numpy = torch.squeeze(action).cpu().detach().numpy()

        return action_numpy, probs_item, value_item
    
    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_probs_arr, vals_arr, \
                reward_arr, done_arr, batches = \
                    self.memory.generate_batches()
            
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            last_gae_lam = 0

            for t in reversed(range(len(reward_arr) - 1)):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    # Temporal Difference Error: difference between what robot expects to happen vs what actually happened
                    # TD = (reward + gamma * value_next) - value_curr
                    next_val = vals_arr[t + 1]
                    delta = (reward_arr[t] + self.gamma * next_val * (1 - int(done_arr[t]))) - vals_arr[t]
                    
                    # GAE formula: smooths the advantages across time
                    # generalized advantage estimation
                    advantage[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * (1 - int(done_arr[t])) * last_gae_lam

            advantage = torch.tensor(advantage).to(self.actor.device)
            values = torch.tensor(vals_arr).to(self.actor.device)
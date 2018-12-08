import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from ounoise import OUNoise
from model import Actor, Critic
from replay_buffer import ReplayBuffer


class Agent:
    """Interacts and learns from the environment"""

    def __init__(self, device, state_size, action_size, random_seed, fc1=128, fc2=128, lr_actor=1e-04, lr_critic=1e-04,
                 weight_decay=0, buffer_size=100000, batch_size=64, gamma=0.99, tau=1e-3):
        """
        Parameters
        ----------
            brain_name (String):
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            fc1 (int): 1st fully connected layer size for model (actor & critic)
            fc2 (int): 2nd fully connected layer size for model (actor & critic)
            device: CPU/GPU

            lr_actor (float): learning rate for Actor
            lr_critic (flaot): learning rate for Critic
            weight_decay (float): weight decay used in model optimizer
            buffer_size (int): replay buffer size
            batch_size (int): batch size to sample from buffer
            gamma (float): parameter used to calculate Q target
            tau (float): soft update interpolation parameter
        """
        self.device = device

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        # Actor network (with target)
        self.actor_local = Actor(self.state_size, self.action_size, random_seed, fc1_units=fc1, fc2_units=fc2).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, random_seed, fc1_units=fc1, fc2_units=fc2).to(device)
        self.actor_optimizer = optim.Adam(params=self.actor_local.parameters(), lr=lr_actor)

        # Critic actor
        self.critic_local = Critic(self.state_size, self.action_size, random_seed, fc1_units=fc1, fc2_units=fc2).to(device)
        self.critic_target = Critic(self.state_size, self.action_size, random_seed, fc1_units=fc1, fc2_units=fc2).to(device)
        self.critic_optimizer = optim.Adam(params=self.critic_local.parameters(), lr=lr_critic,
                                           weight_decay=weight_decay)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, self.device, random_seed)

        self.make_copy(self.critic_local, self.critic_target)
        self.make_copy(self.actor_local, self.actor_target)

        print("Initilized agent with state size = {} and action size = {}".format(
            self.state_size,
            self.action_size))

    def make_copy(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def step(self, state, action, reward, next_state, done):
        """
        Save experience in replay memory, and use random sample from buffer to learn
        """

        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > self.batch_size:
            batch = self.memory.sample()
            self.learn(batch)

    def act(self, state, add_noise=True):
        """
        Returns actions for given state as per current policy.
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.actor_local.eval()

        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()

        self.actor_local.train()

        if add_noise:
            action += self.noise.sample()

        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, batch):
        """
        Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Parameters
        ----------
            batch (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = batch

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_target_next = self.critic_target(next_states, actions_next)
        # compute Q targets for next states (y_i)
        Q_targets = rewards + (self.gamma * Q_target_next * (1.0 - dones))
        # Compute citic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimise loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimise loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)

    def soft_update(self, local_model, target_model):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Parameters
        ----------
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


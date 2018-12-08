import numpy as np
import random
from collections import namedtuple, deque

import torch


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, device, seed):
        """
        Initialize a ReplayBuffer object
        Parameters
        ----------
        buffer_size (int): maximum size of buffer
        batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory i.e. the buffer
        self.batch_size = batch_size
        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.seed = random.seed(seed)

        self.device = device

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """
        Randomly sample a batch from memory
        """
        batch = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in batch if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in batch if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in batch if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in batch if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in batch if e is not None]).astype(np.uint8)).float().to(
            self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """
        return the current size of memory
        """
        return len(self.memory)

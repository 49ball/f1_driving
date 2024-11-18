from collections import deque
import numpy as np
import torch
import os

EPS = 1e-8 

class RolloutBuffer:
    def __init__(
            self, device:torch.device, 
            total_steps:int,
            batch_size:int) -> None:
        self.device = device
        self.total_steps = total_steps
        self.batch_size = batch_size

    ################
    # Public Methods
    ################

    def addBatch(self, states, actions):
            self.states = states.copy()
            self.actions = actions.copy()
        
    @torch.no_grad()
    def getBatches(self):
        indices = np.random.permutation(self.total_steps)[:self.batch_size]

        # convert to tensor
        states_tensor = torch.tensor(self.states[indices], device=self.device, dtype=torch.float32)
        actions_tensor = torch.tensor(self.actions[indices], device=self.device, dtype=torch.float32)

        return states_tensor, actions_tensor
    
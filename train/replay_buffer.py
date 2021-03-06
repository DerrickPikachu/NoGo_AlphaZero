import abc
from collections import deque, namedtuple
from sys import maxsize
from torch.utils.data import Dataset
import torch
import random


Transition = namedtuple('Transition', 
                        ('state', 'action', 'reward'))


class BufferInterface(Dataset):
    def __init__(self, buffer_size:int) -> None:
        pass
    
    @abc.abstractmethod
    def append(self, transition:Transition) -> None:
        pass
    
    @abc.abstractmethod
    def sample(self, num_samples) -> list:
        pass
    
    def __len__(self) -> int:
        pass
    
    def __getitem__(self, index) -> Transition:
        pass


class ReplayBuffer(BufferInterface):
    def __init__(self, buffer_size: int) -> None:
        super().__init__(buffer_size)
        self.size = buffer_size
        self.buffer = deque([], maxlen=buffer_size)
        
    def append(self, transition:Transition) -> None:
        self.buffer.append(transition)
        if len(self.buffer) > self.size:
            self.buffer.popleft()
            
    def sample(self, num_samples) -> list:
        return random.sample(self.buffer, num_samples)
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def __getitem__(self, index) -> Transition:
        return self.buffer[index]

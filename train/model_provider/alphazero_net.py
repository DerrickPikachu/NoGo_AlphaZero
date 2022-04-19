import torch
import sys

sys.path.append('../')

from network import AlphaZeroResnet

class Forwarder:
    def __init__(self, directory: str) -> None:
        self.directory_root = directory
        self.network = AlphaZeroResnet(1, 10, input_size=(9, 9))
    
    def evaluate(self, state: torch.Tensor):
        policy_logic, value = self.network.forward(state)
        # softmax
        policy = policy_logic
        total = torch.exp(policy_logic).sum(1)
        for i in range(len(total)):
            policy[i] = torch.exp(policy_logic[i]) / total
        return policy, value
    
    def load_weight(self, name: str):
        path = self.directory_root + '/' + name
        self.network.load_state_dict(torch.load(path))
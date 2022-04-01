import torch
import abc

from replay_buffer import ReplayBuffer, Transition
from network import AlphaZeroResnet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TrainerInterface(abc.ABC):
    def __init__(self) -> None:
        pass
    
    @abc.abstractmethod
    def train(self) -> None:
        pass
    
    @abc.abstractmethod
    def model_forward(self, batch_state:torch.Tensor):
        pass
    
    @abc.abstractmethod
    def _model_update(self, p_loss, v_loss):
        pass
    
    @abc.abstractmethod
    def compute_loss(self, action:torch.Tensor, value:torch.Tensor,
                     real_action, real_value):
        pass
    
    
class Trainer(TrainerInterface):
    def __init__(self) -> None:
        super().__init__()
        self.policy_loss_func = torch.nn.CrossEntropyLoss()
        self.value_loss_func = torch.nn.MSELoss()
        self.network = AlphaZeroResnet(1, 10, input_size=(9, 9))
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=1.e-4,
            nesterov=True
        )
        self.network.train()
        self.network.to(device)
        
    def train(self, state, action, reward) -> None:
        policy, value = self.model_forward(state)
        p_loss, v_loss = self.compute_loss(policy, value, action, reward)
        self._model_update(p_loss, v_loss)
        
    def model_forward(self, batch_state: torch.Tensor):
        # TODO: check the input shape should be 
        # (batch size, 1, borad_size, board_size), 
        # otherwise throw exception
        action_logic, value = self.network(batch_state)
        # softmax
        policy = action_logic
        total = torch.exp(action_logic).sum(1)
        for i in range(len(total)):
            policy[i] = torch.exp(policy[i]) / total[i]
        return policy, value
    
    def _model_update(self, p_loss: torch.Tensor, v_loss: torch.Tensor):
        # TODO: maybe clamp the gradient
        self.optimizer.zero_grad()
        total_loss = p_loss + v_loss
        total_loss.backward()
        self.optimizer.step()
        
    
    def compute_loss(self, action: torch.Tensor, value: torch.Tensor,
                     real_action, real_value):
        p_loss = self.policy_loss_func.forward(action, real_action)
        v_loss = self.value_loss_func.forward(value, real_value)
        return p_loss, v_loss
    
import torch
import sys
import argparse

# Specify the project root path
sys.path.append('/desktop')

from train.network import AlphaZeroResnet

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
        

class Mediator:
    def __init__(self, path, board_size) -> None:
        # TODO: here needs to use config
        self.forwarder = Forwarder(path)
        self.board_size = board_size
    
    def process_cmd(self, cmd: str):
        if cmd == 'forward':
            board_str = input()
            policy, value = self.model_forward(board_str)
            print(self.encode_result(policy, value))
        elif cmd == 'refresh':
            model_name = input()
            self.refresh_model(model_name)
        elif cmd == 'exit':
            return 0
    
    def model_forward(self, board_state):
        preprocess_state = self._preprocess(board_state)
        policy, value = self.forwarder.evaluate(preprocess_state)
        post_policy, post_value = self._postprocess(policy, value)
        return post_policy, post_value
    
    def _preprocess(self, board_state: str):
        parsed_state = board_state.split(',')
        tensor_state = torch.zeros(len(parsed_state))
        for i in range(len(parsed_state)):
            tensor_state[i] = float(parsed_state[i])
        return tensor_state.view(
            (1, 1, self.board_size, self.board_size))
    
    def _postprocess(self, raw_policy, raw_value):
        policy = raw_policy.view(81)
        value = raw_value.view(1)
        return policy.tolist(), value.item()
    
    def refresh_model(self, name):
        self.forwarder.load_weight(name)

    def encode_result(self, policy, value):
        encoded_str = ''
        for prob in policy:
            encoded_str += str(prob)
            encoded_str += ','
        encoded_str = encoded_str[:len(encoded_str) - 1]
        encoded_str += ';' + str(value)
        return encoded_str
        

def main():
    parser = argparse.ArgumentParser(__package__, description=__doc__)
    parser.add_argument('-p', '--path', required=True, help='The path to saved weight')
    parser.add_argument('-bs', '--board_size',
                        type=int,
                        default=9,
                        help='The board size of the game')
    args = parser.parse_args()
    mediator = Mediator(args.path, args.board_size)
    print(args.path)
    print(args.board_size)
    # while (True):
    #     cmd = input()
    #     is_exit = mediator.process_cmd(cmd)
    #     if is_exit == 0:
    #         break


if __name__ == "__main__":
    main()
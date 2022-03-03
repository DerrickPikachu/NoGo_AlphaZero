from network import *
import torch

def basic_block_test():
    block = BasicBlock(32, False)
    test_input = torch.ones((1, 32, 2, 2))
    result = block(test_input)
    print(result.size())
    print('basic block test pass')


def basic_block_extend_test():
    block = BasicBlock(32, True)
    test_input = torch.ones((1, 32, 2, 2))
    result = block(test_input)
    print(result.size())
    print('basic block extend test pass')


def alphazero_resnet_test():
    net = AlphaZeroResnet(3, 40)
    test_input = torch.ones((1, 3, 19, 19))
    result = net(test_input)
    print(result[0].size())
    print(result[1].size())
    print('alphazero resnet test pass')

if __name__ == "__main__":
    basic_block_test()
    basic_block_extend_test()
    alphazero_resnet_test()
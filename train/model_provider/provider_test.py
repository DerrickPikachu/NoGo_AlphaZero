import unittest
from unittest.mock import MagicMock
import sys

from alphazero_net import *

sys.path.append('../')

from network import AlphaZeroResnet


class ForwarderTest(unittest.TestCase):
    def setUp(self):
        self.forwarder = Forwarder('test_model')
    
    def tearDown(self):
        del self.forwarder
    
    def testEvaluate(self):
        test_state = torch.rand((1, 1, 9, 9))
        action, value = self.forwarder.evaluate(test_state)
        action = action.view((81))
        sum = 0
        for action_prob in action:
            sum += action_prob.item()
        self.assertAlmostEqual(sum, 1.0, delta=0.0001)
        self.assertTrue(-1.0 <= value.item() <= 1.0)
        
    def testEvaluateModelForward(self):
        fake_action_logic = torch.rand((1, 81))
        fake_value = torch.rand((1, 1))
        test_state = torch.rand((1, 1, 9, 9))
        network_mock = MagicMock(spec=AlphaZeroResnet)
        network_mock.forward.return_value = \
            (fake_action_logic, fake_value)
        self.forwarder.network = network_mock
        
        _, _ = self.forwarder.evaluate(test_state)
        
        network_mock.forward.assert_called_once_with(test_state)
        
    def testLoadWeight(self):
        fake_network = AlphaZeroResnet(1, 10, input_size=(9, 9))
        torch.save(fake_network.state_dict(), 'test_model/fake_weight.pth')
        
        self.forwarder.load_weight('fake_weight.pth')
        
        origin_param = [np for np in fake_network.parameters()]
        load_param = [np for np in self.forwarder.network.parameters()]
        for p0, p1 in zip(origin_param, load_param):
            self.assertTrue(torch.equal(p0, p1))
        


if __name__ == "__main__":
    unittest.main()
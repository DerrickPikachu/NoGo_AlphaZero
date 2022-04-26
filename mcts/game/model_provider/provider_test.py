import unittest
from unittest.mock import MagicMock
import sys

from alphazero_net import *

sys.path.append('/desktop')

from train.network import AlphaZeroResnet


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
        
        
class MediatorTest(unittest.TestCase):
    def setUp(self):
        self.mediator = Mediator('test_model', board_size=9)
    
    def tearDown(self):
        del self.mediator
    
    def testModelForward(self):  # TODO: None test
        fake_policy = torch.tensor([1 / 81] * 81).view((1, 81))
        fake_value = torch.tensor([0.5]).view((1, 1))
        forwarder_mock = MagicMock(spec=Forwarder)
        forwarder_mock.evaluate.return_value = (fake_policy, fake_value)
        self.mediator.forwarder = forwarder_mock
        
        test_input = ','.join(['0.0'] * 81)
        policy, value = self.mediator.model_forward(test_input)
        
        preprocess_result = torch.tensor([0.0] * 81).view((1, 1, 9, 9))
        forwarder_mock.evaluate.assert_called_once()
        evaluate_args = forwarder_mock.evaluate.call_args
        self.assertTrue(torch.equal(evaluate_args, preprocess_result))
        self.assertEqual(value, 0.5)
        for prob in policy:
            self.assertAlmostEqual(prob, 1 / 81, delta=0.0001)
        
    def testRefreshModel(self):
        forwarder_mock = MagicMock(spec=Forwarder)
        self.mediator.forwarder = forwarder_mock
        self.mediator.refresh_model('test_model')
        forwarder_mock.load_weight.assert_called_once_with('test_model')
        
    def testEncodeResult(self):
        policy = [1 / 81] * 81
        value = 0.5
        ans_str = ''
        for i in range(81):
            ans_str += str(1 / 81)
            if i == 80:
                ans_str += ';'
            else:
                ans_str += ','
        ans_str += str(value)
        output_str = self.mediator.encode_result(policy, value)
        self.assertEqual(output_str, ans_str)


if __name__ == "__main__":
    unittest.main()
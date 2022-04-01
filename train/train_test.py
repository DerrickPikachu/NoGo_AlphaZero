import unittest
from unittest import mock
import torch
import trajectory_pb2 as trajectory
from torch.utils.data import DataLoader
from torchtest import assert_vars_change

from replay_buffer import *
from trainer import *


class ReplayBufferTest(unittest.TestCase):
    def setUp(self):
        self.replay_buffer = ReplayBuffer(20)
        trajectory_file = open('test_trajectory', 'rb')
        raw = trajectory_file.read()
        self.trajectory_pb = trajectory.trajectory()
        self.trajectory_pb.ParseFromString(raw)
        trajectory_file.close()
    
    def tearDown(self):
        del self.replay_buffer
    
    def testAppend(self):
        dummy_state = torch.zeros(10)
        test_transition = Transition(dummy_state, 10, 1.0)
        self.replay_buffer.append(test_transition)
        self.assertEqual(len(self.replay_buffer), 1)
        
    def testAppendOverSize(self):
        self.replay_buffer = ReplayBuffer(2)
        for i in range(3):
            state_tensor = torch.tensor(self.trajectory_pb.transitions[i].state)
            action = self.trajectory_pb.transitions[i].action_id
            reward = self.trajectory_pb.transitions[i].reward
            transition = Transition(state_tensor, action, reward)
            self.replay_buffer.append(transition)
            
        self.assertEqual(len(self.replay_buffer), 2)
        for i in range(2):
            self.assertEqual(self.replay_buffer[i].action, 
                             self.trajectory_pb.transitions[i+1].action_id)
        
    def testAppendTrajectory(self):
        for tran in self.trajectory_pb.transitions:
            state_tensor = torch.tensor(tran.state)
            transition = Transition(state_tensor, tran.action_id, tran.reward)
            self.replay_buffer.append(transition)
            
        self.assertEqual(len(self.replay_buffer), 20)
        trajectory_len = len(self.trajectory_pb.transitions)
        for i in range(20):
            self.assertEqual(self.replay_buffer[i].action,
                             self.trajectory_pb.transitions[trajectory_len - 20 + i].action_id)
        
    def testSample(self):
        for tran in self.trajectory_pb.transitions:
            state_tensor = torch.tensor(tran.state)
            transition = Transition(state_tensor, tran.action_id, tran.reward)
            self.replay_buffer.append(transition)
        sampled_transitions = self.replay_buffer.sample(num_samples=10)
        self.assertEqual(len(sampled_transitions), 10)
        founded = 0
        for transitions in sampled_transitions:
            for i in range(len(self.replay_buffer)):
                if self.replay_buffer[i].action == transitions.action:
                    founded += 1
                    break
        self.assertEqual(founded, 10)
        
    def testLoadData(self):
        for tran in self.trajectory_pb.transitions:
            state_tensor = torch.tensor(tran.state)
            transition = Transition(state_tensor, tran.action_id, tran.reward)
            self.replay_buffer.append(transition)
        data_loader = DataLoader(
            dataset=self.replay_buffer, 
            batch_size=1,
            shuffle=True
        )
        founded = 0
        for transition in data_loader:
            for i in range(len(self.replay_buffer)):
                if self.replay_buffer[i].action == transition.action:
                    founded += 1
                    break
        self.assertEqual(founded, 20)
        
        
class TrainerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.trainer = Trainer()
        trajectory_file = open('test_trajectory', 'rb')
        raw = trajectory_file.read()
        trajectory_pb = trajectory.trajectory()
        trajectory_pb.ParseFromString(raw)
        
        self.replay_buffer = ReplayBuffer(buffer_size=64)
        for tran in trajectory_pb.transitions:
            state_tensor = torch.tensor(tran.state).view((9, 9))
            transition = Transition(state_tensor, tran.action_id, tran.reward)
            self.replay_buffer.append(transition)
        trajectory_file.close()
        
        data_loader = DataLoader(
            dataset=self.replay_buffer,
            batch_size=32,
            shuffle=True
        )
        data_iter = iter(data_loader)
        batch_transition = next(data_iter)
        self.batch_state = batch_transition.state.view(32, 1, 9, 9).to(device)
        self.batch_action = batch_transition.action.to(device)
        self.batch_reward = batch_transition.reward.view(32, 1).float().to(device)
    
    def tearDown(self) -> None:
        del self.trainer
        del self.replay_buffer
        
    def testModelForward(self):
        action, value = self.trainer.model_forward(self.batch_state)
        self.assertEqual(action.shape, (32, 81))
        for action_vector in action:
            sum = 0
            for action_prob in action_vector:
                sum += action_prob.item()
            self.assertAlmostEqual(sum, 1.0, delta=0.0001)
        for estimate_value in value:
            self.assertTrue(-1.0 <= estimate_value.item() <= 1.0)
    
    def testModelUpdate(self):
        parameters = [np for np in self.trainer.network.parameters()]
        temporary_params = [p.clone() for p in parameters]
        
        policy, value = self.trainer.model_forward(self.batch_state)
        p_loss, v_loss = self.trainer.compute_loss(
            policy, value, self.batch_action, self.batch_reward)
        self.trainer._model_update(p_loss, v_loss)
        
        for p0, p1 in zip(temporary_params, parameters):
            self.assertTrue(not torch.equal(p0, p1))
        
    
    def testComputeLoss(self):
        m_policy_loss_func = mock.Mock(
            wraps=self.trainer.policy_loss_func)
        m_value_loss_func = mock.Mock(
            wraps=self.trainer.value_loss_func)
        self.trainer.policy_loss_func = m_policy_loss_func
        self.trainer.value_loss_func = m_value_loss_func
        
        action, value = self.trainer.model_forward(self.batch_state)
        loss = self.trainer.compute_loss(
            action, value, self.batch_action, self.batch_reward)
        policy_loss, value_loss = loss
        
        self.assertTrue(policy_loss >= 0.0 and value_loss >= 0.0)
        m_policy_loss_func.forward.assert_called_once_with(
            action, self.batch_action)
        m_value_loss_func.forward.assert_called_once_with(
            value, self.batch_reward)
    
    def testTrain(self):
        parameters = [np for np in self.trainer.network.parameters()]
        temporary_params = [p.clone() for p in parameters]
        self.trainer.train(self.batch_state, self.batch_action, self.batch_reward)
        for p0, p1 in zip(temporary_params, parameters):
            self.assertTrue(not torch.equal(p0, p1))
        
    def testTrainerDevice(self):
        self.assertTrue(torch.cuda.is_available())
        self.assertTrue(next(self.trainer.network.parameters()).is_cuda)
        
    
        
if __name__ == "__main__":
    unittest.main()
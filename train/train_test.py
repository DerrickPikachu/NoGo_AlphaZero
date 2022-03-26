import unittest
import torch
import trajectory_pb2 as trajectory

from replay_buffer import *


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
        sampled_index = self.replay_buffer.sample(num_samples=10)
        self.assertEqual(len(sampled_index), 10)
        for index in sampled_index:
            self.assertTrue(index < len(self.replay_buffer) and index >= 0)

if __name__ == "__main__":
    unittest.main()
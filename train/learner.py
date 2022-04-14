from trajectory_socket import TrajectoryServer
from replay_buffer import ReplayBuffer, Transition
import trajectory_pb2 as trajectory

import torch
import yaml
from pathlib import Path
# import argparse


def self_play_loop(config: dict, actor_socket: TrajectoryServer):
    replay_buffer = ReplayBuffer(config['replayer_buffer']['size'])
    # file = open('test_trajectory', 'wb')
    while True:
        byte_int = actor_socket.recv(4)
        size = 0
        for i in range(4):
            size *= pow(2, 8)
            size += int(byte_int[i])
        
        raw = actor_socket.recv(size)
        print("receive raw data len: {}".format(len(raw)))
        parsed_trajectory = trajectory.trajectory()
        parsed_trajectory.ParseFromString(raw)
        for tran in parsed_trajectory.transitions:
            state_tensor = torch.tensor(tran.state)
            transition = Transition(state_tensor, tran.action_id, tran.reward)
            replay_buffer.append(transition)
        # file.write(parsed_trajectory.SerializeToString())
        print("num_transition: ", len(parsed_trajectory.transitions))
        print("last reward: ", parsed_trajectory.transitions[-1].reward)
        print("replay buffer size: ", len(replay_buffer))
        print("-----------------------------------------")


def main():
    config = yaml.safe_load(Path('learner_config.yaml').read_text())
    learner_server = TrajectoryServer('0.0.0.0', 7000)
    learner_server.start()
    self_play_loop(config, learner_server)


if __name__ == "__main__":
    main()
    
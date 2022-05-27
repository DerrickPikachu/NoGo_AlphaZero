from trajectory_socket import TrajectoryServer
from replay_buffer import ReplayBuffer, Transition
from trainer import Trainer, device
import trajectory_pb2 as trajectory

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from pathlib import Path
from tqdm import tqdm
# import argparse

def training(trainer: Trainer, replay_buffer: ReplayBuffer, 
             writer: SummaryWriter, iter: int, config: dict):
    batch_size = config['trainer']['batch_size']
    board_size = config['game']['board_size']
    if len(replay_buffer) < batch_size * config['trainer']['batch_to_train']:
        return
    
    data_loader = DataLoader(
        dataset=replay_buffer,
        batch_size=batch_size,
        shuffle=True
    )
    print('---------New Iteration---------')
    step = 0
    total_p_loss = 0.0
    total_v_loss = 0.0
    for batch_transition in tqdm(data_loader):
        batch_state = batch_transition.state.view(
            -1, 1, board_size, board_size).to(device)
        batch_action = batch_transition.action.to(device)
        batch_reward = batch_transition.reward.view(-1, 1).float().to(device)

        p_loss, v_loss = trainer.train(batch_state, batch_action, batch_reward)

        if iter % config['trainer']['checkpoint_freq'] == 0:
            trainer.save_model(
                config['trainer']['checkpoint_dir'] + str(iter) + '.pth')
            trainer.save_weight(config['trainer']['weight_dir'] + 'lastest.pt')
            trainer.save_weight(config['trainer']['weight_dir'] + str(iter) + '.pt')
        
        total_p_loss += p_loss.item()
        total_v_loss += v_loss.item()
        step += 1
    
    p_loss = total_p_loss / step
    v_loss = total_v_loss / step
    writer.add_scalar('loss/policy', p_loss, iter)
    writer.add_scalar('loss/value', v_loss, iter)
    writer.add_scalar('loss/total', p_loss + v_loss, iter)
    print('policy loss: {}'.format(p_loss))
    print('value loss: {}'.format(v_loss))
    print('total: ', p_loss + v_loss)


def self_play_loop(config: dict, actor_socket: TrajectoryServer):
    replay_buffer = ReplayBuffer(config['replayer_buffer']['size'])
    trainer = Trainer(config)
    writer = SummaryWriter(log_dir=config['trainer']['log_dir'])
    iter = 0

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
        last_reward = float(parsed_trajectory.transitions[-1].reward)
        print("num_transition: ", len(parsed_trajectory.transitions))
        print("last reward: ", last_reward)
        for tran in parsed_trajectory.transitions:
            state_tensor = torch.tensor(tran.state)
            transition = Transition(state_tensor, tran.action_id, last_reward)
            replay_buffer.append(transition)
        print("replay buffer size: ", len(replay_buffer))

        # training(trainer, replay_buffer, writer, iter, config)
        print("-----------------------------------------")
        iter += 1


def main():
    config = yaml.safe_load(Path('learner_config.yaml').read_text())
    learner_server = TrajectoryServer('0.0.0.0', 7000)
    learner_server.start()
    self_play_loop(config, learner_server)


if __name__ == "__main__":
    main()
    
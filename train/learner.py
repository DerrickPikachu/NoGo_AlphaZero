from re import purge
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
import pickle
# import argparse

def store_replay_buffer(replay_buffer: ReplayBuffer, config: dict):
    file = open(config['replay_buffer']['store_path'], 'wb')
    pickle.dump(replay_buffer, file)
    file.close()

def restore_replay_buffer(config: dict):
    file = open(config['replay_buffer']['store_path'], 'rb')
    return pickle.load(file)
    

def training(trainer: Trainer, replay_buffer: ReplayBuffer, 
             writer: SummaryWriter, iter: int, config: dict):
    batch_size = config['trainer']['batch_size']
    board_size = config['game']['board_size']
    
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
        step += 1
        batch_state = batch_transition.state.view(
            -1, 1, board_size, board_size).to(device)
        batch_action = batch_transition.action.to(device)
        batch_reward = batch_transition.reward.view(-1, 1).float().to(device)

        p_loss, v_loss = trainer.train(batch_state, batch_action, batch_reward)

        if iter % config['trainer']['checkpoint_freq'] == 0:
            trainer.save_model(
                config['trainer']['checkpoint_dir'] + str(iter) + '.pth')
            trainer.save_weight(config['trainer']['weight_dir'] + 'latest.pt')
            trainer.save_weight(config['trainer']['weight_dir'] + str(iter) + '.pt')
        
        total_p_loss += p_loss.item()
        total_v_loss += v_loss.item()
    
    p_loss = total_p_loss / step
    v_loss = total_v_loss / step
    writer.add_scalar('loss/policy', p_loss, iter)
    writer.add_scalar('loss/value', v_loss, iter)
    writer.add_scalar('loss/total', p_loss + v_loss, iter)
    print('policy loss: {}'.format(p_loss))
    print('value loss: {}'.format(v_loss))
    print('total: ', p_loss + v_loss)


def self_play_loop(config: dict, actor_socket: TrajectoryServer):
    replay_buffer = ReplayBuffer(config['replay_buffer']['size'])
    trainer = Trainer(config)
    iter = 0
    if config['restore']['active'] == True:
        print("------------Restore Trajectory------------")
        replay_buffer = restore_replay_buffer(config)
        trainer.load_model(config['trainer']['checkpoint_dir'] + 
            str(config['restore']['iter']) + '.pth')
        iter = config['restore']['iter']
        print('replay_buffer size: ', len(replay_buffer))
        print('start iteration: ', iter)
    else:
        print('no restore')
        trainer.save_model(config['trainer']['checkpoint_dir'] + '0.pth')
        trainer.save_weight(config['trainer']['weight_dir'] + 'latest.pt')
    
    writer = SummaryWriter(
        log_dir=config['trainer']['log_dir'],
        purge_step=iter
    )

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

        state_to_train = \
            config['game']['board_size'] * config['trainer']['batch_to_train']
        if len(replay_buffer) < state_to_train:
            # training(trainer, replay_buffer, writer, iter, config)
            iter += 1
        print("-----------------------------------------")
        store_replay_buffer(replay_buffer, config)

def train_loop(config: dict):
    replay_buffer = ReplayBuffer(config['replay_buffer']['size'])
    trainer = Trainer(config)
    iter = 0
    if config['restore']['active'] == True:
        print("------------Restore Trajectory------------")
        replay_buffer = restore_replay_buffer(config)
        print('replay_buffer size: ', len(replay_buffer))

    trainer.save_model(config['trainer']['checkpoint_dir'] + '0.pth')
    trainer.save_weight(config['trainer']['weight_dir'] + 'latest.pt')    
    writer = SummaryWriter(
        log_dir=config['trainer']['log_dir'],
        purge_step=iter
    )
    while True:
        training(trainer, replay_buffer, writer, iter, config)
        iter += 1
    

def main():
    config = yaml.safe_load(Path('learner_config.yaml').read_text())
    if (config['learn_from_dataset'] == False):
        learner_server = TrajectoryServer('0.0.0.0', 7000)
        learner_server.start()
        self_play_loop(config, learner_server)
    elif (config['learn_from_dataset'] == True):
        print('---------Training From Dataset---------')
        train_loop(config)


if __name__ == "__main__":
    main()
    
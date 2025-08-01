import torch
import torch.nn.functional as F
import numpy as np
import os
from model import Qnet


class DQN:
    def __init__(self, state_dim, action_dim, learning_rate, gamma, epsilon,
                 epsilon_end, epsilon_decay, step_count, target_update, device, checkpoint_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = Qnet(state_dim, action_dim).to(self.device)
        self.target_q_net = Qnet(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100,gamma = 0.995)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.step_count = step_count
        self.target_update = target_update
        self.count = 0
        self.checkpoint_dir = checkpoint_dir

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            # (state_dim,)
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            # (batch_size = 1, state_dim)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        # (state_dim,)
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        # (batch_size = 1, state_dim)
        # (n,)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        # (n,1) view(-1,1)表示第二维为1，第一维自动推断
        # (n,)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        # (n,1)
        # (state_dim,)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        # (batch_size = 1, state_dim)
        # (n,)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        # (n,1)
        #self.q_net(states)——>(batch_size, action_dim)
        #actions——>(batch_size, 1)，即每个样本所选动作的 Q 值
        q_values = self.q_net(states).gather(1, actions)
        max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
        max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1
        return dqn_loss.item()

    def save_model(self, q_net_path, target_q_net_path, return_list, loss_list, episode, case_index, epsilon,
                   best_return):
        torch.save(self.q_net.state_dict(), q_net_path)
        torch.save(self.target_q_net.state_dict(), target_q_net_path)
        torch.save({
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }, self.checkpoint_dir + "/optimizer_checkpoint.pth")
        np.savez(self.checkpoint_dir + f"/checkpoint.npz",
                 return_list=return_list,
                 loss_list=loss_list,
                 episode=episode,
                 case_index=case_index,
                 epsilon=epsilon,
                 best_return=best_return)

    def save_best_model(self, q_net_path):
        torch.save(self.q_net.state_dict(), q_net_path)

    def load_model(self, q_net_path, target_q_net, checkpoint_path):
        self.q_net.load_state_dict(torch.load(q_net_path))
        self.target_q_net.load_state_dict(torch.load(target_q_net))
        checkpoint = np.load(checkpoint_path, allow_pickle=True)

        optimizer_checkpoint_path = self.checkpoint_dir + "/optimizer_checkpoint.pth"
        if os.path.exists(optimizer_checkpoint_path):
            optimizer_checkpoint = torch.load(optimizer_checkpoint_path)
            self.optimizer.load_state_dict(optimizer_checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(optimizer_checkpoint['scheduler_state_dict'])
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = optimizer_checkpoint['learning_rate']
            print(f"恢复学习率: {optimizer_checkpoint['learning_rate']}")

        return (checkpoint['episode'], checkpoint['return_list'].tolist(),
                checkpoint['case_index'], checkpoint['epsilon'],
                checkpoint['loss_list'].tolist(), checkpoint['best_return'])

    def update_epsilon(self):
        if self.step_count > 0:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)
        self.step_count += 1
        print(f'epsilon：{self.epsilon}')
        return
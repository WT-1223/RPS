import torch
import os

# DQN Hyperparameters
lr = 1e-2
num_episodes = 500
gamma = 0.98
epsilon = 1.0
epsilon_end = 0.05
epsilon_decay = 0.99
target_update = 10
buffer_size = 2000
minimal_size = 500
batch_size = 64
step_count = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Paths
checkpoint_dir = "checkpoints_DQN_modify"
q_net_save_path = os.path.join(checkpoint_dir, "q_net_model_DQN.pth")
target_q_net_save_path = os.path.join(checkpoint_dir, "target_q_net_save_path.pth")
best_q_net_save_path = os.path.join(checkpoint_dir, "q_net_model_DQN_best.pth")
checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.npz")
replay_buffer_path = os.path.join(checkpoint_dir, "replay_buffer.pkl")
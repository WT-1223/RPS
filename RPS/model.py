import torch
import torch.nn.functional as F

# MLP
class Qnet(torch.nn.Module):
    """Q网络，用于近似Q值函数。"""
    def __init__(self, state_dim, action_dim):
        """初始化Q网络。
        :param state_dim: 状态空间的维度
        :param action_dim: 动作空间的维度
        """
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 128)
        self.fc4 = torch.nn.Linear(128, action_dim)
    def forward(self, x):
        """前向传播。
        :param x: 输入状态
        :return: 每个动作的Q值
        """
        x = self.fc1(x)
        shortcut = x
        x = F.relu(self.fc2(x))
        x = self.fc3(x) + shortcut
        x = F.layer_norm(x, normalized_shape=x.shape[1:])
        return self.fc4(x)
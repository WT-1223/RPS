import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def plot_learning_curve(episodes_list, return_list, title):
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(title)
    plt.show()

def plot_loss_curve(loss_value_list, loss_list, title):
    plt.plot(loss_value_list, loss_list)
    plt.xlabel('Update_Num')
    plt.ylabel('Loss')
    plt.title(title)
    plt.show()

def evaluate_agent(env, agent, num_episodes=1, case_index=None):
    agent.q_net.eval()
    total_return = 0
    with torch.no_grad():
        for _ in range(num_episodes):
            state = env.reset(case_index=case_index)
            done = False
            episode_return = 0
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                episode_return += reward
                state = next_state
            total_return += episode_return
            case_index = case_index + 1
    agent.q_net.train()
    avg_return = total_return / num_episodes
    return avg_return, case_index

def calculate_rl_loss(log_probs: List, normalized_rewards: List[float]) -> torch.Tensor:
    """
    完整实现 Soft Q-Learning (SQL) 的 on-policy 策略梯度损失 。

    损失函数基于 REINFORCE 目标: Loss = - E [A * log(pi)],
    其中规范化奖励 (Normalized Rewards) 充当基线稳定化后的优势函数 (Advantage) 。
    """
    # 将规范化奖励转换为 Tensor
    rewards_tensor = torch.tensor(normalized_rewards, dtype=torch.float32)

    # 将 log_probs 列表转换为 Tensor (需要堆叠)
    # 注意：log_probs 是在 GPU 上生成的，需要确保在 CPU 上进行操作或确保设备一致
    log_probs_tensor = torch.stack(log_probs)

    # 策略梯度核心计算: 目标是最大化 E
    # PyTorch 优化器默认执行梯度下降，因此我们最小化 - E
    # rewards_tensor (Advantage) 乘以 log_probs_tensor (Policy likelihood)
    policy_gradient_term = rewards_tensor * log_probs_tensor

    # 最终损失：对策略梯度项取负均值
    loss = -torch.mean(policy_gradient_term)

    # SQL 理论上包含一个熵正则化项，此处省略，专注于 Policy Gradient 核心部分
    # 熵正则化项: - self.entropy_coeff * torch.mean(exp(log_probs_tensor) * log_probs_tensor)

    return loss
import numpy as np
import torch
import matplotlib.pyplot as plt

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
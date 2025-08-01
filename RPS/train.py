import gym
from tqdm import tqdm
from gym.envs.registration import register
from DB import DB
from agent import Agent
from replay_buffer import ReplayBuffer, save_replay_buffer, load_replay_buffer
from dqn_agent import DQN
from config import *
from utils import *

# Initialize components
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

replay_buffer = ReplayBuffer(buffer_size)
dialogue_agent = Agent()
db = DB(host='localhost', user='root', password='020201', database='law_data_total')

# Initialize environment
register(
    id='lawyer_suspector_env_v0',
    entry_point='Lawyer-Suspector-Env.lawyer-suspector-env:LawyerSuspectorEnv',
)
env = gym.make('lawyer_suspector_env_v0', db_connection=db, agent=dialogue_agent, rl_model='DQN')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

DQN_agent = DQN(
    state_dim=state_dim,
    action_dim=action_dim,
    learning_rate=lr,
    gamma=gamma,
    epsilon=epsilon,
    epsilon_end=epsilon_end,
    epsilon_decay=epsilon_decay,
    step_count=step_count,
    target_update=target_update,
    device=device,
    checkpoint_dir=checkpoint_dir,
)

# Training variables
return_list = []
loss_list = []
best_return = float('-inf')

# Recover environment if checkpoint exists
if os.path.exists(checkpoint_path):
    print("恢复上次训练进度...")
    episode, return_list, case_index, DQN_agent.epsilon, loss_list, best_return = DQN_agent.load_model(
        q_net_save_path, target_q_net_save_path, checkpoint_path)
    replay_buffer = load_replay_buffer(replay_buffer_path, buffer_size)
    print(
        f"恢复的episode: {episode}, 上次返回值列表的最后值: {round(return_list[-1], 2)}, 上次案件条数: {case_index + 1}")
    print(f'replay_buffer的大小:{replay_buffer.size()}')
    print(f'epsilon的值为：{DQN_agent.epsilon}')
    print(f"恢复的 best_return: {best_return}")
    case_index = case_index + 1
else:
    print("没有保存的训练进度和模型，开始从头训练...")
    episode, return_list, loss_list, case_index,DQN_agent.epsilon, best_return = 0, [], [], 0, 1, float('-inf')

# Training loop
remaining_episodes = num_episodes
with tqdm(total=remaining_episodes, initial=episode + 1, desc=f"Training or Evaluating Progress") as pbar:
    for i_episode in range(episode + 1, num_episodes + 1):
        episode_return = 0
        state = env.reset(case_index=case_index)
        done = False

        while not done:
            action = DQN_agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_return += reward

            if replay_buffer.size() > minimal_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                transition_dict = {
                    'states': b_s,
                    'actions': b_a,
                    'next_states': b_ns,
                    'rewards': b_r,
                    'dones': b_d
                }
                loss = DQN_agent.update(transition_dict)
                loss_list.append(loss)
        # 更新探索率
        DQN_agent.update_epsilon()
        # 更新学习率
        DQN_agent.optimizer.step()
        DQN_agent.scheduler.step()
        return_list.append(episode_return)

        if episode_return > best_return:
            best_return = episode_return
            DQN_agent.save_best_model(best_q_net_save_path)
            print(f"第{i_episode}回合保存最优模型")

        DQN_agent.save_model(q_net_save_path, target_q_net_save_path, return_list, loss_list,
                             i_episode, case_index, DQN_agent.epsilon, best_return)
        save_replay_buffer(replay_buffer, replay_buffer_path)

        if (i_episode) % 10 == 0:
            pbar.set_postfix({'episode': f'{i_episode}', 'return': f'{np.mean(return_list[-10:]):.2f}'})
            plot_learning_curve(list(range(len(return_list))), return_list, 'DQN on LawyerSuspectEnv')

        pbar.update(1)
        case_index = case_index + 1
        print(f"此episode的return值为： {episode_return:.2f}")

# Plot results
plot_loss_curve(list(range(len(loss_list))), loss_list, 'DQN on LawyerSuspectEnv')
plot_learning_curve(list(range(len(return_list))), return_list, 'DQN on LawyerSuspectEnv')
mv_return = moving_average(return_list, 4)
plot_learning_curve(list(range(len(mv_return))), mv_return, 'DQN on LawyerSuspectEnv')

# Final evaluation
print("开始加载最优模型进行大规模评估...")
DQN_agent.load_model(best_q_net_save_path, target_q_net_save_path, checkpoint_path)
final_eval_episodes = 500
final_avg_return = evaluate_agent(env, DQN_agent, final_eval_episodes, case_index=500)
print(final_avg_return)
print(f"最终评估: 在 {final_eval_episodes} 轮评估中，平均回报为 {final_avg_return[0]:.2f}")
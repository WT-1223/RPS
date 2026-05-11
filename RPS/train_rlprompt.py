import gym
from tqdm import tqdm
from gym.envs.registration import register
from DB import DB
from agent import Agent
from replay_buffer import ReplayBuffer, save_replay_buffer, load_replay_buffer
from dqn_agent import DQN
from config import *
from utils import *
from policy_model import *
from reward_calculator_Cos import RewardCalculator

# Initialize components
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

replay_buffer = ReplayBuffer(buffer_size)
dialogue_agent = Agent()
db = DB(host='localhost', user='root', password='020201', database='law_data_total')

# Initialize environment
register(
    id='lawyer_suspector_env_v0',
    entry_point='Lawyer-Suspector-Env.lawyer-suspector-env-RL:LawyerSuspectorEnv',
)
env = gym.make('lawyer_suspector_env_v0', db_connection=db, agent=dialogue_agent, rl_model='DQN')

lawyer_policy_model = PolicyNetwork(prompt_length=10)
lawyer_optim = optim.Adam(lawyer_policy_model.parameters(), lr=1e-3)


episode, case_index = 0, 30
# Training loop
remaining_episodes = num_episodes = 470
with tqdm(total=remaining_episodes, initial=episode + 1, desc=f"Training or Evaluating Progress") as pbar:
    for i_episode in range(episode + 1, num_episodes + 1):
        episode_return = []
        state = env.reset(case_index=case_index)
        done = False
        lawyer_prompts_with_log_probs = []
        while not done:
            dialogue_history = env.get_dialog_history_text()
            prompt_str, log_prob_tensor = lawyer_policy_model.generate_prompt_with_log_prob(dialogue_history, top_k=256)
            lawyer_prompts_with_log_probs.append((prompt_str, log_prob_tensor))
            # 2. 环境交互：执行对话并获取规范化奖励
            # 分离提示词列表和对数概率列表
            prompts_only = [p for p in lawyer_prompts_with_log_probs]
            print(prompts_only)
            log_probs_only = [p[1] for p in lawyer_prompts_with_log_probs]

            next_state, reward, done, _  = env.step(prompts_only[0])
            episode_return.append(reward)

        normalized_rewards = RewardCalculator(agent=dialogue_agent, db_connection=db).normalize(rewards=episode_return)
        case_index = case_index + 1
            # 3. 策略更新
        lawyer_optim.zero_grad()
        lawyer_loss = calculate_rl_loss(log_probs_only, normalized_rewards)
        lawyer_loss.backward()
        lawyer_optim.step()

save_path = r"F:\王涛\E\RPS\RPS\checkpoints\optimized_lawyer_policy.pth"  # 建议全英文
os.makedirs(os.path.dirname(save_path), exist_ok=True)

torch.save(lawyer_policy_model.mlp.state_dict(), save_path)
print("Saved to:", save_path)
torch.save(lawyer_policy_model.mlp.state_dict(), "F:\王涛\E\RPS\RPS\checkpoints\optimized_lawyer_policy.pth")

# Plot results


# Final evaluation
print("开始加载最优模型进行大规模评估...")
lawyer_policy_model = PolicyNetwork(prompt_length=10)
state_dict = torch.load("F:\王涛\E\RPS\RPS\checkpoints\optimized_lawyer_policy.pth")
lawyer_policy_model.mlp.load_state_dict(state_dict)
lawyer_policy_model.eval() # 切换到评估模式
print(f"律师 Policy Network (MLP) 参数已从 F:\王涛\E\RPS\RPS\checkpoints\optimized_lawyer_policy.pth 加载。")
final_eval_episodes = 444
case_index=556
total_return = 0

with torch.no_grad():
    for _ in range(final_eval_episodes):
        state = env.reset(case_index=case_index)
        done = False
        episode_return = 0
        lawyer_prompts_with_log_probs = []
        while not done:
            dialogue_history = env.get_dialog_history_text()
            prompt_str, log_prob_tensor = lawyer_policy_model.generate_prompt_with_log_prob(dialogue_history,
                                                                                            top_k=256)
            lawyer_prompts_with_log_probs.append((prompt_str, log_prob_tensor))
            # 2. 环境交互：执行对话并获取规范化奖励
            # 分离提示词列表和对数概率列表
            prompts_only = [p for p in lawyer_prompts_with_log_probs]
            log_probs_only = [p[1] for p in lawyer_prompts_with_log_probs]

            next_state, reward, done, _ = env.step(min(prompts_only, key=lambda x: x[1].item())[0].replace('}', ''))
            episode_return += reward

        total_return += episode_return.sum()
        case_index = case_index + 1
lawyer_policy_model.train()
avg_return = total_return / num_episodes
# final_avg_return = evaluate_agent(env, DQN_agent, final_eval_episodes, case_index=500)
print(avg_return)
print(f"最终评估: 在 {final_eval_episodes} 轮评估中，平均回报为 {avg_return:.2f}")
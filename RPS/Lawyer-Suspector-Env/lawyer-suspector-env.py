import numpy as np
import json
import gym
from gym import spaces
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer
from reward_calculator_Cos import RewardCalculator
import os


class LawyerSuspectorEnv(gym.Env):
    """自定义律师与嫌疑人对话环境，基于强化学习的对话策略选择。"""

    def __init__(self, db_connection, agent, rl_model):
        """初始化环境。

        Args:
            db_connection: 数据库连接对象。
            agent: 生成对话的智能体。
            rl_model: 强化学习模型选择（如'PPO'或'DQN'）。
        """
        super(LawyerSuspectorEnv, self).__init__()
        self.action_space = spaces.Discrete(5)  # 动作空间：重述、虚张声势、直接询问、选择、大模型生成
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(768,), dtype=np.float32)  # 观测空间为768维向量
        self.sentence_model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')  # 使用Sentence BERT模型
        self.db = db_connection  # 数据库连接
        self.agent = agent  # 生成对话的智能体
        self.rl_model = rl_model  # 强化学习模型选择
        self.reward_calculator = RewardCalculator(agent, db_connection)  # RewardCalculator初始化
        self.chat_history = []  # 存储对话历史
        self.current_step = 0  # 跟踪当前对话轮次
        self.max_steps = 10  # 10轮对话为结束条件
        self.case_info = None  # 存储案件信息
        self.case_index = 0  # 跟踪当前案件索引，按顺序获取案件
        self.similarity = []  # 用于计算奖励值的相似度列表

    def reset(self, case_index=0):
        """重置环境并加载新案件。

        Args:
            case_index: 案件索引，默认为0。

        Returns:
            np.ndarray: 初始状态向量。
        """
        self._initialize_environment(case_index)
        self._load_case_info()
        dialog_vector = self._initialize_dialogue()
        self._save_initial_dialogue()

        return dialog_vector  # agent在初始时能够观察到的状态向量

    def _initialize_environment(self, case_index):
        """初始化环境变量。"""
        self.chat_history.clear()  # 清空对话历史
        self.current_step = 0  # 记录当前是第几步
        self.clear_memory()  # 清除记忆
        self.case_index = case_index  # 需要从外部传递，目的是为了满足断电重连的功能
        self.similarity = []  # 清空用于计算奖励值的列表

    def _load_case_info(self):
        """从数据库中加载案件信息。"""
        messages = self.db.fetch_records("*", "law_data_total_detail")  # 从数据库中获取案件信息
        if not messages:
            raise ValueError("数据库中没有可用案件。")  # 确保数据库中有可用案件

        if self.case_index >= len(messages):
            self.case_index = 0  # 如果超出了案件列表的长度，重置案件索引，确保从第一个案件开始

        case = messages[self.case_index]  # 获取案件索引为case_index的案件
        self.case_info = self._process_case(case)  # 提取案件信息

    def _initialize_dialogue(self):
        """初始化对话。"""
        initial_lawyer_input = "您好，我是律师。请放心，我会尽全力维护您的权益。"
        suspect_output = "您好，我是当事人。我需要您的帮助。"  # 初始化对话（律师和当事人初始对话）
        self.chat_history.append({"律师": initial_lawyer_input, "当事人": suspect_output, "策略": "N/A"})  # 把初始化的对话加入对话历史列表中
        dialog_texts = []
        for entry in self.chat_history:
            lawyer_text = entry["律师"]
            suspect_text = entry["当事人"]
            dialog_texts.append(f"律师: {lawyer_text} | 当事人: {suspect_text}")

            # 将所有对话的文本合并成一个大字符串
            combined_dialogue = " ".join(dialog_texts)

            # 将合并后的对话文本嵌入为向量
            dialog_vector = self.sentence_model.encode(combined_dialogue)
        print(f"\n轮次: {self.current_step} 策略: N/A")
        print(f"律师: {self.chat_history[-1]['律师']}")
        print(f"当事人: {self.chat_history[-1]['当事人']}")
        return dialog_vector

    def _save_initial_dialogue(self):
        """保存初始对话到数据库。"""
        if self.rl_model == 'PPO':
            self.db.save_to_db_PPO(self.case_info["id"], 0, self.chat_history[-1])
        else:
            self.db.save_to_db_DQN(self.case_info["id"], 0, self.chat_history[-1])  # 保存第一轮对话到数据库

    def _process_case(self, case):
        """处理并验证案件信息。

        Args:
            case: 从数据库中获取的案件信息。

        Returns:
            dict: 处理后的案件信息。
        """
        case_info = {
            "fact": case.get("fact", ""),  # 从字典对象中提取，get("key", "default_value")
            "buli": case.get("buli", ""),
            "youli": case.get("youli", ""),
            "zhongli": case.get("zhongli", ""),
            "extraction_content": case.get("extraction_content", ""),
            "plaintiff": case.get("plaintiff", ""),
            "defendant": case.get("defendant", ""),
            "id": case.get("id")  # 确保案件ID存在
        }
        if case_info["id"] is None:
            raise ValueError("案件信息缺少ID")  # 如果没有ID，抛出错误
        return case_info

    def step(self, action):
        """根据律师的策略进行对话并更新状态。"""
        self._validate_action(action)
        self.current_step += 1
        done = self.current_step == self.max_steps

        if action in [0, 1, 2, 3]:
            policy = self._map_action_to_policy(action)
            lawyer_output, suspect_output = self._generate_dialogue(policy)
        elif action == 4:
            # 大模型生成prompt
            prompt = self._generate_prompt_policy_action()
            lawyer_output, suspect_output = self._generate_dialogue_with_custom_prompt(prompt)
            policy = f"PromptPolicy: {prompt[:20]}..."  # 记录动作内容
        else:
            raise ValueError("无效的动作。")

        self._update_chat_history(lawyer_output, suspect_output, policy)
        dialog_vector = self._encode_dialogue()
        reward = self._calculate_reward()
        self._print_dialogue_info(policy, lawyer_output, suspect_output, reward)
        self._save_dialogue_to_db()
        return dialog_vector, reward, done, {}

    def _validate_action(self, action):
        """验证动作是否有效。"""
        if action not in [0, 1, 2, 3, 4]:
            raise ValueError("无效的动作。")

    def _map_action_to_policy(self, action):
        """将动作映射到策略。"""
        return ["Accurate-evidence", "Adversarial", "Comprehensive", "Multi-angle"][action]

    def _generate_dialogue(self, policy):
        """生成对话。"""
        lawyer_prompt = self._generate_lawyer_prompt(policy)
        suspector_prompt = self._generate_suspector_prompt()

        lawyer = self.agent.init_chatbot(lawyer_prompt, "qwen2:7b", self.agent.lawyer_memory)
        suspect = self.agent.init_chatbot(suspector_prompt, "qwen2:7b", self.agent.party_memory)

        suspector_output = self.chat_history[-1]['当事人']
        lawyer_output = lawyer({"question": suspector_output})["text"]
        suspect_output = suspect({"question": lawyer_output})["text"]

        return lawyer_output, suspect_output

    def _update_chat_history(self, lawyer_output, suspect_output, policy):
        """更新对话历史。"""
        self.chat_history.append({"律师": lawyer_output, "当事人": suspect_output, "策略": policy})

    # def _encode_dialogue(self, lawyer_output, suspect_output):
    #     """将对话编码为向量。"""
    #     last_dialog_text = f"律师: {lawyer_output} | 当事人: {suspect_output}"
    #     return self.sentence_model.encode(last_dialog_text)

    def _encode_dialogue(self):
        """将所有对话历史编码为向量。"""
        dialog_texts = []
        for entry in self.chat_history:
            lawyer_text = entry["律师"]
            suspect_text = entry["当事人"]
            dialog_texts.append(f"{lawyer_text}{suspect_text}")

        # 将所有对话的文本合并成一个大字符串
        combined_dialogue = " ".join(dialog_texts)

        # 将合并后的对话文本嵌入为向量
        dialog_vector = self.sentence_model.encode(combined_dialogue)

        return dialog_vector

    def _calculate_reward(self):
        """计算奖励值。"""
        if self.current_step != 1:
            similarities = self.reward_calculator.calculate_reward(self.chat_history, self.case_info["id"])
            self.similarity.append(similarities.squeeze(0).tolist())
            current_max = np.max(np.array(self.similarity), axis=0)
            if len(self.similarity) > 1:
                previous_max = np.max(np.array(self.similarity[:-1]), axis=0)
                improvement = np.maximum((current_max - previous_max) / (1-previous_max), 0)  # 负值变为0
                final_improvement = [i for i in improvement if i > 0]  # 只保留正值
                improvement_count = len(final_improvement)  # 统计有提升的项目数量
                reward_1 = sum(final_improvement) / improvement_count if improvement_count > 0 else 0
            else:
                reward_1 = 0
            reward_2 = reward_1
            # print(reward_2)
        else:
            similarities = self.reward_calculator.calculate_reward(self.chat_history, self.case_info["id"])
            self.similarity.append(similarities.squeeze(0).tolist())
            reward_2 = 0

        return reward_2

    def _print_dialogue_info(self, policy, lawyer_output, suspect_output, reward):
        """打印对话信息。"""
        print(f"\n轮次: {self.current_step} 策略: {policy}")
        print(f"律师: {lawyer_output}")
        print(f"当事人: {suspect_output}")
        print(f"奖励值：{round(reward, 2)}")

    def _save_dialogue_to_db(self):
        """保存对话到数据库。"""
        if self.rl_model == 'PPO':
            self.db.save_to_db_PPO(self.case_info["id"], self.current_step, self.chat_history[-1])
        else:
            self.db.save_to_db_DQN(self.case_info["id"], self.current_step, self.chat_history[-1])

    def get_dialog_history_text(self):
        """将对话历史转换为字符串格式。

        Returns:
            str: 对话历史的字符串表示。
        """
        return " ".join(
            [f"律师: {entry['律师']} 当事人: {entry['当事人']} 策略: {entry['策略']}" for entry in self.chat_history]
        )

    def _generate_suspector_prompt(self):
        """根据案件信息生成嫌疑人的提示词。

        Returns:
            str: 嫌疑人提示词。
        """
        return self.agent.read_file("prompt/suspector_prompt.txt").format(
            party=self.case_info["defendant"],
            pla_or_def="被告" if self.case_info["defendant"] else "原告",
            fact=self.case_info["fact"],
            buli=self.case_info["buli"],
            zhongli=self.case_info["zhongli"],
            youli=self.case_info["youli"],
            advance=""
        )

    def _generate_lawyer_prompt(self, policy):
        """获取律师的提示词。

        Args:
            policy: 当前策略。

        Returns:
            str: 律师提示词。
        """
        # with open('prompt/policy_prompt.json', 'r', encoding='utf-8') as file:
        #     data = json.load(file)
        try:
            with open(f'prompt/policy/{policy}.txt', 'r', encoding='utf-8') as file:
                policy = file.read()
                # print(policy)
            # policy = data["policy"][policy]
        except KeyError as e:
            print(f"参数错误: 缺少字段 {e}")

        with open("prompt/lawyer_prompt.txt", 'r', encoding='utf-8') as file:
            prompt = file.read()

        return prompt.format(policy=policy)

    def clear_memory(self):
        """清除之前的对话记忆。"""
        self.agent.lawyer_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.agent.party_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def render(self, mode='human'):
        """渲染当前对话历史。

        Args:
            mode: 渲染模式，默认为'human'。
        """
        print(self.get_dialog_history_text())

    def _generate_prompt_policy_action(self):
        """
        使用大模型根据对话历史生成prompt，并保存到prompt/LLM-prompt目录。
        Returns:
            str: 生成的prompt文本。
        """
        dialog_history = self.get_dialog_history_text()
        prompt = self.agent.generate_prompt(dialog_history)
        # 保存prompt到文件
        case_id = self.case_info["id"]
        step = self.current_step
        save_dir = "prompt/LLM-prompt-detail"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{case_id}_第{step}轮.txt")
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(prompt)
        return prompt

    def _generate_dialogue_with_custom_prompt(self, prompt):
        """
        用自定义prompt生成律师发言。
        Args:
            prompt (str): 由大模型生成的律师prompt。
        Returns:
            tuple: (律师发言, 当事人发言)
        """
        suspector_prompt = self._generate_suspector_prompt()
        lawyer = self.agent.init_chatbot(prompt, "qwen2:7b", self.agent.lawyer_memory)
        suspect = self.agent.init_chatbot(suspector_prompt, "qwen2:7b", self.agent.party_memory)
        suspector_output = self.chat_history[-1]['当事人']
        lawyer_output = lawyer({"question": suspector_output})["text"]
        suspect_output = suspect({"question": lawyer_output})["text"]
        return lawyer_output, suspect_output
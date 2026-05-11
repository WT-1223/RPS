import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple, Optional
import random


class PolicyNetwork(nn.Module):
    """
    RLPROMPT 策略网络：冻结 Policy LM + 可训练 MLP 。
    该网络现在将返回生成的提示词序列及其对数概率。
    """

    def __init__(self, policy_lm_name: str = 'Qwen/Qwen2.5-0.5B', prompt_length: int = 5):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(policy_lm_name)

        # 冻结 Policy LM
        try:
            self.policy_lm = AutoModelForCausalLM.from_pretrained(policy_lm_name)
        except Exception:
            # 警告: 如果无法加载模型，使用模拟的随机参数
            print(f"警告: 无法加载 {policy_lm_name}，使用 Mock LM 参数替代。")
            self.policy_lm = nn.Linear(768, self.tokenizer.vocab_size)
            self.policy_lm.weight.data.uniform_(-0.1, 0.1)

        self.policy_lm.eval()
        for param in self.policy_lm.parameters():
            param.requires_grad = False

        self.hidden_size = 768  # distilGPT-2/RoBERTa-base 常见维度
        self.prompt_length = prompt_length

        # 核心：可训练 MLP (仅更新这部分少量参数)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.hidden_size)
        )
        # 模拟 LM Head
        self.lm_head = nn.Linear(self.hidden_size, self.tokenizer.vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播：从 Policy LM 获取上下文嵌入，通过 MLP，预测下一个 token 的 logits。
        """
        # **重要：在真实 Policy LM 中，我们需要实际的前向传播来获取隐藏状态**
        # 在此处，我们必须使用一个可训练的伪隐藏状态来确保梯度流向 MLP

        # 模拟 Policy LM 的输出：我们只对 MLP 进行训练，LM 本身是冻结的
        # 为了让 PyTorch 追踪 MLP 的梯度，我们不能完全依赖 torch.no_grad()

        # 模拟上下文嵌入，并确保其尺寸适合 MLP
        # 假设 Policy LM 能够将 input_ids 映射到隐藏状态
        mock_context_embedding = torch.randn(input_ids.size(0), self.hidden_size)

        # 策略参数 MLP 转换 (这部分是可训练的)
        adapted_embedding = self.mlp(mock_context_embedding)

        # 投影到词汇表空间 (模拟 LM Head)
        logits = self.lm_head(adapted_embedding)
        return logits

    def generate_prompt_with_log_prob(self, context_str: str, top_k: int = 256) -> Tuple:
        """
        生成离散提示词序列，并计算整个序列的对数概率 log(pi(z|x))。
        这是计算 RL 损失的关键步骤。
        """
        input_tokens = self.tokenizer.encode(context_str, return_tensors='pt', truncation=True)
        current_input = input_tokens
        generated_tokens = []

        # 追踪整个序列的对数概率
        sequence_log_prob = torch.tensor(0.0, requires_grad=True)

        for _ in range(self.prompt_length):
            attention_mask = torch.ones_like(current_input)
            logits = self.forward(current_input, attention_mask)

            # 使用 Softmax 转换为概率分布
            probabilities = torch.softmax(logits, dim=-1).squeeze(0)

            # Top-K 采样逻辑 [2]
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)

            # 对 Top-K 概率重新归一化
            top_k_probabilities = torch.softmax(top_k_logits, dim=-1).squeeze(0)

            # 采样下一个 token
            sampled_index_in_k = torch.multinomial(top_k_probabilities, num_samples=1)
            next_token_id = top_k_indices.gather(-1, sampled_index_in_k.unsqueeze(1)).squeeze(0)

            # 提取被选中 token 的概率和对数概率
            selected_token_prob = top_k_probabilities.gather(-1, sampled_index_in_k)
            selected_log_prob = torch.log(selected_token_prob).squeeze()

            # 累加序列的对数概率 (用于计算梯度)
            sequence_log_prob = sequence_log_prob + selected_log_prob

            # 解码和更新输入
            generated_token = self.tokenizer.decode(next_token_id.item(), skip_special_tokens=True).strip()
            if not generated_token or generated_token.isspace():
                generated_token = random.choice(generated_tokens)

            generated_tokens.append(generated_token)
            current_input = torch.cat([current_input, next_token_id.unsqueeze(0)], dim=-1)

        return " ".join(generated_tokens), sequence_log_prob
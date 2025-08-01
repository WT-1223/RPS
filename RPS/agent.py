import warnings
import datetime
import json
from openai import OpenAI
from collections import Counter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

warnings.filterwarnings("ignore")


class Agent:
    def __init__(self):
        self.current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
        self.party_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.lawyer_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.model_config = {
            "model_name": [
                "glm",
                "gpt",
                "deepseek",
                "moonshot"
            ],
            "config": [
                {
                    "name": "glm",
                    "api_key": "cbbea13466b92475b635d95daf06fd41.MJAnx3BinduYwqKA",
                    "base_url": "https://open.bigmodel.cn/api/paas/v4"
                },
                {
                    "name": "gpt",
                    "api_key": "sk-MUlbPssibf9TQeKyCb0bAbD93b0e435aBaC4D325A320Fd99",
                    "base_url": "https://api.rcouyi.com/v1"
                },
                {
                    # "name": "deepseek",
                    "name": "qwen",
                    "api_key": "ollama",
                    "base_url": "http://localhost:11434/v1/"
                    # "api_key": "sk-6fa9758572754f4882f95e6bddc1fbe0",
                    # "base_url": "https://api.deepseek.com"
                },
                {
                    "name": "moonshot",
                    "api_key": "sk-lSc2EpX8GpyPnOjQDEb9WKiQHSzGJ0zWKMwhzyBdzAzJ2bqO",
                    "base_url": "https://api.moonshot.cn/v1"
                }
            ]
        }

    def read_file(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            return file.read()

    def write_to_json(self, filename, content):
        with open(filename, 'w', encoding="utf-8") as file:
            json.dump(content, file, indent=4, ensure_ascii=False)

    def find_mode(self, lst):
        if not lst:
            return None
        counter = Counter(lst)
        mode, _ = counter.most_common(1)[0]
        return mode

    def model(self, prompt, model_name, format=None, n=1, max_retries=10):
        for model in self.model_config["config"]:
            if model["name"] in model_name.lower():
                api_key = model["api_key"]
                base_url = model["base_url"]
                break

        client = OpenAI(api_key=api_key, base_url=base_url)

        completion = client.chat.completions.create(
            model=model_name,
            response_format={"type": format},
            messages=[{"role": "system", "content": "你是一个有用的智能助手"},
                      {"role": "user", "content": prompt}],
            top_p=0.5,
            temperature=0.2,
            n=n
        )

        if n == 1:
            return completion.choices[0].message.content
        else:
            return [item.message.content for item in completion.choices]

    def init_chatbot(self, template, model_name, memory):
        for model in self.model_config["config"]:
            if model["name"] in model_name.lower():
                api_key = model["api_key"]
                base_url = model["base_url"]

        llm = ChatOpenAI(temperature=0.6,
                         model=model_name,
                         openai_api_key=api_key,
                         openai_api_base=base_url)

        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")
        prompt = ChatPromptTemplate(
            messages=[system_message_prompt, MessagesPlaceholder(variable_name="chat_history"), human_message_prompt]
        )

        conversation = LLMChain(llm=llm, prompt=prompt, memory=memory)
        return conversation

    def generate_prompt(self, dialog_history, system_prompt=None):
        """
        使用qwen大模型根据对话历史生成结构化prompt。
        Args:
            dialog_history (str): 当前对话历史文本。
            system_prompt (str): 可选，系统指令。
        Returns:
            str: 生成的结构化prompt文本。
        """
        if system_prompt is None:
            system_prompt = (
                "你是律师对话策略生成器。请根据当前律师与当事人的全部对话历史，"
                "生成下一步律师可以采用的对话策略，要求输出如下结构：\n"
                "核心目的：简要描述本轮提问的核心目标。\n"
                "要点：分条列出提问时应注意的策略或技巧。\n"
                "例子：给出1-2个本轮可用的具体提问句式。\n"
                "注意：不要直接输出律师的发言内容，只输出上述结构化内容。"
            )
        prompt = f"{system_prompt}\n\n对话历史：{dialog_history}"
        return self.model(prompt, model_name="qwen:32b")

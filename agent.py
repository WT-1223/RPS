import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import LLMMathChain
from langchain.agents import Tool
from collections import Counter
from langchain_core.prompts import (
ChatPromptTemplate,
MessagesPlaceholder,
SystemMessagePromptTemplate,
HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import datetime
import json
# 忽略所有警告
import warnings

from  tqdm import tqdm
warnings.filterwarnings("ignore")

# from transformers import AutoModelForCausalLM, AutoTokenizer

class Agent:
    def __init__(self):
        self.current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
        self.party_memory=  ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.lawyer_memory= ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.model_config={
  "model_name": [
    "glm",
    "gpt",
    "deepseek",
    "moonshot"
  ],
  "config": [
    {
      "name": "glm",
      "api_key": "5a4f9b5dc7162e4e456801cc61fdcae1.vMUfFqmKDyV9gpHh",
      "base_url": "https://open.bigmodel.cn/api/paas/v4/"
    },
    {
      "name": "gpt",
      "api_key": "sk-MUlbPssibf9TQeKyCb0bAbD93b0e435aBaC4D325A320Fd99",
      "base_url": "https://api.rcouyi.com/v1"
    },
    {
      "name": "qwen",
      # "name": "deepseek",
      "api_key": "ollama",
      "base_url": "http://localhost:11434/v1/"
      # "api_key": "sk-f9423bf8a6794ac5a1ebcbde818672e2",
      # "base_url": "https://api.deepseek.com/v1"
    },
    {
      "name": "moonshot",
      "api_key": "sk-uVa7wiPGUcSUl1fXGDS847dbZhhFUPLa048mA6yVPrdr6fPN",
      "base_url": "https://api.moonshot.cn/v1"
    }
  ]
}
        # self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
        # self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct").to("cuda")
        # self.generate_kwargs = {
        #     'max_new_tokens': 512,
        #     'do_sample': True,
        #     'top_p': 0.9,
        #     'temperature': 0.7,
        #     'pad_token_id': self.tokenizer.eos_token_id
        # }

    def read_file(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    
    def write_to_json(self, filename,content):
        with open(filename, 'w',encoding="utf-8") as file:
            json.dump(content, file, indent=4,ensure_ascii=False)

    def find_mode(self,lst):
        if not lst:
            return None  # 如果列表为空，返回 None

        counter = Counter(lst)
        mode, count = counter.most_common(1)[0]  # 获取出现频率最高的元素及其频率
        return mode

    def generateprompt(self,personality, policy, party):
        # 读取 JSON 文件
        with open('prompt/classify_prompt_v2.json', 'r', encoding='utf-8') as file:
            data = json.load(file)

        try:
            personality = data["personality"][personality]
            with open(f'prompt/policy/{policy}.txt', 'r', encoding='utf-8') as file:
                policy = file.read()
                print(policy)

        except KeyError as e:
            print(f"参数错误: 缺少字段 {e}")
        # 读取文件内容
        with open("prompt/lawyer_v2.txt", 'r', encoding='utf-8') as file:
            content = file.read()

        # 替换占位符
        content = content.format(party=party, personality=personality, policy=policy)
        return content

    def model(self, prompt, model_name,format=None,n=1):
        for model in self.model_config["config"]:
            if model["name"] in model_name.lower():
                api_key = model["api_key"]
                base_url = model["base_url"]

        client = OpenAI(
            api_key=api_key,
            base_url= base_url

        )
        completion = client.chat.completions.create(
            model=model_name,
            # response_format={"type": format},
            messages=[
                {"role": "system", "content": "你是一个有用的智能对话助手"},
                {"role": "user",
                 "content": prompt}
            ],
            top_p=0.5,
            temperature=0.2,
            n=n
        )

        # print(completion.choices[0].message)
        all_reslut = []
        if n==1:
            return completion.choices[0].message.content
        else:
            for item in completion.choices:
                all_reslut.append(item.message.content)
            return all_reslut

    # def init_chatbot(self, system_prompt):
    #     # 构造初始的 messages 列表
    #     messages = [{'role': 'system', 'content': system_prompt}]
    #
    #     def chatbot_fn(inputs):
    #         user_input = inputs['question']
    #         messages.append({'role': 'user', 'content': user_input})
    #
    #         prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    #         inputs_tensor = self.tokenizer([prompt_text], return_tensors='pt').to(self.model.device)
    #         outputs = self.model.generate(**inputs_tensor, **self.generate_kwargs)
    #
    #         response = self.tokenizer.decode(outputs[0][inputs_tensor["input_ids"].shape[-1]:],
    #                                          skip_special_tokens=True)
    #         response = response.strip()
    #
    #         messages.append({'role': 'assistant', 'content': response})
    #
    #         return {'text': response}
    #
    #     return chatbot_fn
    def init_chatbot(self, template,model_name,memory):
        for model in self.model_config["config"]:
            if model["name"] in model_name.lower():
                api_key = model["api_key"]
                base_url = model["base_url"]

        llm = ChatOpenAI( temperature=0.6,
                            model=model_name,
                            openai_api_key=api_key,
                            openai_api_base=base_url
                          # base_url="http://10.220.138.110:8000/v1",
                          # api_key="EMPTY",

                          )
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")
        prompt = ChatPromptTemplate(
            messages=[system_message_prompt, MessagesPlaceholder(variable_name="chat_history"), human_message_prompt])

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

if __name__ == '__main__':
    # 调用示例
    agent=Agent()
    prompt_lawyer = agent.generateprompt("extravert", "restate", "defendant")
    lawyer = agent.init_chatbot(prompt_lawyer, 'deepseek-chat', agent.lawyer_memory)
    print(lawyer)
    # memore=agent.party_memory
    # role1=agent.init_chatbot("","glm-4",memore)
    # response=role1({"question": "你是那个公司开发的"})["text"]
    # print(response)
    # response=agent.model("你是谁","glm-4","text",1)
    # print(response)
from tqdm import tqdm
from agent import Agent
from DB import DB
import ast
import json
import threading
from langchain.memory import ConversationBufferMemory

import pymysql


class Conversation:
    def __init__(self,conversation_round):
        self.conversation_round = conversation_round

    def run_case(self,personality,policy,table,start,applicant):
        db = DB(host='localhost', user='root', password='020201', database='deception_detection')
        messages = db.fetch_records("*","law_data_total_detail")

        for item in tqdm(messages[start:]):
            plaintiff=item["Plaintiff"]
            defendant=item["Defendant"]
            id = item["id"]
            fact = item["Case_Content"]
            buli_info = item["Unfavorable_Information"]
            youli_info = item["Favorable_Information"]
            zhongli_info = item["Neutral_Information"]

            if policy not in ["normal","normal_blank"]:
                prompt_lawyer = agent.generateprompt(personality, policy, defendant)
            elif policy == "normal":
                prompt_lawyer = agent.read_file("prompt/lawyer_normal.txt")
            else:
                prompt_lawyer = ""

            if applicant == "defendant":
                prompt_suspector = agent.read_file("prompt/suspector.txt").format(party=defendant,
                                                                              pla_or_def="被告",
                                                                              fact=fact,
                                                                              buli=buli_info,
                                                                              zhongli=zhongli_info,
                                                                              youli=youli_info,
                                                                              advance="")
            else:
                prompt_suspector = agent.read_file("prompt/suspector.txt").format(party=plaintiff,
                                                                              pla_or_def="原告",
                                                                              fact=fact,
                                                                              buli=youli_info,
                                                                              zhongli=zhongli_info,
                                                                              youli=buli_info,
                                                                              advance="")
            print(prompt_suspector)

            model_name="qwen2:7b"
            # model_name = "moonshot-v1-8k"
            # 清除上一轮对话留下来的记忆
            agent.lawyer_memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            agent.party_memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            lawyer = agent.init_chatbot(prompt_lawyer, model_name,agent.lawyer_memory)
            suspector = agent.init_chatbot(prompt_suspector, model_name,agent.party_memory)
            # lawyer = agent.init_chatbot(prompt_lawyer)
            # suspector = agent.init_chatbot(prompt_suspector)
            # # 存储对话记录的路径
            if self.conversation_round == 1:
                chat_history = []
                conversation = 1
                while conversation <= 10:
                    chat_history_index = []
                    # 第一轮当事人先说话
                    initial_suspector_input = "您好，我是律师。请放心，我会尽全力维护您的权益。"
                    suspect_output = "您好，我是当事人。我需要您的帮助。"
                    # 记录初始对话
                    chat_history_index.append({"律师": initial_suspector_input, "当事人": suspect_output})
                    # 设置对话最大轮次
                    max_rounds = 20
                    # 开始对话
                    for round_num in range(1, max_rounds + 1):
                        if round_num % 2 == 0:  # 偶数轮，当事人回答

                            suspect_output = suspector({"question": lawyer_output})["text"]

                            print(f"当事人: {suspect_output}")

                            chat_history_index.append({"律师": lawyer_output, "当事人": suspect_output})
                        else:  # 奇数轮，律师提问
                            lawyer_output = lawyer({"question": suspect_output})["text"]
                            print(f"律师: {lawyer_output}")

                    chat_history_index = str(chat_history_index)
                    chat_history.append(chat_history_index)
                    conversation += 1
                chat_history = str(chat_history)
                db.insert_or_update_record_to_direct(id,chat_history,table)
            else:
                chat_history_index = []
                # 第一轮当事人先说话
                initial_suspector_input = "您好，我是律师。请放心，我会尽全力维护您的权益。"
                suspect_output = "您好，我是当事人。我需要您的帮助。"
                # 记录初始对话
                chat_history_index.append({"律师": initial_suspector_input, "当事人": suspect_output})
                # 设置对话最大轮次
                max_rounds = 20
                # 开始对话
                for round_num in range(1, max_rounds + 1):
                    if round_num % 2 == 0:  # 偶数轮，当事人回答

                        suspect_output = suspector({"question": lawyer_output})["text"]

                        print(f"当事人: {suspect_output}")

                        chat_history_index.append({"律师": lawyer_output, "当事人": suspect_output})
                    else:  # 奇数轮，律师提问
                        lawyer_output = lawyer({"question": suspect_output})["text"]
                        print(f"律师: {lawyer_output}")

                chat_history_index = str(chat_history_index)
                db.insert_or_update_record_to_direct(id, chat_history_index, table)


if __name__ == '__main__':
    agent = Agent()

    # thread1 = threading.Thread(target=Conversation(0).run_case, args=("extravert", "restate", "restate_total_defendant", 0, "defendant"))
    # thread2 = threading.Thread(target=Conversation(0).run_case, args=("extravert", "normal", "normal", 500, "defendant"))
    # thread3 = threading.Thread(target=Conversation(0).run_case, args=("extravert", "choice", "choice_total_defendant", 0, "defendant"))
    # thread4 = threading.Thread(target=Conversation(0).run_case, args=("extravert", "bluff", "bluff_total_defendant", 500, "defendant"))
    # thread5 = threading.Thread(target=Conversation(0).run_case, args=("extravert", "direct", "direct_total_defendant", 0, "defendant"))
    # thread6 = threading.Thread(target=Conversation(0).run_case, args=("extravert", "normal" , "normal_blank", 0, "defendant"))
    thread7 = threading.Thread(target=Conversation(0).run_case, args=("extravert", "Accurate-evidence", "accurate_detail", 500, "defendant"))
    thread8 = threading.Thread(target=Conversation(0).run_case, args=("extravert", "Multi-angle", "multi_angle_detail", 500, "defendant"))
    thread9 = threading.Thread(target=Conversation(0).run_case, args=("extravert", "Comprehensive", "comprehensive_detail", 500, "defendant"))
    thread10 = threading.Thread(target=Conversation(0).run_case, args=("extravert","Adversarial","adversarial_detail", 500, "defendant"))
    # 启动线程
    # thread1.start()
    # thread2.start()
    # thread3.start()
    # thread4.start()
    # thread5.start()
    # thread6.start()
    thread7.start()
    thread8.start()
    thread9.start()
    thread10.start()

    # 等待线程完成
    # thread1.join()
    # thread2.join()
    # thread3.join()
    # thread4.join()
    # thread5.join()
    # thread6.join()
    thread7.join()
    thread8.join()
    thread9.join()
    thread10.join()

    print("Done")







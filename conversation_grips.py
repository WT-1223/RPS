
from tqdm import tqdm
from agent import Agent
from DB import DB
import ast
import json
import threading
from langchain.memory import ConversationBufferMemory
import re
import random
import os

import pymysql


def split_policy_to_phrases(policy_text: str):
    """
    简单把策略 prompt 切成“句子级短语”，作为 GrIPS 的操作单元。
    按中文句号、问号、叹号和换行拆。
    """
    raw_phrases = re.split(r'[。！？\n]', policy_text)
    return [p.strip() for p in raw_phrases if p.strip()]


def paraphrase_phrase_with_llm(agent, phrase: str, chat_history_index, model_name: str):
    """
    用 LLM 做有“对话历史感知”的 paraphrase（改写）操作。
    """
    system_prompt = (
        "你是一个帮助优化律师提问策略的AI助手。"
        "给定已发生的律师-当事人对话，以及一条策略句子，"
        "请在不改变核心目标（从当事人那里挖掘真实案情信息）的前提下，"
        "用更清晰、具体、便于大模型执行的方式改写这条策略句子。"
        "只输出改写后的句子。"
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    rewriter = agent.init_chatbot(system_prompt, model_name, memory)

    history_str = json.dumps(chat_history_index, ensure_ascii=False)

    user_q = (
        f"当前对话历史：{history_str}\n"
        f"原策略句子：{phrase}\n"
        f"请改写："
    )
    resp = rewriter({"question": user_q})["text"]
    return resp.strip()


def random_edit_with_history(agent, base_text, deleted_pool, chat_history_index, model_name):
    """
    在当前策略 prompt 上做一次 GrIPS 风格的随机编辑：del/swap/par/add。
    """
    phrases = split_policy_to_phrases(base_text)
    if not phrases:
        return base_text, deleted_pool

    ops = ["del", "swap", "par", "add"]
    if not deleted_pool:
        ops = [op for op in ops if op != "add"]

    op = random.choice(ops)

    if op == "del" and len(phrases) > 1:
        idx = random.randrange(len(phrases))
        deleted_pool.append(phrases[idx])
        del phrases[idx]

    elif op == "swap" and len(phrases) > 1:
        i, j = random.sample(range(len(phrases)), 2)
        phrases[i], phrases[j] = phrases[j], phrases[i]

    elif op == "par":
        idx = random.randrange(len(phrases))
        phrases[idx] = paraphrase_phrase_with_llm(
            agent, phrases[idx], chat_history_index, model_name
        )

    elif op == "add" and deleted_pool:
        insert_phrase = random.choice(deleted_pool)
        pos = random.randrange(len(phrases) + 1)
        phrases.insert(pos, insert_phrase)

    new_text = "。".join(phrases)
    return new_text, deleted_pool


def score_policy_with_llm(agent, candidate_text, chat_history_index, model_name):
    """
    用 LLM 给某个策略 prompt 打分：
    分数含义：在下一轮对话中，这个策略有多大可能帮助律师从当事人那里获得新的、关键的事实信息。
    输出 1~10 的浮点评分。
    """
    system_prompt = (
        "你是一个评估律师询问策略质量的法律助手。"
        "根据已有对话和给定的策略提示词，"
        "评估该策略在下一轮帮助律师从当事人获得新的、关键事实信息方面的有效性，"
        "给出 1 到 10 的分数，只输出一个数字。"
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    scorer = agent.init_chatbot(system_prompt, model_name, memory)

    history_str = json.dumps(chat_history_index, ensure_ascii=False)

    user_q = (
        f"当前对话历史：{history_str}\n"
        f"候选策略提示词如下：\n{candidate_text}\n"
        f"请按照 1-10 打分，只返回数字："
    )
    resp = scorer({"question": user_q})["text"]

    try:
        m = re.search(r"([0-9]+(\.[0-9]+)?)", resp)
        if m:
            return float(m.group(1))
    except Exception:
        pass

    # 解析失败就给一个中性分
    return 5.0


def grips_update_policy(agent, current_policy_prompt, chat_history_index, model_name, num_candidates=3):
    """
    GrIPS 一步更新：以当前策略 prompt 为中心，生成若干候选 + 原始版本，一起打分，选最优。
    """
    deleted_pool = []
    candidates = [current_policy_prompt]  # 把原策略也当作一个候选，避免越改越差

    for _ in range(num_candidates):
        new_text, deleted_pool = random_edit_with_history(
            agent, current_policy_prompt, deleted_pool, chat_history_index, model_name
        )
        candidates.append(new_text)

    scored = []
    for cand in candidates:
        score = score_policy_with_llm(agent, cand, chat_history_index, model_name)
        scored.append((score, cand))

    best_score, best_cand = max(scored, key=lambda x: x[0])
    return best_cand


class Conversation:
    def __init__(self, conversation_round):
        self.conversation_round = conversation_round

    def run_case(self, personality, policy, table, start, applicant):
        db = DB(host='localhost', user='root', password='020201', database='deception_detection')
        messages = db.fetch_records("*", "law_data_total_initial")

        # === 在所有 case 之前，初始化 current_policy_text（跨 case 共享） ===
        current_policy_text = None
        if self.conversation_round != 1 and policy not in ["normal", "normal_blank"]:
            grips_path = f"prompt/policy/{policy}_grips_latest.txt"
            base_path = f"prompt/policy/{policy}.txt"
            if os.path.exists(grips_path):
                current_policy_text = agent.read_file(grips_path)
            else:
                current_policy_text = agent.read_file(base_path)

        for item in tqdm(messages[start:start+100]):
            plaintiff = item["Plaintiff"]
            defendant = item["Defendant"]
            id = item["id"]
            fact = item["Case_Content"]
            buli_info = item["Unfavorable_Information"]
            youli_info = item["Favorable_Information"]
            zhongli_info = item["Neutral_Information"]

            # === 根据是否使用 GrIPS，构造本 case 的律师 prompt ===
            if policy not in ["normal", "normal_blank"]:
                if self.conversation_round == 1:
                    # baseline：不用 GrIPS，就按原来策略名生成
                    prompt_lawyer = agent.generateprompt(personality, policy, defendant)
                else:
                    # GrIPS 模式：用当前优化后的策略文本 + defendant 生成 prompt
                    prompt_lawyer = agent.generateprompt_grips(personality, current_policy_text, defendant)
            elif policy == "normal":
                prompt_lawyer = agent.read_file("prompt/lawyer_normal.txt")
            else:
                prompt_lawyer = ""

            if applicant == "defendant":
                prompt_suspector = agent.read_file("prompt/suspector.txt").format(
                    party=defendant,
                    pla_or_def="被告",
                    fact=fact,
                    buli=buli_info,
                    zhongli=zhongli_info,
                    youli=youli_info,
                    advance=""
                )
            else:
                prompt_suspector = agent.read_file("prompt/suspector.txt").format(
                    party=plaintiff,
                    pla_or_def="原告",
                    fact=fact,
                    buli=youli_info,
                    zhongli=zhongli_info,
                    youli=buli_info,
                    advance=""
                )
            print(prompt_suspector)

            model_name = "gpt-4o-mini"
            # model_name = "moonshot-v1-8k"

            # 清除上一轮对话留下来的记忆（每个 case 重置 memory）
            agent.lawyer_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            agent.party_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

            lawyer = agent.init_chatbot(prompt_lawyer, model_name, agent.lawyer_memory)
            suspector = agent.init_chatbot(prompt_suspector, model_name, agent.party_memory)

            if self.conversation_round == 1:
                # ==== baseline 流程，保持你原来的逻辑 ====
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
                db.insert_or_update_record_to_direct(id, chat_history, table)

            else:
                # ==== GrIPS 在线更新版 ====
                chat_history_index = []

                # 初始寒暄
                initial_suspector_input = "您好，我是律师。请放心，我会尽全力维护您的权益。"
                suspect_output = "您好，我是当事人。我需要您的帮助。"
                chat_history_index.append({"律师": initial_suspector_input, "当事人": suspect_output})

                max_rounds = 20

                for round_num in range(1, max_rounds + 1):
                    if round_num % 2 == 0:
                        # 偶数轮：当事人回答
                        suspect_output = suspector({"question": lawyer_output})["text"]
                        print(f"当事人: {suspect_output}")
                        chat_history_index.append({"律师": lawyer_output, "当事人": suspect_output})

                        # —— GrIPS：在当前对话历史基础上更新 “策略文本” ——
                        if current_policy_text is not None:
                            current_policy_text = grips_update_policy(
                                agent,
                                current_policy_text,
                                chat_history_index,
                                model_name
                            )

                            # 基于新的策略文本 + 当前被告人，生成更新后的律师 prompt
                            updated_prompt_lawyer = agent.generateprompt_grips(
                                personality,
                                current_policy_text,
                                defendant
                            )

                            # 用新的 prompt 重新初始化律师（共享当前 memory）
                            lawyer = agent.init_chatbot(updated_prompt_lawyer, model_name, agent.lawyer_memory)

                    else:
                        # 奇数轮：律师提问（使用当前 lawyer 这个 chatbot）
                        lawyer_output = lawyer({"question": suspect_output})["text"]
                        print(f"律师: {lawyer_output}")

                # 把本 case 的对话写入数据库
                chat_history_index = str(chat_history_index)
                db.insert_or_update_record_to_direct(id, chat_history_index, table)

        # === 所有 case 跑完后，保存最终学出来的策略文本 ===
        if self.conversation_round != 1 and policy not in ["normal", "normal_blank"] and current_policy_text:
            final_path = f"prompt/policy/{policy}_grips_final.txt"
            latest_path = f"prompt/policy/{policy}_grips_latest.txt"
            with open(final_path, "w", encoding="utf-8") as f:
                f.write(current_policy_text)
            with open(latest_path, "w", encoding="utf-8") as f:
                f.write(current_policy_text)

        print(current_policy_text)

if __name__ == '__main__':
    agent = Agent()

    # 示例：只跑带 GrIPS 的 Adversarial 策略
    thread1 = threading.Thread(
        target=Conversation(0).run_case,
        args=("extravert", "Adversarial", "adversarial_grips",492, "defendant")
    )
    thread2 = threading.Thread(
        target=Conversation(0).run_case,
        args=("extravert", "Accurate-evidence", "accurate_grips",600, "defendant")
    )
    thread3 = threading.Thread(
        target=Conversation(0).run_case,
        args=("extravert", "Multi-angle", "multi_angle_grips",600, "defendant")
    )
    thread4 = threading.Thread(
        target=Conversation(0).run_case,
        args=("extravert", "Comprehensive", "comprehensive_grips",600, "defendant")
    )

    # thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()

    # thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()

    print("Done")


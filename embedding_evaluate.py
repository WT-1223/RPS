from DB import DB
import json
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
import threading
import os
os.environ['KERAS_BACKEND'] = 'mxnet'

class Embedding_Evaluation:
    def __init__(self, table_name, start):
        self.table_name = table_name
        self.start = start

    def run_evaluate(self):
        model = SentenceTransformer("paraphrase-xlm-r-multilingual-v1")
        db = DB(host='localhost', user='root', password='020201', database='deception_detection')
        messages_direct = db.fetch_records("*", self.table_name)
        for message in tqdm(messages_direct[self.start:], desc='Processing messages'):
            try:
                target_data = []
                id = message["id"]
                chatHistory = json.loads(message["chat_history"].replace("'", '"'))
                print(chatHistory)
                extract_content = db.fetch_record_by_id("Key_Information", "law_data_total_detail", id)
                extract_content = json.loads(extract_content[0]["Key_Information"])
                if "cases" in extract_content.keys():
                    for case in extract_content["cases"]:
                        case_time = f'时间:{case["时间"]}'
                        address = f'地点:{case["地点"]}'
                        role = f'人物:{case["人物"]}'
                        event = case["事件"]
                        target_data.extend([case_time, address, role])
                        target_data.extend(event)
                    target_data = [f"{value}" for value in target_data]
                    print("target_data:", target_data)
                else:
                    case_time = f'时间:{extract_content["时间"]}'
                    address = f'地点:{extract_content["地点"]}'
                    role = f'人物:{extract_content["人物"]}'
                    event = extract_content["事件"]
                    target_data.extend([case_time, address, role])
                    target_data.extend(event)
                    target_data = [f"{value}" for value in target_data]
                    print(target_data)
                embeddings_total = model.encode(target_data)
                # if chatHistory[0] is not list:
                all_info = {}
                similarity = []
                for index in range(1, 11):
                    user_ans = []
                    print("\n" + '*' * 20)
                    print(f"第{index}轮对话\n")

                    # Build conversation content
                    conversation = ""

                    for item in chatHistory[0:index + 1]:
                        conversation += f"律师:{item['律师']}\n当事人:{item['当事人']}\n"
                    user_ans.append(chatHistory[index]['当事人'])
                    embeddings_ans = model.encode(user_ans)
                    similarities = model.similarity(embeddings_ans, embeddings_total)
                    similarity.append(similarities.squeeze(0).tolist())
                    index_max = np.max(np.array(similarity), axis=0)
                    all_info[f"result"] = np.average(index_max)
                    all_info["similarity"] = similarity
                    # result_index.append(np.average(index_max))
                    print(all_info[f"result"])
                    # all_info.append(result_index)
                    db.add_column_and_update(self.table_name, f"对话轮数{index}", json.dumps(all_info), "json", id)
                    # else:
                    #     print(2)
                    #     result_info = []
                    #     all_info = []
                    #     for chat in chatHistory:
                    #         result_index = []
                    #         similarity = []
                    #         for index in range(1, 11):
                    #             user_ans = []
                    #             print("\n" + '*' * 20)
                    #             print(f"第{index}轮对话\n")
                    #
                    #             # Build conversation content
                    #             conversation = ""
                    #
                    #             for item in chat[0:index + 1]:
                    #                 conversation += f"律师:{item['律师']}\n当事人:{item['当事人']}\n"
                    #             user_ans.append(chat[index]['当事人'])
                    #             embeddings_ans = model.encode(user_ans)
                    #             similarities = model.similarity(embeddings_ans, embeddings_total)
                    #             similarity.append(similarities.squeeze(0).tolist())
                    #             index_max = np.max(np.array(similarity), axis=0)
                    #             result_index.append(np.average(index_max))
                    #         all_info.append(result_index)
                    #         result_info = np.max(np.array(all_info), axis=0)
                    #         print(result_info)
                    #     for index, value in enumerate(result_info):
                    #         db.add_column_and_update(self.table_name, f"对话轮数{index+1}", json.dumps({'result':value}), "json", id)

            except Exception as e:
                print(e)


if __name__ == '__main__':
    thread1 = threading.Thread(target=Embedding_Evaluation("accurate_detail", 500).run_evaluate)
    thread1.start()
    thread1.join()
    #
    thread2 = threading.Thread(target=Embedding_Evaluation("adversarial_detail", 500).run_evaluate)
    thread2.start()
    thread2.join()
    #
    thread3 = threading.Thread(target=Embedding_Evaluation("comprehensive_detail", 500).run_evaluate)
    thread3.start()
    thread3.join()

    thread4 = threading.Thread(target=Embedding_Evaluation("multi_angle_detail", 500).run_evaluate)
    thread4.start()
    thread4.join()

    # thread5 = threading.Thread(target=Embedding_Evaluation("normal_blank_detail", 725).run_evaluate)
    # thread5.start()
    # thread5.join()

    # thread6 = threading.Thread(target=Embedding_Evaluation("dqn_data_new",0).run_evaluate)
    # thread6.start()
    # thread6.join()

    # thread7 = threading.Thread(target=Embedding_Evaluation("closed", 0).run_evaluate)
    # thread7.start()
    # thread7.join()

    # thread8 = threading.Thread(target=Embedding_Evaluation("open",0).run_evaluate)
    # thread8.start()
    # thread8.join()

    # thread9 = threading.Thread(target=Embedding_Evaluation("prompt",0).run_evaluate)
    # thread9.start()
    # thread9.join()

    print("Done")
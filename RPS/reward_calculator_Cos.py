from sentence_transformers import SentenceTransformer, util
import json
import torch,gc

class RewardCalculator:
    def __init__(self, agent, db_connection):
        self.agent = agent
        self.db = db_connection

    def calculate_reward(self, chat_history, case_id):
        gc.collect()
        torch.cuda.empty_cache()
        model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
        extract_content = self.db.fetch_record_by_id("extraction_content", "law_data_total_initial", case_id)
        extract_content = json.loads(extract_content[0]["extraction_content"])
        target_data = []
        if "cases" in extract_content.keys():
            for case in extract_content["cases"]:
                case_time = case["时间"]
                address = case["地点"]
                role =  case["人物"]
                event = case["事件"]
                if type(case_time) is str:
                    target_data.append(case_time)
                else:
                    target_data.extend(case_time)
                if type(address) is str:
                    target_data.append(address)
                else:
                    target_data.extend(address)
                if type(role) is str:
                    target_data.append(role)
                else:
                    target_data.extend(role)
                if type(event) is str:
                    target_data.append(event)
                if isinstance(event, list) and all(isinstance(i, dict) for i in event):
                    target_data.extend([value for d in event for value in d.values()])
                else:
                    target_data.extend(event)


        else:
            case_time = extract_content["时间"]
            address = extract_content["地点"]
            role = extract_content["人物"]
            event = extract_content["事件"]
            if type(case_time) is str:
                target_data.append(case_time)
            else:
                target_data.extend(case_time)
            if type(address) is str:
                target_data.append(address)
            else:
                target_data.extend(address)
            if type(role) is str:
                target_data.append(role)
            else:
                target_data.extend(role)
            if type(event) is str:
                target_data.append(event)
            if isinstance(event, list) and all(isinstance(i, dict) for i in event):
                target_data.extend([value for d in event for value in d.values()])
            else:
                target_data.extend(event)

        target_data = [f"{value}" for value in target_data]
        embeddings_total = model.encode(target_data)
        user_ans = [chat_history[-1]['当事人']]
        embeddings_ans = model.encode(user_ans)
        similarities = util.cos_sim(embeddings_ans, embeddings_total)
        return similarities
    
import json

# from networkx.algorithms.bipartite import color
# from scipy.special import label

from DB import DB
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import math
import os
from matplotlib import mathtext

os.environ["PATH"] = os.pathsep+r"C:\texlive\2025\bin\windows"

rcParams['text.usetex'] = True

class PlotterVariance:
    def get_data(self,table_name,start,end):
        db = DB(host='localhost', user='root', password='020201', database='deception_detection')
        message = db.fetch_records('*', table_name)
        all_f1 = []
        for item in message:
            if item['id'] in range(start, end):
                round_f1 = []
                for i in range(1, 11):
                    # print(item[f"对话轮数{i}"])
                    f1_score = json.loads(item["对话轮数{}".format(i)])["result"]
                    round_f1.append(f1_score)
                all_f1.append(round_f1)
        range_mean = np.mean(np.array(all_f1),axis=0)
        range_std = np.std(np.array(all_f1), axis=0) / math.sqrt(end-start)
        range_std = range_std * 1.96
        return range_mean, range_std

    def get_data_dqn(self,table_name):
        db = DB(host='localhost', user='root', password='020201', database='deception_detection')
        message = db.fetch_records('*', table_name)
        all_f1 = []
        for item in message:
            if item['id'] in range(501, 1001):
                round_f1 = []
                for i in range(1, 11):
                    # print(item[f"对话轮数{i}"])
                    f1_score = json.loads(item["对话轮数{}".format(i)])["result"]
                    round_f1.append(f1_score)
                all_f1.append(round_f1)
        range_mean = np.mean(np.array(all_f1),axis=0)
        range_std = np.std(np.array(all_f1),axis=0) / math.sqrt(500)
        range_std = range_std * 1.96
        return range_mean, range_std

    def plot_variance(self,direct, se_direct, bluff, se_bluff, restate,  se_restate,choice,se_choice,normal,se_normal,dqn,se_dqn,llm,se_llm,grips,se_grips,rlprompt,se_rlprompt):
        plt.figure(figsize=(16, 10))
        x_values = np.arange(1, len(dqn) + 1)
        plt.errorbar(x_values, normal, yerr=se_normal, label='Normal', color='black', linestyle='solid', linewidth=3, elinewidth=3, marker='o')
        # plt.errorbar(x_values, choice, yerr=se_choice, label='Exploratory Information Gathering', color='c', linestyle='dashed', linewidth=3, elinewidth=3, marker='h')
        # plt.errorbar(x_values, direct, yerr=se_direct, label='Precise Evidence Confirmation', color='m', linestyle='-', linewidth=3, elinewidth=3, marker='v')
        plt.errorbar(x_values, bluff, yerr=se_bluff, label='Corroborative Detail Verification', color='green', linestyle='-.', linewidth=3, elinewidth=3, marker='s')
        # plt.errorbar(x_values, restate, yerr=se_restate, label='Confrontational Disclosure Questioning', color='red', linestyle=':', linewidth=3, elinewidth=3, marker='p')
        plt.errorbar(x_values, llm, yerr=se_llm, label='Adaptive Generative Questioning', color=(0.2,0.5,0.7), linestyle='--', linewidth=3, elinewidth=3, marker='x')
        plt.errorbar(x_values, grips, yerr=se_grips, label='GRIPS-Corroborative Detail Verification', color='orange', linestyle='-.', linewidth=3, elinewidth=3, marker='D')
        plt.errorbar(x_values, rlprompt, yerr=se_rlprompt, label='RLPrompt', color='purple', linestyle='--', linewidth=3, elinewidth=3, marker='p')
        plt.errorbar(x_values,dqn,yerr=se_dqn,label='RPS', color='blue', linestyle='--', linewidth=3, elinewidth=3,marker='o')
        # plt.title('IELegal-base Results', fontsize=20)
        plt.xlabel('$t$', fontsize=24)
        plt.ylabel('$d(\hat{I}_{t-1}, I)$', fontsize=24)
        plt.xticks(x_values, fontsize=24)
        plt.yticks(fontsize=24)
        plt.legend(fontsize=16)
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    plotter = PlotterVariance()

    # detail
    # dqn,se_dqn = plotter.get_data("dqn_data_new_llm_prompt_detail_fix")
    # average_direct, std_err_direct = plotter.get_data("accurate_detail_copy1")
    # average_bluff, std_err_bluff = plotter.get_data("multi_angle_detail_copy1")
    # average_restate, std_err_restate = plotter.get_data("adversarial_detail_copy1")
    # average_choice, std_err_choice = plotter.get_data("comprehensive_detail_copy1")
    # average_normal, std_err_normal = plotter.get_data("normal_blank_detail")
    # llm, std_err_llm = plotter.get_data("llm_detail_fix")
    # plotter.plot_variance(average_direct, std_err_direct,average_bluff,std_err_bluff,average_restate, std_err_restate,average_choice, std_err_choice,average_normal, std_err_normal,dqn,se_dqn,llm,std_err_llm)

    # initial
    dqn, se_dqn = plotter.get_data_dqn("dqn_data_new_llm_prompt")
    average_direct, std_err_direct = plotter.get_data("accurate",501,601)
    average_bluff, std_err_bluff = plotter.get_data("multi_angle",501,701)
    average_restate, std_err_restate = plotter.get_data("adversarial",1,501)
    average_choice, std_err_choice = plotter.get_data("comprehensive",501,701)
    average_normal, std_err_normal = plotter.get_data("normal_blank",501,1001)
    llm,std_err_llm = plotter.get_data("llm_fix",501,1001)
    grips, std_err_grips = plotter.get_data("multi_angle_grips",501,701)
    rlprompt,std_err_rlpprompt = plotter.get_data("dialogue_history_rlprompt",501,1001)
    plotter.plot_variance(average_direct, std_err_direct, average_bluff, std_err_bluff, average_restate, std_err_restate, average_choice, std_err_choice, average_normal, std_err_normal, dqn, se_dqn,llm,std_err_llm,grips,std_err_grips,rlprompt,std_err_rlpprompt)

    # user_honest
    # dqn, se_dqn = plotter.get_data("dqn_data_new")
    # average_direct, std_err_direct = plotter.get_data("accurate_user_honest")
    # average_bluff, std_err_bluff = plotter.get_data("multi_angle_user_honest")
    # average_restate, std_err_restate = plotter.get_data("adversarial_user_honest")
    # average_choice, std_err_choice = plotter.get_data("comprehensive_user_honest")
    # average_normal, std_err_normal = plotter.get_data("normal_blank")
    # plotter.plot_variance(average_direct, std_err_direct, average_bluff, std_err_bluff, average_restate, std_err_restate, average_choice, std_err_choice, average_normal, std_err_normal, dqn, se_dqn)

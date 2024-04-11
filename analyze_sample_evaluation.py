import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
from matplotlib import rcParams
from typing import *
import pandas as pd
import seaborn as sns
import math
import os

config = {
            "font.family": 'Times New Roman',
            "font.size": 12,
            "mathtext.fontset": 'stix',
            "font.serif": ['SimSun'],
            'axes.unicode_minus': False
         }
rcParams.update(config)

plusnoise = 1
train_corre = 0
reg = 0
beta= 0
finetune = 0
radius_type = "Llower"

if __name__ == "__main__":

    #load
    total_path = "result_sampling_evaluation/"
    method = "ngmv2"
    p=0.9

    for epoch_num in [3]:
        for ori_sigma in [0.5]:

            input_number=0
            pro_sum=0

            for i in range(0, 100):
                if  train_corre != 0:

                    #optimization result
                    if ori_sigma == 0.5:
                        if reg == 1 and train_corre == 0.01:
                            corre_num = 0.016
                            k = 0.7017 / 0.5
                        elif reg == 1 and train_corre == 0.005:
                            corre_num = 0.022
                            k = 0.7008 / 0.5
                        elif reg == 1 and train_corre == 0.015:
                            corre_num = 0.021
                            k = 0.6986 / 0.5
                        elif reg == 1 and train_corre == 0.02:
                            corre_num = 0.0195
                            k = 0.699 / 0.5
                        else:
                            corre_num = 0.017
                            k = 0.699 / 0.5
                    file_path = "/sigmaori" + str(
                                    ori_sigma) + '_correnum' + str(
                                    corre_num) + '_p' + str(p) + "_plusnoise" + str(plusnoise) + "_reg" + str(
                                    reg) + "_beta" + str(beta) + "_traincorre" + str(train_corre) + "_finetune" + str(
                                    finetune) + "_epoch" + str(epoch_num) + "_SCR_" + radius_type

                else:
                    k = 1
                    corre_num = 0
                    file_path = "/sigmaori" + str(
                        ori_sigma) + '_correnum' + str(
                        corre_num) + '_p' + str(p) + "_plusnoise" + str(plusnoise) + "_reg" + str(
                        reg) + "_beta" + str(beta) + "_traincorre" + str(train_corre) + "_finetune" + str(
                        finetune) + "_epoch" + str(epoch_num) + "_SCR_" + radius_type

                file_name = total_path + method + file_path + "/" + str(i)

                df = pd.read_csv(file_name, delimiter="\t")

                pro_sum += df["correct"].mean()
                input_number += 1

            print(pro_sum / input_number)



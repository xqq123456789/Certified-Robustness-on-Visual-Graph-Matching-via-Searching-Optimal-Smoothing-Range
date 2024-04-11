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

plusnoise = 0
reg = 0
train_corre = 0
beta= 0
finetune = 0
epoch_num = 3
p = 0.9

if __name__ == "__main__":

    method_name = "ngmv2/voc"

    #optimization result
    if method_name == "ngmv2":
        corre_num = 0.016
    if method_name == "pca":
        corre_num = 0.025
    if method_name == "cie":
        corre_num = 0.029
    if method_name == "gmn":
        corre_num = 0.027

    data_file_name = "marginal radii_under_keypoint/"
    result_SCR = np.zeros([1, 3])
    result_RS = np.zeros([1, 3])
    i=0
    for sigma in [0.5]:
        j = 0
        for L_type in ["Lmax", "Lvolume", "Llower"]:

            RS_file = data_file_name + method_name + "/sigmaori" + str(sigma) + '_p' + str(p) + "_plusnoise" + str(
                plusnoise) + "_reg" + str(reg) + "_beta" + str(beta) + "_traincorre" + str(
                train_corre) + "_finetune" + str(finetune) + "_epoch" + str(epoch_num) + "_n1000_RS_" + L_type

            df_RS = pd.read_csv(RS_file, delimiter="\t")
            result_RS[i][j] = 4*(df_RS["correct"] * (df_RS["radius_A"])* (df_RS["radius_B"])).mean()
            j+=1
        i+=1
        
    print(result_RS)
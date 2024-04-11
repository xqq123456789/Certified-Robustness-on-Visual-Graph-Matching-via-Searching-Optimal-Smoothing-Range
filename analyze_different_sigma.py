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

method="ngmv2/voc"
method_name="NGMv2"
L_type="Llower"

file_name ="figure"
if os.path.exists(file_name)==False:
    os.mkdir(file_name)
if os.path.exists(file_name+"/"+method)==False:
    os.mkdir(file_name+"/"+method)

class Accuracy(object):
    def at_radii(self, radii: np.ndarray):
        raise NotImplementedError()

class ApproximateAccuracy(Accuracy):
    def __init__(self, data_file_path: str):
        self.data_file_path = data_file_path

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):
        return (df["correct"] & (df["radius_A"] >= radius)).mean()

class Line(object):
    def __init__(self, quantity: Accuracy, legend: str, plot_fmt: str = "", scale_x: float = 1):
        self.quantity = quantity
        self.legend = legend
        self.plot_fmt = plot_fmt
        self.scale_x = scale_x

def plot_certified_accuracy(outfile: str, title: str, max_radius: float,
                            lines: List[Line], radius_step: float = 0.01) -> None:
    print(outfile)
    radii = np.arange(0, max_radius + radius_step, radius_step)
    linestyle_str = ['dashed', 'dashed', 'dashed', 'dashed', 'dashed', "dashed"]
    color_str = ['r', 'y', 'g', 'b', 'c', 'm', 'gold']

    plt.figure()

    item = 0
    for line in lines:
        plt.rc('font', family='Times New Roman')
        plt.plot(radii * line.scale_x * 2, line.quantity.at_radii(radii), line.plot_fmt, linestyle=linestyle_str[item],
                 color=color_str[item], linewidth=2)
        item += 1

    plt.ylim((0, 0.7))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=20)
    L_output = "radius"

    plt.rc('font', family='Times New Roman')
    plt.xlabel(L_output, fontsize=26)
    plt.rc('font', family='Times New Roman')
    plt.ylabel("certified accuracy", fontsize=26)
    plt.legend([method.legend for method in lines], loc='upper right', fontsize=14)
    plt.tight_layout()
    plt.title(title, fontsize=24)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=600)
    plt.close()

def markdown_table_certified_accuracy(outfile: str, radius_start: float, radius_stop: float, radius_step: float,
                                      methods: List[Line]):
    radii = np.arange(radius_start, radius_stop + radius_step, radius_step)
    accuracies = np.zeros((len(methods), len(radii)))
    for i, method in enumerate(methods):
        accuracies[i, :] = method.quantity.at_radii(radii)

    f = open(outfile, 'w')
    f.write("|  | ")
    for radius in radii:
        f.write("r = {:.3} |".format(radius))
    f.write("\n")

    f.write("| --- | ")
    for i in range(len(radii)):
        f.write(" --- |")
    f.write("\n")

    for i, method in enumerate(methods):
        f.write("<b> {} </b>| ".format(method.legend))
        for j, radius in enumerate(radii):
            if i == accuracies[:, j].argmax():
                txt = "{:.3f}<b>*</b> |".format(accuracies[i, j])
            else:
                txt = "{:.3f} |".format(accuracies[i, j])
            f.write(txt)
        f.write("\n")
    f.close()


if __name__ == "__main__":

    data_file_name = "marginal radii_under_keypoint/"
    p=0.9
    sigma = 0.5
    for L_type in ["Llower","Lmax","Lvolume"]:

        if L_type == "Lmax":
            length = 10
        if L_type == "Lvolume":
            length = 16
        if L_type == "Llower":
            length = 8
        plot_certified_accuracy(
            file_name + "/" + method + "/sigma"+str(sigma)+"_different_sigma_" + L_type, method_name+ " $s=$" + str(0.9),length, [

                Line(ApproximateAccuracy(
                    data_file_name + method + "/sigmaori" + str(0.5) + '_correnum0.016_p' + str(
                        p) + "_plusnoise1" + "_reg1" + "_beta100" + "_traincorre" + str(
                        0.01) + "_finetune0" + "_epoch" + str(3) + "_SCR_" + L_type), "$\sigma=0.5$"),

                Line(ApproximateAccuracy(
                    data_file_name + method + "/sigmaori" + str(1) + '_correnum0.013_p' + str(
                        p) + "_plusnoise1" + "_reg1" + "_beta100" + "_traincorre" + str(
                        0.01) + "_finetune0" + "_epoch" + str(3) + "_SCR_" + L_type), "$\sigma=1$"),

                Line(ApproximateAccuracy(
                    data_file_name + method + "/sigmaori" + str(1.5) + '_correnum0.013_p' + str(
                        p) + "_plusnoise1" + "_reg1" + "_beta100" + "_traincorre" + str(
                        0.01) + "_finetune0" + "_epoch" + str(3) + "_SCR_" + L_type), "$\sigma=1.5$"),

                Line(ApproximateAccuracy(
                    data_file_name + method + "/sigmaori" + str(2) + '_correnum0.013_p' + str(
                        p) + "_plusnoise1" + "_reg1" + "_beta100" + "_traincorre" + str(
                        0.01) + "_finetune0" + "_epoch" + str(3) + "_SCR_" + L_type), "$\sigma=2$"),

            ])


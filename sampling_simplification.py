import time
import datetime
from pathlib import Path
import xlwt

from src.dataset.data_loader import GMDataset, get_dataloader
from src.evaluation_metric import *
from src.parallel import DataParallel
from src.utils.model_sl import load_model
from src.utils.data_to_cuda import data_to_cuda
from src.utils.timer import Timer

from src.utils.config import cfg
from pygmtools.benchmark import Benchmark

from torch.autograd import Variable
from torch.distributions.normal import Normal
from core_match_less_corre_915 import Smooth
import numpy as np
import os
import argparse
from scipy.stats import norm
import pandas as pd

method="ngmv2/"
input_file_name = "marginal radii_under_keypoint"
output_file_name = "result_sampling_evaluation/"
radius_type = "Llower"

plusnoise = 1
train_corre = 0.01
reg = 0
beta= 0
finetune = 0

if os.path.exists(output_file_name) == False:
    os.mkdir(output_file_name)

if os.path.exists(output_file_name + method) == False:
    os.mkdir(output_file_name + method)

def corr(length: int, sigma: float, k: float, corre_num: float):
    covar_result = torch.eye(length * 2).cuda()
    for i in range(length - 1):
        covar_result[i][i + 1] = corre_num
        covar_result[i + 1][i] = corre_num
        covar_result[i + length][i + length + 1] = corre_num
        covar_result[i + length + 1][i + length] = corre_num
    covar_result = covar_result * sigma * k
    Sigma = torch.mm(covar_result, covar_result)

    return covar_result, Sigma

def belong_space_SCR(Bmatrix_A_inver, Bmatrix_B_inver, Sigma_A_inver, Sigma_B_inver, delta_A, delta_B, p):

    delta_A_trans = delta_A.T
    delta_B_trans = delta_B.T
    numerator_A = torch.matmul(torch.matmul(delta_A, Sigma_A_inver), delta_A_trans)
    numerator_B = torch.matmul(torch.matmul(delta_B, Sigma_B_inver), delta_B_trans)
    numerator = numerator_A + numerator_B
    denominator_A = torch.matmul(delta_A, Bmatrix_A_inver)
    denominator_B = torch.matmul(delta_B, Bmatrix_B_inver)
    denominator = torch.norm(denominator_A+denominator_B)

    if numerator < 0:
        return 1
    elif numerator/denominator < norm.ppf(p):
        return 1
    else:
        return 0

def belong_space_RS(sigma, delta_A, delta_B, p):

    delta_A_trans = delta_A.T
    delta_B_trans = delta_B.T
    numerator = (torch.matmul(delta_A, delta_A_trans) + torch.matmul(delta_B, delta_B_trans))/(sigma*sigma)
    denominator = torch.norm(delta_A/sigma + delta_B/sigma)

    if numerator < 0:
        return 1
    elif numerator/denominator < norm.ppf(p):

        return 1
    else:
        return 0


if __name__ == '__main__':
    from src.utils.dup_stdout_manager import DupStdoutFileManager
    from src.utils.parse_args import parse_args

    args = parse_args('Deep learning of graph matching evaluation code.')

    import importlib
    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    torch.manual_seed(cfg.RANDOM_SEED)

    ds_dict = cfg[cfg.DATASET_FULL_NAME] if ('DATASET_FULL_NAME' in cfg) and (cfg.DATASET_FULL_NAME in cfg) else {}
    benchmark = Benchmark(name=cfg.DATASET_FULL_NAME,
                          sets='test',
                          problem=cfg.PROBLEM.TYPE,
                          obj_resize=cfg.PROBLEM.RESCALE,
                          filter=cfg.PROBLEM.FILTER,
                          **ds_dict)

    cls = None if cfg.EVAL.CLASS in ['none', 'all'] else cfg.EVAL.CLASS
    if cls is None:
        clss = benchmark.classes
    else:
        clss = [cls]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net()
    model = model.to(device)
    model = DataParallel(model, device_ids=cfg.GPUS)

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)
    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    wb = xlwt.Workbook()
    ws = wb.add_sheet('epoch{}'.format(cfg.EVAL.EPOCH))

    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / ('eval_log_' + now_time + '.log'))) as _:

        for epoch_num in [3]:

            if epoch_num<10:
                model_path = cfg.PRETRAINED_PATH[:-4]+str(epoch_num)+".pt"
            else:
                model_path = cfg.PRETRAINED_PATH[:-5]+str(epoch_num)+".pt"
            print('Loading model parameters from {}'.format(model_path))

            load_model(model, model_path)
            for p in [0.9]:

                for ori_sigma in [0.5]:

                    if train_corre == 0:  # RS

                        corre_num = 0
                        k = 1
                        df = pd.read_csv(
                            input_file_name + "/" + method + "/voc/sigmaori" + str(
                                ori_sigma) + '_p' + str(p) + "_plusnoise" + str(plusnoise) + "_reg" + str(
                                reg) + "_beta" + str(beta) + "_traincorre" + str(train_corre) + "_finetune" + str(
                                finetune) + "_epoch" + str(epoch_num) + "_RS_" + radius_type, delimiter="\t")
                        output_path = output_file_name + "/" + method + "/sigmaori" + str(
                            ori_sigma) + '_correnum' + str(
                            corre_num) + '_p' + str(p) + "_plusnoise" + str(plusnoise) + "_reg" + str(
                            reg) + "_beta" + str(beta) + "_traincorre" + str(train_corre) + "_finetune" + str(
                            finetune) + "_epoch" + str(epoch_num) + "_SCR_" + radius_type

                    else:  # SCR-GM

                        #optimization result
                        if method == "ngmv2":
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
                        df = pd.read_csv(input_file_name + "/" + method + "/voc/sigmaori" + str(
                            ori_sigma) + '_correnum' + str(
                            corre_num) + '_p' + str(p) + "_plusnoise" + str(plusnoise) + "_reg" + str(
                            reg) + "_beta" + str(beta) + "_traincorre" + str(train_corre) + "_finetune" + str(
                            finetune) + "_epoch" + str(epoch_num) + "_SCR_" + radius_type, delimiter="\t")
                        output_path = output_file_name + "/" + method + "/sigmaori" + str(
                            ori_sigma) + '_correnum' + str(
                            corre_num) + '_p' + str(p) + "_plusnoise" + str(plusnoise) + "_reg" + str(
                            reg) + "_beta" + str(beta) + "_traincorre" + str(train_corre) + "_finetune" + str(
                            finetune) + "_epoch" + str(epoch_num) + "_SCR_" + radius_type

                    if os.path.exists(output_path) == False:
                        os.mkdir(output_path)

                    model.eval()
                    dataloaders = []

                    # create the smoothed classifier g
                    smoothed_classifier = Smooth(model)

                    for cls in clss:
                        image_dataset = GMDataset(name=cfg.DATASET_FULL_NAME,
                                                  bm=benchmark,
                                                  problem=cfg.PROBLEM.TYPE,
                                                  length=cfg.EVAL.SAMPLES,
                                                  cls=cls,
                                                  using_all_graphs=cfg.PROBLEM.TEST_ALL_GRAPHS)
                        torch.manual_seed(cfg.RANDOM_SEED)
                        dataloader = get_dataloader(image_dataset, shuffle=True)
                        dataloaders.append(dataloader)

                    timer = Timer()
                    number = 0

                    for i, cls in enumerate(clss):

                        running_since = time.time()
                        iter_num = 0

                        for inputs in dataloaders[i]:
                            if iter_num >= cfg.EVAL.SAMPLES / inputs['batch_size']:
                                break
                            if model.module.device != torch.device('cpu'):
                                inputs = data_to_cuda(inputs)

                            batch_num = inputs['batch_size']

                            iter_num = iter_num + 1

                            with torch.set_grad_enabled(False):

                                if df["pABar_item"][number] > 0.5:

                                    point = inputs['Ps']

                                    if train_corre != 0:  # SCR

                                        f = open(output_path + "/" + str(number), 'w')

                                        print("number\tpABar_item\tradius_A\tradius_B\tnorm_A\tnorm_B\tcorrect", file=f,
                                              flush=True)

                                        length_A = point[0].size()[1]
                                        length_B = point[1].size()[1]
                                        B_A, Sigma_A = corr(length_A, ori_sigma, k, corre_num)
                                        B_B, Sigma_B = corr(length_B, ori_sigma, k, corre_num)


                                        radius_A = df["radius_A"][number]
                                        radius_B = df["radius_B"][number]

                                        Sigma_A_inver = torch.inverse(Sigma_A)
                                        Sigma_B_inver = torch.inverse(Sigma_B)
                                        B_A_inver = torch.inverse(B_A)
                                        B_B_inver = torch.inverse(B_B)

                                        sample_num = 0

                                        for scale_A in range(0, 20, 2):
                                            for scale_B in range(0, 20, 2):
                                                for i in range(10):

                                                    # uniform
                                                    delta_A = torch.from_numpy(
                                                        np.random.uniform(-ori_sigma,
                                                                          ori_sigma, [1,length_A * 2])).float().to(
                                                        device)
                                                    delta_B = torch.from_numpy(
                                                        np.random.uniform(-ori_sigma,
                                                                          ori_sigma,
                                                                          [1, length_B * 2])).float().to(
                                                        device)

                                                    #gaussian
                                                    '''
                                                    delta_A = torch.from_numpy(np.random.normal(0,ori_sigma/2,length_A * 2)).float().to(device)
                                                    delta_B = torch.from_numpy(np.random.normal(0, ori_sigma/2, length_B * 2)).float().to(
                                                        device)
                                                    '''

                                                    if belong_space_SCR(B_A_inver, B_B_inver, Sigma_A_inver,
                                                                        Sigma_B_inver, delta_A, delta_B,
                                                                        df["pABar_item"][number]):

                                                        print("{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{}".format(
                                                            sample_num, df["pABar_item"][number],
                                                            radius_A, radius_B, torch.norm(delta_A).float(),
                                                            torch.norm(delta_B).float(), "1"),
                                                              file=f, flush=True)

                                                        sample_num += 1
                                                    else:
                                                        print("{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{}".format(
                                                            sample_num, df["pABar_item"][number],
                                                            radius_A, radius_B, torch.norm(delta_A).float(),
                                                            torch.norm(delta_B).float(), "0"),
                                                            file=f, flush=True)
                                                        sample_num += 1

                                    else:  # RS

                                        f = open(output_path + "/" + str(number), 'w')
                                        print(
                                            "number\tpABar_item\tradius_A\tradius_B\tnorm_A\tnorm_B\tcorrect",
                                            file=f, flush=True)
                                        radius_A = df["radius_A"][number]
                                        radius_B = df["radius_B"][number]
                                        length_A = point[0].size()[1]
                                        length_B = point[1].size()[1]

                                        B_A = torch.eye(length_A * 2).cuda() * ori_sigma
                                        B_B = torch.eye(length_B * 2).cuda() * ori_sigma

                                        Sigma_A = torch.matmul(B_A, B_A)
                                        Sigma_B = torch.matmul(B_B, B_B)

                                        Sigma_A_inver = torch.inverse(Sigma_A)
                                        Sigma_B_inver = torch.inverse(Sigma_B)
                                        B_A_inver = torch.inverse(B_A)
                                        B_B_inver = torch.inverse(B_B)

                                        sample_num = 0
                                        for scale_A in range(0, 20, 4):
                                            for scale_B in range(0, 20, 4):
                                                for i in range(1):

                                                    # uniform

                                                    delta_A = torch.from_numpy(
                                                        np.random.uniform(-ori_sigma,
                                                                          ori_sigma, [1, length_A * 2])).float().to(
                                                        device)
                                                    delta_B = torch.from_numpy(
                                                        np.random.uniform(-ori_sigma,
                                                                          ori_sigma,
                                                                          [1, length_B * 2])).float().to(
                                                        device)

                                                    '''
                                                    #gaussian 
                                                    delta_A = torch.from_numpy(
                                                        np.random.normal(0, ori_sigma, length_A * 2)).float().to(device)
                                                    delta_B = torch.from_numpy(
                                                        np.random.normal(0, ori_sigma, length_B * 2)).float().to(
                                                        device)
                                                    '''

                                                    if belong_space_RS(ori_sigma,delta_A, delta_B,
                                                                                            df["pABar_item"][number]):

                                                        print("{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{}".format(
                                                            sample_num, df["pABar_item"][number],
                                                            radius_A, radius_B, torch.norm(delta_A).float(),
                                                            torch.norm(delta_B).float(), "1"),
                                                              file=f, flush=True)

                                                        sample_num += 1
                                                    else:
                                                        print("{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{}".format(
                                                            sample_num, df["pABar_item"][number],
                                                            radius_A, radius_B, torch.norm(delta_A).float(),
                                                            torch.norm(delta_B).float(), "0"),
                                                            file=f, flush=True)
                                                        sample_num += 1

                            number += 1

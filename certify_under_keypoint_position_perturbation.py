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
from core_match_less_corre import Smooth
import numpy as np
import os
import argparse

gm_method = "ngmv2"
method = gm_method+"/voc"

file_name = "marginal_radii_under_keypoint"
if os.path.exists(file_name)==False:
    os.mkdir(file_name)
if os.path.exists(file_name+"/"+method)==False:
    os.mkdir(file_name+"/"+method)
plusnoise = 1
reg = 1
train_corre = 0.01
beta= 100
n_num = 4
finetune = 0
n = 1000

if __name__ == '__main__':
    from src.utils.dup_stdout_manager import DupStdoutFileManager
    from src.utils.parse_args import parse_args
    from src.utils.print_easydict import print_easydict

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

                    #optimization result
                    if gm_method == "ngmv2":
                        if ori_sigma==0.5:
                            if reg==1 and train_corre==0.01:
                                corre_num = 0.016
                                k = 0.7017 / 0.5
                            elif reg==1 and train_corre==0.005:
                                corre_num = 0.022
                                k = 0.7008 / 0.5
                            elif reg==1 and train_corre==0.015:
                                corre_num = 0.021
                                k = 0.6986 / 0.5
                            elif reg==1 and train_corre==0.02:
                                corre_num = 0.0195
                                k = 0.699 / 0.5
                            else:
                                if cr ==1 :
                                    corre_num = 0.020
                                    k = 0.700 / 0.5
                                else:
                                    corre_num = 0.017
                                    k = 0.699 / 0.5
                        if ori_sigma==1:
                            if reg == 1 and train_corre == 0.01:
                                corre_num = 0.013
                                k = 1.1992 / 1
                        if ori_sigma==1.5:
                            if reg == 1 and train_corre == 0.01:
                                corre_num = 0.0116
                                k = 1.6997 / 1.5
                        if ori_sigma==2:
                            if reg == 1 and train_corre == 0.01:
                                corre_num = 0.013
                                k = 2.1963 / 2
                    if gm_method == "pca" and ori_sigma == 0.5:
                        corre_num = 0.025
                        k = 0.6980 / 0.5
                    if gm_method == "cie" and ori_sigma == 0.5:
                        corre_num = 0.029
                        k = 0.6973 / 0.5
                    if gm_method == "gmn" and ori_sigma == 0.5:
                        corre_num = 0.027
                        k = 0.6985 / 0.5


                    for cov in [1]:

                        if cov == 0:
                            cov_pro = 10  # delete 100% correlation probability
                        else:
                            cov_pro = 0

                        if cov == 0:  # RS
                            k = 0
                            corre_num = 0
                            f1 = open(
                                file_name + "/" + method + "/sigmaori" + str(ori_sigma) + '_p' + str(p)+ "_plusnoise"+str(plusnoise)+"_reg"+str(reg)+"_beta"+str(beta)+"_traincorre"+str(train_corre)+"_nnum"+str(n_num)+"_finetune"+str(finetune)+"_epoch"+ str(epoch_num)+ "_n" + str(n) + "_RS_Lvolume",
                                'w')
                            f2 = open(
                                file_name + "/" + method + "/sigmaori" + str(ori_sigma) + '_p' + str(p) + "_plusnoise"+str(plusnoise)+"_reg"+str(reg)+"_beta"+str(beta)+"_traincorre"+str(train_corre)+"_nnum"+str(n_num)+"_finetune"+str(finetune)+"_epoch"+ str(epoch_num)+ "_n" + str(n) + "_RS_Llower", 'w')
                            f3 = open(file_name + "/" + method + "/sigmaori" + str(ori_sigma) + '_p' + str(p) + "_plusnoise"+str(plusnoise)+"_reg"+str(reg)+"_beta"+str(beta)+"_traincorre"+str(train_corre)+"_nnum"+str(n_num)+"_finetune"+str(finetune)+"_epoch"+ str(epoch_num)+ "_n" + str(n) + "_RS_Lmax",
                                      'w')
                        else:  # SCR-GM
                            f1 = open(file_name + "/" + method + "/sigmaori" + str(ori_sigma) + '_correnum' + str(
                                corre_num) + '_p' + str(p) + "_plusnoise"+str(plusnoise)+"_reg"+str(reg)+"_beta"+str(beta)+"_traincorre"+str(train_corre)+"_nnum"+str(n_num)+"_finetune"+str(finetune)+"_epoch"+ str(epoch_num)+ "_n" + str(n) +"_SCR_Lvolume", 'w')
                            f2 = open(file_name + "/" + method + "/sigmaori" + str(ori_sigma) + '_correnum' + str(
                                corre_num) + '_p' + str(p) + "_plusnoise"+str(plusnoise)+"_reg"+str(reg)+"_beta"+str(beta)+"_traincorre"+str(train_corre)+"_nnum"+str(n_num)+"_finetune"+str(finetune)+"_epoch"+ str(epoch_num)+ "_n" + str(n) + "_SCR_Llower", 'w')
                            f3 = open(file_name + "/" + method + "/sigmaori" + str(ori_sigma) + '_correnum' + str(
                                corre_num) + '_p' + str(p) + "_plusnoise"+str(plusnoise)+"_reg"+str(reg)+"_beta"+str(beta)+"_traincorre"+str(train_corre)+"_nnum"+str(n_num)+"_finetune"+str(finetune)+"_epoch"+ str(epoch_num)+ "_n" + str(n) + "_SCR_Lmax", 'w')

                        print("method", method, "/ori_sigma:", ori_sigma, "/cov:", cov)
                        print("idx\tpABar_item\tradius_A\tradius_B\tcorrect\ttime", file=f1, flush=True)
                        print("idx\tpABar_item\tradius_A\tradius_B\tcorrect\ttime", file=f2, flush=True)
                        print("idx\tpABar_item\tradius_A\tradius_B\tcorrect\ttime", file=f3, flush=True)
                        verbose = True
                        print('Start evaluation...')
                        since = time.time()

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
                            if verbose:
                                print('Evaluating class {}: {}/{}'.format(cls, i, len(clss)))

                            running_since = time.time()
                            iter_num = 0

                            for inputs in dataloaders[i]:
                                if iter_num >= cfg.EVAL.SAMPLES / inputs['batch_size']:
                                    break
                                if model.module.device != torch.device('cpu'):
                                    inputs = data_to_cuda(inputs)

                                print("sample_number: ", iter_num)

                                batch_num = inputs['batch_size']

                                iter_num = iter_num + 1
                                batch_num = inputs['batch_size']

                                with torch.set_grad_enabled(False):

                                    before_time = time.time()
                                    clas_item_A = inputs['ns'][0].cpu().item()
                                    clas_item_B = inputs['ns'][1].cpu().item()

                                    pABar_item, result_Lvolume_A, result_Llower_A, result_Lmax_A, result_Lvolume_B, result_Llower_B, result_Lmax_B = smoothed_classifier.certify(
                                        inputs, n0=100, n=n, alpha=0.001, batch_size=8,
                                        clas_A=clas_item_A,
                                        clas_B=clas_item_B,
                                        sigma_pro=cov_pro, sigma=ori_sigma, k=k, p=p, corre_num=corre_num)

                                    after_time = time.time()

                                    time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))

                                    if result_Lvolume_A[0][0] != -1:
                                        print("{}\t{:.3}\t{:.3}\t{:.3}\t{}\t{}".format(number, pABar_item,
                                                                                       result_Lvolume_A[0][1],
                                                                                       result_Lvolume_B[0][1], 1,
                                                                                       time_elapsed), file=f1, flush=True)
                                    else:
                                        print("{}\t{:.3}\t{:.3}\t{:.3}\t{}\t{}".format(number, pABar_item,
                                                                                       result_Lvolume_A[0][1],
                                                                                       result_Lvolume_B[0][1], 0,
                                                                                       time_elapsed), file=f1, flush=True)

                                    if result_Llower_A[0][0] != -1:
                                        print("{}\t{:.3}\t{:.3}\t{:.3}\t{}\t{}".format(number, pABar_item,
                                                                                       result_Llower_A[0][1],
                                                                                       result_Llower_B[0][1], 1,
                                                                                       time_elapsed), file=f2, flush=True)
                                    else:
                                        print("{}\t{:.3}\t{:.3}\t{:.3}\t{}\t{}".format(number, pABar_item,
                                                                                       result_Llower_A[0][1],
                                                                                       result_Llower_B[0][1], 0,
                                                                                       time_elapsed), file=f2,
                                              flush=True)

                                    if result_Lmax_A[0][0] != -1:
                                        print("{}\t{:.3}\t{:.3}\t{:.3}\t{}\t{}".format(number, pABar_item,
                                                                                       result_Lmax_A[0][1],
                                                                                       result_Lmax_B[0][1], 1,
                                                                                       time_elapsed), file=f3, flush=True)
                                    else:
                                        print("{}\t{:.3}\t{:.3}\t{:.3}\t{}\t{}".format(number, pABar_item,
                                                                                       result_Lmax_A[0][1],
                                                                                       result_Lmax_B[0][1], 0,
                                                                                       time_elapsed), file=f3,
                                              flush=True)
                                    number += 1

                        f1.close()
                        f2.close()
                        f3.close()

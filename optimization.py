
import time
from datetime import datetime
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
import numpy as np
from torch.distributions import multivariate_normal
import math
import copy

#parameters
learning_rate = 0.001
verbose = True
iteration_num= 10
method="ngmv2" #pca ngmv2 cie

#The relationship between b and eigenvalues
corre_and_prod= [[0.1304, 58], [0.1648, 80],
    [0.1570, 54], [0.2074, 53],
    [0.2153, 62], [0.2117, 52],
    [0.2478, 58], [0.2759, 56],
    [0.2285, 64], [0.2537, 51],
     [0.2035, 51], [0.2650, 51],
     [0.2643, 53], [0.2553, 53], [0.2619, 56]]

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
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    wb = xlwt.Workbook()
    ws = wb.add_sheet('epoch{}'.format(cfg.EVAL.EPOCH))

    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / ('eval_log_' + now_time + '.log'))) as _:

        model_path = ''
        if cfg.EVAL.EPOCH is not None and cfg.EVAL.EPOCH > 0:
            model_path = str(Path(cfg.OUTPUT_PATH) / 'params' / 'params_{:04}.pt'.format(cfg.EVAL.EPOCH))
        if len(cfg.PRETRAINED_PATH) > 0:
            model_path = cfg.PRETRAINED_PATH
        if len(model_path) > 0:
            print('Loading model parameters from {}'.format(model_path))
            load_model(model, model_path)

        verbose = True
        print('Start evaluation...')
        since = time.time()

        model.eval()

        dataloaders = []

        for sigma in [0.5]:

            theta = Variable(torch.ones(1).to(device) * sigma, requires_grad=True).to(device)
            corre_num = Variable(torch.ones(1).to(device) * 0.01, requires_grad=True).to(device)
            optimizer1 = torch.optim.Adam([theta], lr=learning_rate)
            optimizer2 = torch.optim.Adam([corre_num], lr=learning_rate)
            initial_theta = theta.detach().clone()

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

            for i, cls in enumerate(clss):

                if verbose:
                    print('Evaluating class {}: {}/{}'.format(cls, i, len(clss)))

                running_since = time.time()
                print("Start!")
                max_gap_sum = 0

                for _ in range(iteration_num):

                    iter_num = 0
                    gap_sum = 0
                    keypoint_num_sum_0 = 0
                    keypoint_num_sum_1 = 0

                    for inputs in dataloaders[i]:

                        iter_num = iter_num + 1

                        if iter_num >= cfg.EVAL.SAMPLES / inputs['batch_size']:
                            break
                        if model.module.device != torch.device('cpu'):
                            inputs = data_to_cuda(inputs)

                        ori_inputs = copy.deepcopy(inputs)
                        keypoint_num_0 = inputs['ns'][0].cpu().item()
                        keypoint_num_sum_0 += keypoint_num_0
                        keypoint_num_1 = inputs['ns'][1].cpu().item()
                        keypoint_num_sum_1 += keypoint_num_1

                        corre_0 = ((torch.ones(keypoint_num_0, keypoint_num_0).to(device) - torch.eye(
                            keypoint_num_0).to(device)) * corre_num + torch.eye(keypoint_num_0).to(device)) * theta
                        corre_0 = torch.tril(corre_0, diagonal=1)
                        corre_0 = torch.triu(corre_0, diagonal=-1)
                        Sigma_0 = torch.mm(corre_0, corre_0)
                        multivar_0 = multivariate_normal.MultivariateNormal(torch.zeros(keypoint_num_0).cuda(), Sigma_0)

                        corre_1 = ((torch.ones(keypoint_num_1, keypoint_num_1).to(device) - torch.eye(
                            keypoint_num_1).to(device)) * corre_num + torch.eye(keypoint_num_1).to(device)) * theta
                        corre_1 = torch.tril(corre_1, diagonal=1)
                        corre_1 = torch.triu(corre_1, diagonal=-1)
                        Sigma_1 = torch.mm(corre_1, corre_1)
                        multivar_1 = multivariate_normal.MultivariateNormal(torch.zeros(keypoint_num_1).to(device),
                                                                            Sigma_1)
                        noise_0 = torch.unsqueeze(torch.cat((torch.unsqueeze(multivar_0.sample(),dim=1),torch.unsqueeze(multivar_0.sample(),dim=1)),dim=1),dim=0)
                        noise_1 = torch.unsqueeze(torch.cat(
                            (torch.unsqueeze(multivar_1.sample(), dim=1), torch.unsqueeze(multivar_1.sample(), dim=1)),
                            dim=1), dim=0)

                        inputs['Ps'][0] = ori_inputs['Ps'][0] + noise_0
                        inputs['Ps'][1] = ori_inputs['Ps'][1] + noise_1

                        out = model(inputs)

                        #distance to true
                        gap = torch.sum(out['perm_mat']*inputs["gt_perm_mat"])/torch.sum(inputs["gt_perm_mat"])

                        gap_sum += gap

                    max_gap_sum = max(max_gap_sum, gap_sum)
                    keypoint_num_ave_0 = keypoint_num_sum_0/cfg.EVAL.SAMPLES
                    keypoint_num_ave_1 = keypoint_num_sum_1 / cfg.EVAL.SAMPLES
                    standard_min_pro_0 = corre_and_prod[int(keypoint_num_ave_0)-5][0]
                    standard_min_pro_1 = corre_and_prod[int(keypoint_num_ave_1) - 5][0]
                    standard_min_pro_num_0 = corre_and_prod[int(keypoint_num_ave_0) - 5][1]
                    standard_min_pro_num_1 = corre_and_prod[int(keypoint_num_ave_1) - 5][1]
                    prod = (1-standard_min_pro_0)/standard_min_pro_num_0*(standard_min_pro_num_0-corre_num*100)+(1-standard_min_pro_1)/standard_min_pro_num_1*(standard_min_pro_num_1-corre_num*100)
                    proxy_radius = prod * gap_sum * theta

                    kappa = 200
                    radius_maximizer = -(proxy_radius.sum() + (gap_sum-max_gap_sum)*(corre_num/kappa))

                    radius_maximizer.backward(retain_graph=True)
                    optimizer1.step()
                    optimizer2.step()
                    optimizer1.zero_grad()
                    optimizer2.zero_grad()

                    with torch.no_grad():
                        torch.max(theta, initial_theta, out=theta)
                        torch.min(corre_num, torch.ones(1).to(device)*0.99, out=corre_num)

            print("Finish!",theta,corre_num)

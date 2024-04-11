import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint
import math
from torch.distributions import multivariate_normal
import os
from src.utils.config import cfg
import random
import copy
import os
import pygmtools as pygm

torch.cuda.set_device(0)

class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module):

        self.base_classifier = base_classifier

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int, clas_A: int, clas_B: int,
                sigma: float, k: float, p:float) -> (int, float):

        self.base_classifier.eval()

        pre_X = self._sample_noise(x, n0, batch_size, clas_A, clas_B, sigma, k)

        #print(pre_X)
        X_sinkhorn = pygm.sinkhorn(pre_X.cpu().numpy())
        X_hungarian = pygm.hungarian(X_sinkhorn)

        result_Lvolume_A = []
        result_Llower_A = []
        result_Lmax_A = []
        result_Lvolume_B = []
        result_Llower_B = []
        result_Lmax_B = []

        if np.sum(np.abs(x["gt_perm_mat"].cpu().numpy() - X_hungarian)) == 0:
            true_num = self.sample_d(x, n, batch_size, p, sigma,k)
            pABar_item = self._lower_confidence_bound(true_num, n, alpha)

            if pABar_item<0.5:
                result_Lvolume_A.append([Smooth.ABSTAIN, 0.0])
                result_Llower_A.append([Smooth.ABSTAIN, 0.0])
                result_Lmax_A.append([Smooth.ABSTAIN, 0.0])

                result_Lvolume_B.append([Smooth.ABSTAIN, 0.0])
                result_Llower_B.append([Smooth.ABSTAIN, 0.0])
                result_Lmax_B.append([Smooth.ABSTAIN, 0.0])
            else:

                radius_Lvolume_A = sigma * k * norm.ppf(pABar_item)
                radius_Llower_A = sigma * k * norm.ppf(pABar_item)
                radius_Lmax_A = sigma * k * norm.ppf(pABar_item)
                radius_Lvolume_B = sigma * k * norm.ppf(pABar_item)
                radius_Llower_B = sigma * k * norm.ppf(pABar_item)
                radius_Lmax_B = sigma * k * norm.ppf(pABar_item)

                result_Lvolume_A.append([1, radius_Lvolume_A])
                result_Llower_A.append([1, radius_Llower_A])
                result_Lmax_A.append([1, radius_Lmax_A])
                result_Lvolume_B.append([1, radius_Lvolume_B])
                result_Llower_B.append([1, radius_Llower_B])
                result_Lmax_B.append([1, radius_Lmax_B])
            return float(pABar_item), result_Lvolume_A, result_Llower_A, result_Lmax_A, result_Lvolume_B, result_Llower_B, result_Lmax_B
        else:
            result_Lvolume_A.append([Smooth.ABSTAIN, 0.0])
            result_Llower_A.append([Smooth.ABSTAIN, 0.0])
            result_Lmax_A.append([Smooth.ABSTAIN, 0.0])

            result_Lvolume_B.append([Smooth.ABSTAIN, 0.0])
            result_Llower_B.append([Smooth.ABSTAIN, 0.0])
            result_Lmax_B.append([Smooth.ABSTAIN, 0.0])

            return float(
                0.0), result_Lvolume_A, result_Llower_A, result_Lmax_A, result_Lvolume_B, result_Llower_B, result_Lmax_B

    def sample_d(self,x:torch.tensor, num:int, batch_size, p:float,sigma:float,k:float):

        true_num=0
        image = x['images']  # the keypoint coordinate
        batch = copy.deepcopy(x)  # deepcopy,don't change the original x

        for _ in range(0,num):

            noise_A = (torch.randn_like(image[0]) * sigma * k).cuda()
            noise_B = (torch.randn_like(image[1]) * sigma * k).cuda()
            batch['images'][0] = image[0] + noise_A
            batch['images'][1] = image[1] + noise_B
            predictions = self.base_classifier(batch)
            if torch.sum(predictions['perm_mat']*batch["gt_perm_mat"])/torch.sum(batch["gt_perm_mat"])>p:
                true_num+=1

        return true_num

    def _sample_noise(self, x: torch.tensor, num: int, batch_size, clas_A, clas_B,
                       sigma: float,  k: float) -> np.ndarray:

        with torch.no_grad():
            batch_size = 1
            counts = torch.zeros([batch_size, clas_A, clas_B], dtype=torch.float).cuda()

            image = x['images']  # the keypoint coordinate
            batch = copy.deepcopy(x)  # deepcopy,don't change the original x

            for _ in range(ceil(num / batch_size)):
                noise_A = (torch.randn_like(image[0]) * sigma * k).cuda()
                noise_B = (torch.randn_like(image[1]) * sigma * k).cuda()
                #noise_trans_A = noise_A.unsqueeze(0)
                #noise_trans_B = noise_B.unsqueeze(0)
                batch['images'][0] = image[0] + noise_A
                batch['images'][1] = image[1] + noise_B
                predictions = self.base_classifier(batch)
                counts += predictions['perm_mat']

            counts_result = torch.sum(counts, dim=0)
            return counts_result


    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:

        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
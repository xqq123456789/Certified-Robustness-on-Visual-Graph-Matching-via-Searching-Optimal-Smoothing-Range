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

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int, clas_A: int, clas_B: int, sigma_pro: int,
                sigma: float, k:float, p:float, corre_num:float) -> (int, float):

        self.base_classifier.eval()

        # find the pre_X under n0 sample.
        pre_X, min_lamda_A, max_lamda_A, prod_proxy_A, multivar_A, min_lamda_B, max_lamda_B, prod_proxy_B, multivar_B= self._sample_noise(x, n0, batch_size, clas_A, clas_B, sigma_pro, sigma, k, corre_num)

        #predict
        X_sinkhorn = pygm.sinkhorn(pre_X.cpu().numpy())
        X_hungarian = pygm.hungarian(X_sinkhorn)

        result_Lvolume_A = []
        result_Llower_A = []
        result_Lmax_A = []
        result_Lvolume_B = []
        result_Llower_B = []
        result_Lmax_B = []

        # if predict true
        if np.sum(np.abs(x["gt_perm_mat"].cpu().numpy()-X_hungarian))==0:
            true_num=self.sample_d(x, n, batch_size, multivar_A, multivar_B, p)
            pABar_item = self._lower_confidence_bound(true_num, n, alpha)

            if pABar_item < 0.5:
                result_Lvolume_A.append([Smooth.ABSTAIN, 0.0])
                result_Llower_A.append([Smooth.ABSTAIN, 0.0])
                result_Lmax_A.append([Smooth.ABSTAIN, 0.0])

                result_Lvolume_B.append([Smooth.ABSTAIN, 0.0])
                result_Llower_B.append([Smooth.ABSTAIN, 0.0])
                result_Lmax_B.append([Smooth.ABSTAIN, 0.0])
            else:
                radius_Lvolume_A = torch.sqrt(prod_proxy_A) * norm.ppf(pABar_item) / 2
                radius_Llower_A = norm.ppf(pABar_item) / (2 * torch.sqrt(max_lamda_A))
                radius_Lmax_A = norm.ppf(pABar_item) / (2 * torch.sqrt(min_lamda_A))
                radius_Lvolume_B = torch.sqrt(prod_proxy_B) * norm.ppf(pABar_item) / 2
                radius_Llower_B = norm.ppf(pABar_item) / (2 * torch.sqrt(max_lamda_B))
                radius_Lmax_B = norm.ppf(pABar_item) / (2 * torch.sqrt(min_lamda_B))

                result_Lvolume_A.append([1, radius_Lvolume_A])
                result_Llower_A.append([1, radius_Llower_A])
                result_Lmax_A.append([1, radius_Lmax_A])
                result_Lvolume_B.append([1, radius_Lvolume_B])
                result_Llower_B.append([1, radius_Llower_B])
                result_Lmax_B.append([1, radius_Lmax_B])

            return float(
                pABar_item), result_Lvolume_A, result_Llower_A, result_Lmax_A, result_Lvolume_B, result_Llower_B, result_Lmax_B

        else:
            result_Lvolume_A.append([Smooth.ABSTAIN, 0.0])
            result_Llower_A.append([Smooth.ABSTAIN, 0.0])
            result_Lmax_A.append([Smooth.ABSTAIN, 0.0])

            result_Lvolume_B.append([Smooth.ABSTAIN, 0.0])
            result_Llower_B.append([Smooth.ABSTAIN, 0.0])
            result_Lmax_B.append([Smooth.ABSTAIN, 0.0])

            return float(0.0), result_Lvolume_A, result_Llower_A, result_Lmax_A, result_Lvolume_B, result_Llower_B, result_Lmax_B


    def sample_d(self,x:torch.tensor, num:int, batch_size, multivar_A:torch.tensor, multivar_B:torch.tensor, p):

        true_num=0
        length_A = x["Ps"][0].shape[1]
        length_B = x["Ps"][1].shape[1]

        point = x['Ps']  # the keypoint coordinate
        batch = copy.deepcopy(x)  # deepcopy,don't change the original x

        for _ in range(0,num):
            noise_A = multivar_A.sample().reshape(length_A, 2)
            noise_trans_A = noise_A.unsqueeze(0)
            noise_B = multivar_B.sample().reshape(length_B, 2)
            noise_trans_B = noise_B.unsqueeze(0)

            batch['Ps'][0] = point[0] + noise_trans_A
            batch['Ps'][1] = point[1] + noise_trans_B
            predictions = self.base_classifier(batch)
            if torch.sum(predictions['perm_mat']*batch["gt_perm_mat"])/torch.sum(batch["gt_perm_mat"])>p:
                true_num+=1

        return true_num

    def _sample_noise(self, x: torch.tensor, num: int, batch_size, clas_A, clas_B, sigma_pro, sigma: float, k:float, corre_num: float) -> np.ndarray:

        with torch.no_grad():
            if num <= 100:
                batch_size = 1

            counts = torch.zeros([batch_size, clas_A, clas_B], dtype=torch.float).cuda()

            point = x['Ps'] #the keypoint coordinate
            batch = copy.deepcopy(x)  # deepcopy,don't change the original x

            #get the covariance, min_lamda, max_lamda and pro_proxy of seperate keypoint
            covar_A, min_lamda_A, max_lamda_A, prod_proxy_A = self.co_variance(point[0], sigma_pro, sigma,  k, corre_num)
            covar_B, min_lamda_B, max_lamda_B, prod_proxy_B = self.co_variance(point[1], sigma_pro, sigma,  k, corre_num)

            length_A = len(covar_A)
            length_B = len(covar_B)
            multivar_A = multivariate_normal.MultivariateNormal(torch.zeros(length_A).cuda(), covar_A)
            multivar_B = multivariate_normal.MultivariateNormal(torch.zeros(length_B).cuda(), covar_B)

            for _ in range(ceil(num / batch_size)):
                noise_A = multivar_A.sample().reshape(int(length_A / 2), 2)
                noise_B = multivar_B.sample().reshape(int(length_B / 2), 2)
                noise_trans_A = noise_A.unsqueeze(0)
                noise_trans_B = noise_B.unsqueeze(0)
                batch['Ps'][0] = point[0] + noise_trans_A
                batch['Ps'][1] = point[1] + noise_trans_B
                predictions = self.base_classifier(batch)
                counts += predictions['perm_mat']

            counts_result = torch.sum(counts, dim=0)
            return counts_result, min_lamda_A, max_lamda_A, prod_proxy_A, multivar_A, min_lamda_B, max_lamda_B, prod_proxy_B, multivar_B

    def co_variance(self, point: torch.tensor, sigma_pro: int, sigma: float, k: float, corre_num:float) -> torch.tensor:

        length = point.size()[1]

        if sigma_pro == 10:  # RS

            covar_result = torch.eye(length * 2).cuda() * sigma * sigma

        else:

            covar_result = self.corr(length, sigma, k, corre_num)

        (evals_covar, evecs_covar) = torch.eig(covar_result, eigenvectors=True)
        pra = (1 / math.gamma(length + 1)) ** (1 / (length * 2))
        prod_proxy = (math.pi * (pra ** 2)) * torch.prod((evals_covar.T[0] ** (1 / (2 * length))).reshape(-1))
        inv_result = torch.inverse(covar_result)  # inverse of Sigma
        (evals, evecs) = torch.eig(inv_result, eigenvectors=True)
        min_lamda = torch.min(evals.T[0])
        max_lamda = torch.max(evals.T[0])

        return covar_result, min_lamda, max_lamda, prod_proxy

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:

        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]

    def corr(self, length: int, sigma: float, k: float, corre_num: float):

        covar_result = torch.eye(length * 2).cuda()
        for i in range(length-1):
            covar_result[i][i + 1] = corre_num
            covar_result[i + 1][i] = corre_num
            covar_result[i + length][i + length+1] = corre_num
            covar_result[i + length+1][i + length] = corre_num
        covar = torch.mm(covar_result,covar_result) * sigma * sigma * k * k

        return covar
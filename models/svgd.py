"""
Stein Variational Gradient Descent for Deep ConvNet on GPU.
Current implementation is mainly using for-loops over model instances.

Oct 29, 2017
Apr 30, 2018
"""  

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.distributions import Gamma
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.misc import log_sum_exp, parameters_to_vector, vector_to_parameters
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.switch_backend('agg')
from time import time
import copy
import sys
import os
import gc
import math

from args import args, device

n_samples = args.n_samples
lr = args.lr
lr_noise = args.lr_noise
ntrain = args.ntrain
ckpt_freq = args.ckpt_freq
ckpt_dir = args.ckpt_dir


class SVGD(object):
    """Base class for Stein Variational Gradient Descent, with for-loops...
    The Bayesian neural network is defined in `bayes_nn.BayesNN` class.    

    References:
        Liu, Qiang, and Dilin Wang. "Stein variational gradient descent:
        A general purpose bayesian inference algorithm."
        Advances In Neural Information Processing Systems. 2016.

    Args:
        model (nn.Module): The model to be instantiated `n_samples` times
        data_loader (utils.data.DataLoader): For training and testing
        n_samples (int): Number of samples for uncertain parameters
    """

    def __init__(self, bayes_nn, train_loader):
        """
        For-loop implementation of SVGD.

        Args:
            bayes_nn (nn.Module): Bayesian NN
            train_loader (utils.data.DataLoader): Training data loader
            logger (dict)

        """
        self.bayes_nn = bayes_nn
        self.train_loader = train_loader
        self.n_samples = n_samples
        self.optimizers, self.schedulers = self._optimizers_schedulers(
                                            lr, lr_noise)


    def _squared_dist(self, X):
        """Computes squared distance between each row of `X`, ||X_i - X_j||^2

        Args:
            X (Tensor): (S, P) where S is number of samples, P is the dim of 
                one sample

        Returns:
            (Tensor) (S, S)
        """
        XXT = torch.mm(X, X.t())
        XTX = XXT.diag()
        return -2.0 * XXT + XTX + XTX.unsqueeze(1)


    def _Kxx_dxKxx(self, X):
        """
        Computes covariance matrix K(X,X) and its gradient w.r.t. X
        for RBF kernel with design matrix X, as in the second term in eqn (8)
        of reference SVGD paper.

        Args:
            X (Tensor): (S, P), design matrix of samples, where S is num of
                samples, P is the dim of each sample which stacks all params
                into a (1, P) row. Thus P could be 1 millions.
        """
        squared_dist = self._squared_dist(X)
        l_square = 0.5 * squared_dist.median() / math.log(self.n_samples)
        Kxx = torch.exp(-0.5 / l_square * squared_dist)
        # matrix form for the second term of optimal functional gradient
        # in eqn (8) of SVGD paper
        dxKxx = (Kxx.sum(1).diag() - Kxx).matmul(X) / l_square

        return Kxx, dxKxx

    
    def _optimizers_schedulers(self, lr, lr_noise):
        """Initialize Adam optimizers and schedulers (ReduceLROnPlateau)

        Args:
            lr (float): learning rate for NN parameters `w`
            lr_noise (float): learning rate for noise precision `log_beta`
        """
        optimizers = []
        schedulers = []
        for i in range(self.n_samples):
            parameters = [{'params': [self.bayes_nn[i].log_beta], 'lr': lr_noise},
                    {'params': self.bayes_nn[i].features.parameters()}]
            optimizer_i = torch.optim.Adam(parameters, lr=lr)
            optimizers.append(optimizer_i)
            schedulers.append(ReduceLROnPlateau(optimizer_i, 
                    mode='min', factor=0.1, patience=10, verbose=True))
        return optimizers, schedulers


    def train(self, epoch, logger):

        self.bayes_nn.train()
        mse = 0.
        for batch_idx, (input, target) in enumerate(self.train_loader):
            input, target = input.to(device), target.to(device)

            self.bayes_nn.zero_grad()
            output = torch.zeros_like(target)
            # all gradients of log joint probability: (S, P)
            grad_log_joint = []
            # all model parameters (particles): (S, P)
            theta = []
            # store the joint probabilities
            log_joint = 0.

            for i in range(self.n_samples):
                output_i = self.bayes_nn[i].forward(input)
                output += output_i.detach()
                log_joint_i = self.bayes_nn._log_joint(i, output_i, target, ntrain)
                # backward to compute gradients of log joint probabilities
                log_joint_i.backward()
                # monitoring purpose
                log_joint += log_joint_i.item()
                # backward frees memory for computation graph
                # computation below does not build computation graph
                # extract parameters and their gradients out from models
                vec_param, vec_grad_log_joint = parameters_to_vector(
                    self.bayes_nn[i].parameters(), both=True)
                grad_log_joint.append(vec_grad_log_joint.unsqueeze(0))
                theta.append(vec_param.unsqueeze(0))

            # calculating the kernel matrix and its gradients
            theta = torch.cat(theta)
            Kxx, dxKxx = self._Kxx_dxKxx(theta)
            grad_log_joint = torch.cat(grad_log_joint)
            # this line needs S x P memory
            grad_logp = torch.mm(Kxx, grad_log_joint)
            # negate grads here!!!
            grad_theta = - (grad_logp + dxKxx) / self.n_samples
            # explicitly deleting variables does not release memory :(
       
            # update param gradients
            for i in range(self.n_samples):
                vector_to_parameters(grad_theta[i],
                    self.bayes_nn[i].parameters(), grad=True)
                self.optimizers[i].step()
            # WEAK: no loss function to suggest when to stop or
            # approximation performance
            mse += F.mse_loss(output / self.n_samples, target).item()

        # logging
        rmse_train = np.sqrt(mse / len(self.train_loader))
        r2_train = 1 - mse * target.numel()  / logger['train_output_var']
        logger['rmse_train'].append(rmse_train)
        logger['r2_train'].append(r2_train)
        logger['log_beta'].append(self.bayes_nn.log_beta.mean().item())
        print("epoch {}, training r2: {:.6f}".format(epoch, r2_train))

        for i in range(self.n_samples):
            self.schedulers[i].step(rmse_train)

        # save trained models
        if epoch % ckpt_freq == 0:
            torch.save(self.bayes_nn.state_dict(), 
                ckpt_dir + "/model_epoch{}.pth".format(epoch))

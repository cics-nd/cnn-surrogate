"""
Particle approximations for posterior of Bayesian neural net used in SVGD.

References:
    Liu, Qiang, and Dilin Wang. "Stein variational gradient descent:
    A general purpose bayesian inference algorithm." NIPS. 2016.

methods:
    __init__
    forward
    compute_loss
    compute_mse_nlp
    predict
    propagate

Note: 
`torch.distributions` is not much used in this implementation to keep simple.
Also we trade computing for memory by using for-loop rather than in a batch way.
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributions import Gamma
from utils.misc import parameters_to_vector, vector_to_parameters, log_sum_exp

import math
import copy
import sys
from time import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BayesNN(nn.Module):
    """Class for Bayesian NNs with Stein Variational Gradient Descent.
    Not for usage independently.
    
    Bayesian NNs: y = f(x, w) + n

    uncertain weights:
            w_i ~ Normal(w_i | mu=0, 1 / alpha) 
            alpha ~ Gamma(alpha | shape=1, rate=0.05) (shared)
            --> w_i ~ StudentT(w_i | mu=0, lambda=shape/rate, nu=2*shape)
            Parameterization of StudentT in Bishop p.103 Eq. (2.159)

    Assumptions on noise:
        Additive, Gaussian, homoscedastic (independent of input), 
        output wise (same for every pixels in the output).
            n ~ Normal(0, 1 / beta)
            beta ~ Gamma(beta | shape=2, rate=2e-6)

    Hyperparameters for weights and noise are pre-defined based on heuristic.

    Given a deterministic `model`, initialize `n_samples` replicates
    of the `model`. (plus `n_samples` of noise precision realizations)

    `model` must implement `reset_parameters` method for the replicates
    to have different initial realizations of uncertain parameters.

    References:
        Liu, Qiang, and Dilin Wang. "Stein variational gradient descent:
        A general purpose bayesian inference algorithm."
        Advances In Neural Information Processing Systems. 2016.

    Args:
        model (nn.Module): The deterministic NN to be instantiated `n_samples` 
            times
        data_loader (utils.data.DataLoader): For training and testing
        n_samples (int): Number of samples for uncertain parameters
    """
    def __init__(self, model, n_samples=20):
        super(BayesNN, self).__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("model {} is not a Module subclass".format(
                torch.typename(model)))

        self.n_samples = n_samples

        # w_i ~ StudentT(w_i | mu=0, lambda=shape/rate, nu=2*shape)
        # for efficiency, represent StudentT params using Gamma params
        self.w_prior_shape = 1.
        self.w_prior_rate = 0.05
        
        # noise variance 1e-6: beta ~ Gamma(beta | shape, rate)
        self.beta_prior_shape = 2.
        self.beta_prior_rate = 1.e-6

        # replicate `n_samples` instances with the same network as `model`
        instances = []
        for i in range(n_samples):
            new_instance = copy.deepcopy(model)
            # initialize each model instance with their defualt initialization
            # instead of the prior
            new_instance.reset_parameters()
            print('Reset parameters in model instance {}'.format(i))
            instances.append(new_instance)
        self.nnets = nn.ModuleList(instances)
        del instances

        # log precision (Gamma) of Gaussian noise
        log_beta = Gamma(self.beta_prior_shape, 
                         self.beta_prior_rate).sample((self.n_samples,)).log()
        for i in range(n_samples):
            self.nnets[i].log_beta = Parameter(log_beta[i])

        print('Total number of parameters: {}'.format(self._num_parameters()))

    def _num_parameters(self):
        count = 0
        for name, param in self.named_parameters():
            # print(name)
            count += param.numel()
        return count

    def __getitem__(self, idx):
        return self.nnets[idx]

    @property
    def log_beta(self):
        return torch.tensor([self.nnets[i].log_beta.item() 
            for i in range(self.n_samples)], device=device)

    def forward(self, input):
        """Computes all the `n_samples` NN output
        Args:
            input: N x iC x iH x iW

        Return:
            output: S x N x oC x oH x oW
        """
        output = []
        for i in range(self.n_samples):
            output.append(self.nnets[i].forward(input))
        output = torch.stack(output)

        return output

    def _log_joint(self, index, output, target, ntrain):
        """Log joint probability or unnormalized posterior for single model
        instance. Ignoring constant terms for efficiency.
        Can be implemented in batch computation, but memory is the bottleneck.
        Thus here we trade computation for memory, e.g. using for loop.

        Args:
            index (int): model index, 0, 1, ..., `n_samples`
            output (Tensor): B x oC x oH x oW
            target (Tensor): B x oC x oH x oW
            ntrain (int): total number of training data, mini-batch is used to
                evaluate the log joint prob

        Returns:
            Log joint probability (zero-dim tensor)
        """
        # Normal(target | output, 1 / beta * I)
        log_likelihood = ntrain / output.size(0) * (
                            - 0.5 * self.nnets[index].log_beta.exp()
                            * (target - output).pow(2).sum()
                            + 0.5 * target.numel() * self.nnets[index].log_beta)
        # log prob of prior of weights, i.e. log prob of studentT
        log_prob_prior_w = torch.tensor(0.).to(device)
        for param in self.nnets[index].features.parameters():
            log_prob_prior_w += \
                torch.log1p(0.5 / self.w_prior_rate * param.pow(2)).sum()
        log_prob_prior_w *= -(self.w_prior_shape + 0.5)
        # log prob of prior of log noise-precision (NOT noise precision)
        log_prob_prior_log_beta = (self.beta_prior_shape * self.nnets[index].log_beta \
                    - self.nnets[index].log_beta.exp() * self.beta_prior_rate)
        return log_likelihood + log_prob_prior_w + log_prob_prior_log_beta


    def _compute_mse_nlp(self, input, target, size_average=True, out=False):
        """Evaluate the MSE and Negative Log Probability.

        Args:
            input (Tensor): (N, iC, iH, iW)
            target (Tensor): (N, oC, oH, oW)
            size_average (bool)
            out (bool): If True, return output of `bayes_nn` w. `input`

        Returns:
            (mse, nlp) if `out` is False, o.w. (mse, nlp, output)
            where output is of size (S, N, oC, oH, oW)
        """
        # S x N x oC x oH x oW
        output = self.forward(input)
        # S x 1 x 1 x 1 x 1
        log_beta = self.log_beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        log_2pi_S = torch.tensor(0.5 * target[0].numel() * math.log(2 * math.pi)
                       + math.log(self.n_samples), device=device)
        # S x N
        exponent = - 0.5 * (log_beta.exp() * ((target - output) ** 2)).view(
            self.n_samples, target.size(0), -1).sum(-1) \
                   + 0.5 * target[0].numel() * self.log_beta.unsqueeze(-1)

        # n = target[0].numel()
        nlp = - log_sum_exp(exponent, dim=0).mean() + log_2pi_S
        mse = ((target - output.mean(0)) ** 2).mean()

        if not size_average:
            mse *= target.numel()
            nlp *= target.size(0)
        if not out:
            return mse, nlp
        else:
            return mse, nlp, output


    def predict(self, x_test):
        """
        Predictive mean and variance at x_test. (only average over w and beta)
        Args:
            x_test (Tensor): [N, *], test input
        """
        # S x N x oC x oH x oW
        y = self.forward(x_test)
        y_pred_mean = y.mean(0)
        # compute predictive variance per pixel
        # N x oC x oH x oW
        EyyT = (y ** 2).mean(0)
        EyEyT = y_pred_mean ** 2
        beta_inv = (- self.log_beta).exp()
        y_pred_var = beta_inv.mean() + EyyT - EyEyT

        return y_pred_mean, y_pred_var


    def propagate(self, mc_loader):
        """
        Mean and Variance statistics of predictive output distribution
        averaging over the input distribution, i.e. uncertainty propagation.

        First compute the conditional predictive mean and var given realizations
        of uncertain surrogate; then compute the statistics of those conditional
        statistics.

        Args:
            mc_loader (torch.utils.data.DataLoader): dataloader for the Monte 
                Carlo data (10,000 is used in this work)

            S: num of samples
            M: num of data
            D: output dimensions
        """
        # First compute conditional statistics
        # S x N x oC x oH x oW
        # self.cpu()
        # x_test = x_test.cpu()
        # print('here')

        # S x oC x oH x oW
        output_size = mc_loader.dataset[0][1].size()
        cond_Ey = torch.zeros(self.n_samples, *output_size, device=device)
        cond_Eyy = torch.zeros_like(cond_Ey)

        for _, (x_mc, _) in enumerate(mc_loader):
            x_mc = x_mc.to(device)
            # S x B x oC x oH x oW            
            y = self.forward(x_mc)
            cond_Ey += y.mean(1)
            cond_Eyy += y.pow(2).mean(1)
        cond_Ey /= len(mc_loader)
        cond_Eyy /= len(mc_loader)
        beta_inv = (- self.log_beta).exp()
        print('Noise variances: {}'.format(beta_inv))
        
        y_cond_pred_var = cond_Eyy - cond_Ey ** 2 \
                     + beta_inv.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # compute statistics of conditional statistics
        return cond_Ey.mean(0), cond_Ey.var(0), \
               y_cond_pred_var.mean(0), y_cond_pred_var.var(0)


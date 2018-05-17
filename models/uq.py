import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.io
from scipy.stats import norm as scipy_norm
import seaborn as sns
from utils.misc import mkdir
from utils.plot import plot_prediction_bayes, plot_MC
from utils.lhs import lhs
from args import args, device
plt.switch_backend('agg')

assert args.post, 'Add --post flag in command line for post-proc tasks, e.g. UQ.'

run_dir = args.run_dir
ntrain = args.ntrain
plot_fn=args.plot_fn
epochs = args.epochs


class UQ(object):
    r"""Class for uncertainty quantification tasks, include:
    
    - prediction at one input realization
    - uncertainty propagation
    - distribution estimate at certain location
    - reliability diagram (assess uncertainty quality)

    Args:
        bayes_nn (bayes_nn.BayesNN): Pre-trained Bayesian NN
        mc_loader (utils.data.DataLoader): Dataloader for Monte Carlo data
    """
    def __init__(self, bayes_nn, mc_loader):
        self.bnn = bayes_nn
        self.mc_loader = mc_loader

    def plot_prediction_at_x(self, n_pred):
        r"""Plot `n_pred` predictions for randomly selected input from MC dataset.

        - target
        - predictive mean
        - error of the above two
        - two standard deviation of predictive output distribution

        Args:
            n_pred: number of candidate predictions
        """
        print('Plotting predictions at x from MC dataset......................')
        np.random.seed(1)
        idx = np.random.permutation(len(self.mc_loader.dataset))[:n_pred]
        for i in idx:
            print('input index: {}'.format(i))
            input, target = self.mc_loader.dataset[i]
            pred_mean, pred_var = self.bnn.predict(input.unsqueeze(0).to(device))
            save_dir = run_dir + '/predict_at_x'
            mkdir(save_dir)
            plot_prediction_bayes(save_dir, target, pred_mean.squeeze(0), 
                pred_var.squeeze(0), epochs, i, plot_fn=plot_fn)


    def propagate_uncertainty(self):
        print("Propagate Uncertainty using pre-trained surrogate .............")
        # compute MC sample mean and variance in mini-batch
        sample_mean_x = torch.zeros_like(self.mc_loader.dataset[0][0])
        sample_var_x = torch.zeros_like(sample_mean_x)
        sample_mean_y = torch.zeros_like(self.mc_loader.dataset[0][1])
        sample_var_y = torch.zeros_like(sample_mean_y)

        for _, (x_test_mc, y_test_mc) in enumerate(self.mc_loader):
            x_test_mc, y_test_mc = x_test_mc, y_test_mc
            sample_mean_x += x_test_mc.mean(0)
            sample_mean_y += y_test_mc.mean(0)
        sample_mean_x /= len(self.mc_loader)
        sample_mean_y /= len(self.mc_loader)

        for _, (x_test_mc, y_test_mc) in enumerate(self.mc_loader):
            x_test_mc, y_test_mc = x_test_mc, y_test_mc
            sample_var_x += ((x_test_mc - sample_mean_x) ** 2).mean(0)
            sample_var_y += ((y_test_mc - sample_mean_y) ** 2).mean(0)
        sample_var_x /= len(self.mc_loader)
        sample_var_y /= len(self.mc_loader)

        # plot input MC
        stats_x = torch.stack((sample_mean_x, sample_var_x)).cpu().numpy()
        fig, _ = plt.subplots(1, 2)
        for i, ax in enumerate(fig.axes):
            # ax.set_title(titles[i])
            ax.set_aspect('equal')
            ax.set_axis_off()
            # im = ax.imshow(stats_x[i].squeeze(0),
            #                interpolation='bilinear', cmap=self.args.cmap)
            im = ax.contourf(stats_x[i].squeeze(0), 50, cmap='jet')
            for c in im.collections:
                c.set_edgecolor("face")
                c.set_linewidth(0.000000000001)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                                format=ticker.ScalarFormatter(useMathText=True))
            cbar.formatter.set_powerlimits((0, 0))
            cbar.ax.yaxis.set_offset_position('left')
            cbar.update_ticks()
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
        out_stats_dir = run_dir + '/out_stats'
        mkdir(out_stats_dir)
        plt.savefig(out_stats_dir + '/input_MC.pdf', di=300, bbox_inches='tight')
        plt.close(fig)
        print("Done plotting input MC, num of training: {}".format(ntrain))

        # MC surrogate predictions
        y_pred_EE, y_pred_VE, y_pred_EV, y_pred_VV = self.bnn.propagate(self.mc_loader)
        print('Done MC predictions')

        # plot the 4 output stats
        # plot the predictive mean
        plot_MC(out_stats_dir, sample_mean_y, y_pred_EE, y_pred_VE, True, ntrain)
        # plot the predictive var
        plot_MC(out_stats_dir, sample_var_y, y_pred_EV, y_pred_VV, False, ntrain)

        # save for MATLAB plotting
        scipy.io.savemat(out_stats_dir + '/out_stats.mat',
                         {'sample_mean': sample_mean_y.cpu().numpy(),
                          'sample_var': sample_var_y.cpu().numpy(),
                          'y_pred_EE': y_pred_EE.cpu().numpy(),
                          'y_pred_VE': y_pred_VE.cpu().numpy(),
                          'y_pred_EV': y_pred_EV.cpu().numpy(),
                          'y_pred_VV': y_pred_VV.cpu().numpy()})
        print('saved output stats to .mat file')


    def plot_dist(self, num_loc):
        """Plot distribution estimate in `num_loc` locations in the domain, 
        which are chosen by Latin Hypercube Sampling.

        Args:
            num_loc (int): number of locations where distribution is estimated
        """
        print('Plotting distribution estimate.................................')

        assert num_loc > 0, 'num_loc must be greater than zero'
        locations = lhs(2, num_loc, criterion='c')
        print('Locations selected by LHS: \n{}'.format(locations))
        # location (ndarray): [0, 1] x [0, 1]: N x 2
        idx = (locations * 65).astype(int)

        print('Propagating...')
        pred, target = [], []
        for _, (x_mc, t_mc) in enumerate(self.mc_loader):
            x_mc = x_mc.to(device)
            # S x B x C x H x W
            y_mc = self.bnn.forward(x_mc)
            # S x B x C x n_points
            pred.append(y_mc[:, :, :, idx[:, 0], idx[:, 1]])
            # B x C x n_points
            target.append(t_mc[:, :, idx[:, 0], idx[:, 1]])
        # S x M x C x n_points --> M x C x n_points
        pred = torch.cat(pred, dim=1).mean(0).cpu().numpy()

        print('pred size: {}'.format(pred.shape))
        # M x C x n_points
        target = torch.cat(target, dim=0).cpu().numpy()
        print('target shape: {}'.format(target.shape))
        dist_dir = run_dir + '/dist_estimate'
        mkdir(dist_dir)
        for loc in range(locations.shape[0]):
            print(loc)
            fig, _ = plt.subplots(1, 3, figsize=(12, 4))
            for c, ax in enumerate(fig.axes):
                sns.kdeplot(target[:, c, loc], color='b', ls='--', label='Monte Carlo', ax=ax)
                sns.kdeplot(pred[:, c, loc], color='r', label='surrogate', ax=ax)
                ax.legend()
            plt.savefig(dist_dir + '/loc_({}, {}).pdf'
                        .format(locations[loc][0], locations[loc][1]), dpi=300)
            plt.close(fig)


    def plot_reliability_diagram(self):
        print("Plotting reliability diagram..................................")
        # percentage: p
        # p_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        p_list = np.linspace(0.01, 0.99, 10)
        freq = []
        for p in p_list:
            count = 0
            numels = 0
            for batch_idx, (input, target) in enumerate(self.mc_loader):
                # only evaluate 2000 of the MC data to save time
                if batch_idx > 4:
                    continue
                pred_mean, pred_var = self.bnn.predict(input.to(device))

                interval = scipy_norm.interval(p, loc=pred_mean.cpu().numpy(),
                                            scale=pred_var.sqrt().cpu().numpy())

                count += ((target.numpy() >= interval[0])
                          & (target.numpy() <= interval[1])).sum()
                numels += target.numel()
                print('p: {}, {} / {} = {}'.format(p, count, numels, 
                    np.true_divide(count, numels)))
            freq.append(np.true_divide(count, numels))
        reliability_dir = run_dir + '/uncertainty_quality'
        mkdir(reliability_dir)
        plt.figure()
        plt.plot(p_list, freq, 'r', label='Bayesian surrogate')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        x = np.linspace(0, 1, 100)
        plt.plot(x, x, 'k--', label='ideal')
        plt.legend(loc='upper left')
        plt.savefig(reliability_dir + "/reliability_diagram.pdf", dpi=300)

        reliability = np.zeros((p_list.shape[0], 2))
        reliability[:, 0] = p_list
        reliability[:, 1] = np.array(freq)
        np.savetxt(reliability_dir + "/reliability_diagram.txt", reliability)
        plt.close()

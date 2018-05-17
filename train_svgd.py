"""
Stein Variational Gradient Descent for Deep ConvNet on GPU.
Current implementation is mainly using for-loops over model instances.
"""

import torch
import numpy as np
from time import time
from args import args, device
from models.dense_ed import DenseED
from models.bayes_nn import BayesNN
from models.svgd import SVGD
from utils.load_data import load_data
from utils.misc import mkdirs, logger
from utils.plot import plot_prediction_bayes, save_stats
import json


args.train_dir = args.run_dir + "/training"
args.pred_dir = args.train_dir + "/predictions"
mkdirs([args.train_dir, args.pred_dir])

# deterministic NN
dense_ed = DenseED(in_channels=args.nic, 
                    out_channels=args.noc, 
                    blocks=args.blocks,
                    growth_rate=args.growth_rate, 
                    init_features=args.init_features,
                    drop_rate=args.drop_rate,
                    bn_size=args.bn_size,
                    bottleneck=args.bottleneck,
                    out_activation=None)
print(dense_ed)
# Bayesian NN
bayes_nn = BayesNN(dense_ed, n_samples=args.n_samples).to(device)

# load data
train_data_dir = args.data_dir + '/kle{}_lhs{}.hdf5'.format(args.kle, args.ntrain)
test_data_dir = args.data_dir + '/kle{}_mc{}.hdf5'.format(args.kle, args.ntest)
train_loader, train_stats = load_data(train_data_dir, args.batch_size)
test_loader, test_stats = load_data(test_data_dir, args.test_batch_size)
logger['train_output_var'] = train_stats['y_var']
logger['test_output_var'] = test_stats['y_var']
print('Loaded data!')

# Initialize SVGD
svgd = SVGD(bayes_nn, train_loader)


def test(epoch, logger, test_fixed=None):
    """Evaluate model during training. 
    Print predictions including 4 rows:
        1. target
        2. predictive mean
        3. error of the above two
        4. two sigma of predictive variance

    Args:
        test_fixed (Tensor): (2, N, *), `test_fixed[0]` is the fixed test input, 
            `test_fixed[1]` is the corresponding target
    """
    bayes_nn.eval()
    
    mse_test, nlp_test = 0., 0.
    for batch_idx, (input, target) in enumerate(test_loader):
        input, target = input.to(device), target.to(device)
        mse, nlp, output = bayes_nn._compute_mse_nlp(input, target, 
                            size_average=True, out=True)
        # output: S x N x oC x oH x oW --> N x oC x oH x oW
        y_pred_mean = output.mean(0)        
        EyyT = (output ** 2).mean(0)
        EyEyT = y_pred_mean ** 2
        y_noise_var = (- bayes_nn.log_beta).exp().mean()
        y_pred_var =  EyyT - EyEyT + y_noise_var

        mse_test += mse.item()
        nlp_test += nlp.item()

        if batch_idx == len(test_loader) - 1 and epoch % args.plot_freq == 0:
            if test_fixed is not None:
                n_samples = test_fixed[0].size(0)
                y_pred_mean, y_pred_var = bayes_nn.predict(test_fixed[0])
                samples_pred_mean = y_pred_mean.cpu().numpy()
                samples_target = test_fixed[1].data.cpu().numpy()
                samples_pred_var = y_pred_var.cpu().numpy()
            else:
                if epoch == args.epochs:
                    n_samples = 6
                else:
                    n_samples = 2
                idx = torch.randperm(input.size(0))[: n_samples]
                samples_pred_mean = y_pred_mean[idx].cpu().numpy()
                samples_target = target[idx].cpu().numpy()
                samples_pred_var = y_pred_var[idx].cpu().numpy()
           
            for i, index in enumerate(idx):
                print('epoch {}: plotting {}-th prediction'.format(epoch, index))
                plot_prediction_bayes(args.pred_dir, samples_target[i], 
                    samples_pred_mean[i], samples_pred_var[i], epoch, index, 
                    plot_fn=args.plot_fn)

    rmse_test = np.sqrt(mse_test / len(test_loader))
    r2_test = 1 - mse_test * target.numel() / logger['test_output_var']
    mnlp_test = nlp_test / len(test_loader)    
    logger['rmse_test'].append(rmse_test)
    logger['r2_test'].append(r2_test)
    logger['mnlp_test'].append(mnlp_test)
    print("epoch {}, testing  r2: {:.6f}, test mnlp: {}".format(
        epoch, r2_test, mnlp_test))


print('Start training.........................................................')
tic = time()
for epoch in range(1, args.epochs + 1):
    svgd.train(epoch, logger)
    with torch.no_grad():
        test(epoch, logger)
training_time = time() - tic
print('Finished training:\n{} epochs\n{} data\n{} samples (SVGD)\n{} seconds'
    .format(args.epochs, args.ntrain, args.n_samples, training_time))

# save training results
x_axis = np.arange(args.log_freq, args.epochs + args.log_freq, args.log_freq)
# plot the rmse, r2-score curve and save them in txt
save_stats(args.train_dir, logger, x_axis)

args.training_time = training_time
args.n_params, args.n_layers = dense_ed._num_parameters_convlayers()
with open(args.run_dir + "/args.txt", 'w') as args_file:
    json.dump(vars(args), args_file, indent=4)

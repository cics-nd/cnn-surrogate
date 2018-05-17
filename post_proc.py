"""
Post processing, mainly for uncertainty quantification tasks using pre-trained
Bayesian NNs.
"""

import torch
from args import args
from models.dense_ed import DenseED
from models.bayes_nn import BayesNN
from models.uq import UQ
from utils.load_data import load_data


assert args.post, 'Add --post flag in command line for post-proc UQ tasks'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
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
# print(dense_ed)
# Bayesian NN
bayes_nn = BayesNN(dense_ed, n_samples=args.n_samples).to(device)

# load the pre-trained model
if args.ckpt_epoch is not None:
    checkpoint = args.ckpt_dir + '/model_epoch{}.pth'.format(args.ckpt_epoch)
else:
    checkpoint = args.ckpt_dir + '/model_epoch{}.pth'.format(args.epochs)
bayes_nn.load_state_dict(torch.load(checkpoint))
print('Loaded pre-trained model: {}'.format(checkpoint))

# load Monte Carlo data
mc_data_dir = args.data_dir + '/kle{}_mc{}.hdf5'.format(args.kle, args.nmc)
mc_loader, _ = load_data(mc_data_dir, args.mc_batch_size)
print('Loaded Monte Carlo data!')

# Now performs UQ tasks
uq = UQ(bayes_nn, mc_loader)

with torch.no_grad():
    uq.plot_prediction_at_x(n_pred=10)
    uq.propagate_uncertainty()
    uq.plot_dist(num_loc=20)
    uq.plot_reliability_diagram()

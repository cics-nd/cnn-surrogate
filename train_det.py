"""
Train deterministic convolutional encoder-decoder networks
"""

import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from args_det import args, device
from models.dense_ed import DenseED
from utils.load_data import load_data
from utils.misc import mkdirs
from utils.plot import plot_prediction_det, save_stats
import json
from time import time

args.train_dir = args.run_dir + "/training"
args.pred_dir = args.train_dir + "/predictions"
mkdirs([args.train_dir, args.pred_dir])

# initialize model
model = DenseED(in_channels=1, out_channels=3, 
                blocks=args.blocks,
                growth_rate=args.growth_rate, 
                init_features=args.init_features,
                drop_rate=args.drop_rate,
                bn_size=args.bn_size,
                bottleneck=args.bottleneck,
                out_activation=None).to(device)
print(model)

# load checkpoint if in post mode
if args.post:
    if args.ckpt_epoch is not None:
        checkpoint = args.ckpt_dir + '/model_epoch{}.pth'.format(args.ckpt_epoch)
    else:
        checkpoint = args.ckpt_dir + '/model_epoch{}.pth'.format(args.epochs)
    model.load_state_dict(torch.load(checkpoint))
    print('Loaded pre-trained model: {}'.format(checkpoint))

# load data
train_data_dir = args.data_dir + '/kle{}_lhs{}.hdf5'.format(args.kle, args.ntrain)
test_data_dir = args.data_dir + '/kle{}_mc{}.hdf5'.format(args.kle, args.ntest)
train_loader, train_stats = load_data(train_data_dir, args.batch_size)
test_loader, test_stats = load_data(test_data_dir, args.test_batch_size)
print('Loaded data!')

optimizer = optim.Adam(model.parameters(), lr=args.lr,
                       weight_decay=args.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                    verbose=True, threshold=0.0001, threshold_mode='rel',
                    cooldown=0, min_lr=0, eps=1e-8)

n_out_pixels_train = args.ntrain * train_loader.dataset[0][1].numel()
n_out_pixels_test = args.ntest * test_loader.dataset[0][1].numel()

logger = {}
logger['rmse_train'] = []
logger['rmse_test'] = []
logger['r2_train'] = []
logger['r2_test'] = []

def test(epoch):
    model.eval()
    mse = 0.
    for batch_idx, (input, target) in enumerate(test_loader):
        input, target = input.to(device), target.to(device)
        output = model(input)
        mse += F.mse_loss(output, target, size_average=False).item()

        # plot predictions
        if epoch % args.plot_freq == 0 and batch_idx == 0:
            n_samples = 6 if epoch == args.epochs else 2
            idx = torch.randperm(input.size(0))[:n_samples]
            samples_output = output.data.cpu()[idx].numpy()
            samples_target = target.data.cpu()[idx].numpy()

            for i in range(n_samples):
                print('epoch {}: plotting prediction {}'.format(epoch, i))
                plot_prediction_det(args.pred_dir, samples_target[i], 
                    samples_output[i], epoch, i, plot_fn=args.plot_fn)

    rmse_test = np.sqrt(mse / n_out_pixels_test)
    r2_score = 1 - mse / test_stats['y_var']
    print("epoch: {}, test r2-score:  {:.6f}".format(epoch, r2_score))

    if epoch % args.log_freq == 0:
        logger['r2_test'].append(r2_score)
        logger['rmse_test'].append(rmse_test)

def train(epoch):
    model.train()
    mse = 0.
    for _, (input, target) in enumerate(train_loader):
        input, target = input.to(device), target.to(device)
        model.zero_grad()
        output = model(input)
        loss = F.mse_loss(output, target, size_average=False)
        loss.backward()
        optimizer.step()
        mse += loss.item()

    rmse = np.sqrt(mse / n_out_pixels_train)
    scheduler.step(rmse)
    r2_score = 1 - mse / train_stats['y_var']
    print("epoch: {}, training r2-score: {:.6f}".format(epoch, r2_score))
    if epoch % args.log_freq == 0:
        logger['r2_train'].append(r2_score)
        logger['rmse_train'].append(rmse)
    # save model
    if epoch % args.ckpt_freq == 0:
        torch.save(model.state_dict(), args.ckpt_dir + "/model_epoch{}.pth".format(epoch))


print('Start training........................................................')
tic = time()
for epoch in range(1, args.epochs + 1):
    train(epoch)
    with torch.no_grad():
        test(epoch)
tic2 = time()
print("Finished training {} epochs with {} data using {} seconds (including long... plotting time)"
      .format(args.epochs, args.ntrain, tic2 - tic))

x_axis = np.arange(args.log_freq, args.epochs + args.log_freq, args.log_freq)
# plot the rmse, r2-score curve and save them in txt
save_stats(args.train_dir, logger, x_axis)

args.training_time = tic2 - tic
args.n_params, args.n_layers = model._num_parameters_convlayers()
with open(args.run_dir + "/args.txt", 'w') as args_file:
    json.dump(vars(args), args_file, indent=4)

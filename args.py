import argparse
import torch
import json
import random
from pprint import pprint
from utils.misc import mkdirs
import time

# always uses cuda if avaliable

class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='Bayesian Convolutional Encoder-Decoder Networks with SVGD')
        self.add_argument('--post', action='store_true', default=False, help='enables post analysis mode, default is training mode')
        self.add_argument('--debug', action='store_true', default=False, help='enters debug mode, to check GPU memory usage, try different hyperparams, etc.')
        self.add_argument('--exp-name', type=str, default='Bayesian', help='experiment name')
        self.add_argument('--exp-dir', type=str, default="./experiments", help='directory to save experiments')       

        # model
        self.add_argument('-ns', '--n-samples', type=int, default=20, help='(20-30) number of model instances in SVGD')        
        self.add_argument('--blocks', type=list, default=[3, 6, 3], help='list of number of layers in each dense block')
        self.add_argument('--growth-rate', type=int, default=16, help='number of output feature maps of each conv layer within each dense block')
        self.add_argument('--init-features', type=int, default=48, help='number of initial features after the first conv layer')        
        self.add_argument('--drop-rate', type=float, default=0., help='dropout rate')
        self.add_argument('--bn-size', type=int, default=8, help='bottleneck size: bn_size * growth_rate')
        self.add_argument('--bottleneck', action='store_true', default=False, help='enables bottleneck design in the dense blocks')

        # data
        self.add_argument('--data-dir', type=str, default="./dataset", help='directory to dataset')
        self.add_argument('--kle', type=int, default=4225, help='num of KLE terms')
        self.add_argument('--ntrain', type=int, default=128, help="number of training data")
        self.add_argument('--ntest', type=int, default=500, help="number of test data") 
        self.add_argument('--nmc', type=int, default=10000, help="number of MC test data for uncertainty propagation")
        ## more details on dataset
        self.add_argument('--nic', type=int, default=1, help="number of input channels")
        self.add_argument('--noc', type=int, default=3, help="number of output channels")
        self.add_argument('--image-size', type=int, default=[65, 65], help="image resolution")
        
        # training
        self.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
        self.add_argument('--lr', type=float, default=0.0025, help='learnign rate')
        self.add_argument('--lr-noise', type=float, default=0.01, help='learnign rate')
        self.add_argument('--batch-size', type=int, default=16, help='batch size for training')
        self.add_argument('--test-batch-size', type=int, default=250, help='batch size for testing')
        self.add_argument('--mc-batch-size', type=int, default=1000, help='batch size for Monte Carlo')
        self.add_argument('--seed', type=int, default=1, help='manual seed used in Tensor')

        # logging
        self.add_argument('--log-freq', type=int, default=1, help='how many epochs to wait before logging training status')
        self.add_argument('--plot-freq', type=int, default=50, help='how many epochs to wait before plotting test output')
        self.add_argument('--ckpt-freq', type=int, default=100, help='how many epochs to wait before saving model')
        self.add_argument('--ckpt-epoch', type=int, default=None, help='which epoch of checkpoints to be loaded in post mode')        
        self.add_argument('--plot-fn', type=str, default='contourf', choices=['contourf', 'imshow'], help='plotting method')
        

    def parse(self):
        args = self.parse_args()
        
        args.run_dir = args.exp_dir + '/' + args.exp_name \
            + '/kle{}/nsamples{}_ntrain{}_batch{}_lr{}_noiselr{}_epochs{}'.format(
                args.kle, args.n_samples, args.ntrain, args.batch_size, args.lr,
                args.lr_noise, args.epochs)

        args.ckpt_dir = args.run_dir + '/checkpoints'
        mkdirs([args.run_dir, args.ckpt_dir])
        
        assert args.epochs % args.ckpt_freq == 0, 'epochs must'\
            'be dividable by ckpt_freq'
        assert args.ntrain % args.batch_size == 0, 'num of training data must'\
            'be dividable by batch size'

        # seed
        if args.seed is None:
            args.seed = random.randint(1, 10000)
        print("Random Seed: ", args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

        print('Arguments:')
        pprint(vars(args))
        
        if not args.post:
            with open(args.run_dir + "/args.txt", 'w') as args_file:
                json.dump(vars(args), args_file, indent=4)

        return args

# global
args = Parser().parse()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

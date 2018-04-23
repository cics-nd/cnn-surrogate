#!/bin/bash

date

# python ./train_det.py --kle 4225 --ntrain 128
# python ./train_det.py --kle 4225 --ntrain 256
# python ./train_det.py --kle 4225 --ntrain 512
# python ./train_det.py --kle 4225 --ntrain 1024

# python ./train_det.py --kle 500 --ntrain 64
python ./train_det.py --kle 500 --ntrain 128
python ./train_det.py --kle 500 --ntrain 256
python ./train_det.py --kle 500 --ntrain 512

python ./train_det.py --kle 50 --ntrain 32
python ./train_det.py --kle 50 --ntrain 64
python ./train_det.py --kle 50 --ntrain 128
python ./train_det.py --kle 50 --ntrain 256

date
# Bayesian Surrogate as Image-to-Image Regression

[Bayesian Deep Convolutional Encoder-Decoder Networks for Surrogate Modeling and Uncertainty Quantification](https://doi.org/10.1016/j.jcp.2018.04.018)

Yinhao Zhu, [Nicholas Zabaras](https://www.zabaras.com)

PyTorch Implementation of Bayesian surrogate modeling 
for PDEs with high-dimensional stochstic input, such as permeability field, 
Young's modulus, etc.

KLE4225 (No dim reduction) | KLE500 | KLE50
:-----:|:------:|:-----:
![](images/kle4225_pred_at_x_312_n512.png?raw=true) | ![](images/kle500_pred_at_x_293_n256.png?raw=true) | ![](images/kle50_pred_at_x_47_n128.png?raw=true)

The three columns in each image correspond to three output fields:
(1) pressure field, (2)(3) velocities field in x and y axies.
The four rows in each image from top to bottom show: (1) simulator output, 
(2) surrogate prediction (predictive mean), (3) error between simulator and prediction, 
(4) two standard devivation of prediction. 

## Dependencies
- python 3 or 2
- PyTorch 0.3.1
- h5py


## Installation
- Install PyTorch and other dependencies

- Clone this repo:
```
git clone https://github.com/cics-nd/cnn-surrogate.git
cd cnn-surrogate
```

## Dataset
Download Darcy flow simulation dataset with input Gaussian random field (with 
exponential kernel) and three output fields (pressure and velocities fields).

To download KLE4225 dataset, 
```bash
bash ./scripts/download_dataset.sh 4225
```
Change 4225 to 500, 50 to download KLE500, KLE50 datasets. 
Data is saved at `./dataset/`.

Number of training data within each dataset:
* KLE50: [32, 64, 128, 256],
* KLE500: [64, 128, 256, 512],
* KLE4225: [128, 256, 512, 1024].

Number of test data: 500.
Number of Monte Carlo sampling data: 10,000.

KLE4225 | KLE500 | KLE50
:-----:|:------:|:-----:
![](images/kle4225_data_312.png?raw=true) | ![](images/kle500_data_293.png?raw=true) | ![](images/kle50_data_47.png?raw=true)

## Training

### Deterministic Surrogate 

Training non-Bayesian surroagte (e.g. with 512 data from KLE4225 dataset):
```
python train_det.py --kle 4225 --ntrain 512
```
The runs are saved at `./experiments/`.

### Bayesian Surrogate: coming soon...


## Citation

If you find this repo useful for your research, please consider to cite:
```latex
@article{ZHU2018415,
title = "Bayesian deep convolutional encoder–decoder networks for surrogate modeling and uncertainty quantification",
journal = "Journal of Computational Physics",
volume = "366",
pages = "415 - 447",
year = "2018",
issn = "0021-9991",
doi = "https://doi.org/10.1016/j.jcp.2018.04.018",
url = "http://www.sciencedirect.com/science/article/pii/S0021999118302341",
author = "Yinhao Zhu and Nicholas Zabaras"
}
```

## Acknowledgments

Thanks Dr. [Steven Atkinson](https://scholar.google.com/citations?user=MVrxtDkAAAAJ&hl=en) for providing Darcy flow simulation dataset.

Code is inspired by [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

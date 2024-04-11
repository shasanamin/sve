# Data Subset Selection
RandE implementation available in `trainers/rande_trainer.py`. For CoRand implementation (Coreset-based method initially, followed by a Random variant for selection; Appendix B.4), see `trainers/crest_rand_trainer.py` and `trainers/crest_rande_trainer.py`

## Installation
This code is tested with Python 3.8.8 and PyTorch 1.9.1 with CUDA 11.5.

To install the required packages, run
```
pip install -r requirements.txt
```

## Usage
```
python train.py --selection_method=rande --dataset=cifar10 --train_frac=0.1
```

`--dataset`: The dataset to use. (default: `cifar10`)
- `cifar10`: CIFAR-10 dataset
- `cifar100`: CIFAR-100 dataset
- `tinyimagenet`: TinyImageNet dataset

:warning: The TinyImageNet dataset is not included in this repository. Please download the dataset from [here](https://www.kaggle.com/c/tiny-imagenet).

`--data_dir`: The directory to store the dataset. (default: `./data`)

`--arch`: The model architecture to use. (default: `resnet20`)
- `resnet20`: ResNet-20 model for CIFAR-10
- `resnet18`: ResNet-18 model for CIFAR-100
- `resnet50`: ResNet-50 model for TinyImageNet

`--seed`: The random seed to use. (default: `0`)

`--selection_method`: The data selection method to use. (default: `random`)

`--train_frac`: The fractrion of training steps to use compared to full training. (default: `0.1`)

See `arguments.py` for more details.

## Adding New Datasets
To add a new dataset, you need to add the dataset loading code in `datasets/dataset.py`. 
Then, you need to add the dataset name to the choices of `--dataset` argument in `utils/arguments.py`.

## Adding New Models
To add a new model, you need to create a new file in `models/` folder, which contains the model class. 
For example, `models/resnet.py` contains the class `ResNet20`, which is the ResNet-20 model. 
Then, you need to add the model name to the choices of `--arch` argument in `utils/arguments.py`.

## Adding New Data Selection Methods
To add a new data selection method, you need to create a new file in `trainers/` folder, which contains a subclass of `SubsetTrainer` class defined in `trainers/subset_trainer.py`.
For example, `trainers/random_trainer.py` contains the class `RandomTrainer`, which is the trainer for random selection. 
Then, you need to add the method name to the choices of `--selection_method` argument in `utils/arguments.py`.

## Acknowledgement
The code is based on [CREST](https://github.com/bigml-cs-ucla/crest), [Craig](https://github.com/baharanm/craig) and [AdaHessian](https://github.com/amirgholami/adahessian).
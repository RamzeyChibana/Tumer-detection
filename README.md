Title
===
This repository is dedicated to experimenting with various models and techniques for predicting outcomes from medical images. It includes multiple approaches to preprocessing, training, and evaluating models for enhanced prediction accuracy.


## Install & Dependence
- python
- pytorch
- numpy

## Dataset Preparation
| Dataset | link |
| ---     | ---   |
| Brain Tumor MRI Dataset | [visit](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) |


## Use
- for train
  ```
  python train.py
  ```
- for test
  ```
  python test.py
  ```
important arguments for **main.py** : 
* `--batch_size` batch size of nodes
* `--epochs` run the model of how many epochs
* `--learning_rate` learning rate 
* `--layers` Num of layers and theire dimension
* `--device` device to train with { cpu / gpu }
* `--verbose` How much more infos in epoch 0:nothing , 1:loss , 2:evaluation 
* `--exp` Continue the Training of existing Experiement


## Directory Hierarchy
```
|—— .gitignore
|—— Experiments
|—— main.py
|—— models.py
|—— test.ipynb
|—— Tr-no_0011.jpg
|—— utils
|    |—— DataPre.py
|    |—— load_data.py
|    |—— parser.py
```

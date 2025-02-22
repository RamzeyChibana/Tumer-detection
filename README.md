Tumer-detection
===
The repository focuses on detecting brain tumors from medical images using a combination of image descriptors and deep learning techniques. It incorporates various image preprocessing methods to extract relevant features and applies deep learning models for accurate tumor detection


## Install & Dependence
- python
- pytorch
- numpy

## Dataset Preparation
| Dataset | link |
| ---     | ---   |
| Brain Tumor MRI Dataset | [visit](https://drive.google.com/file/d/1ciJ0qX-YOVXc7oLWTQhqR3avy2iOuC-l/view?usp=sharing) |


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
|—— Tr-no_0011.jpg
|—— utils
|    |—— DataPre.py
|    |—— load_data.py
|    |—— parser.py
|___
```

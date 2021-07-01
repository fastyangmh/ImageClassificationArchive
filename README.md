# ImageClassification

[繁體中文](README_zh_TW.md)

**ImageClassification** is a general image classification project that helps users quickly complete image classification tasks to reduce development time. It provides features such as training, prediction, evaluation, and hyperparameter optimization. Users only need to create a dataset in a specified format, that is Can quickly complete image classification tasks.

This project is developed and tested under macOS >=10.15.7, Ubuntu >=20.04, Python 64-bit >= 3.7. If you encounter any problems in use, please [file an issue](https://github.com/fastyangmh/ImageClassification/issues) on GitHub.

## **Features**

This project has provided the following features:
- [x] Early stopping  
- [x] Data balance by the loss function  
- [x] Pre-defined dataset
- [x] Various SOTA model
- [x] Learning rate scheduler
- [x] Hyperparameter optimization
- [x] K-fold cross-validation
- [x] The profiler during training for identifying bottleneck
- [x] Transfer learning
- [x] GUI while predicting
- [x] Automatic mixed-precision

## **Installation**

The following packages are used in this project:
```
argparse
timm
torch
numpy
pytorch-lightning
torchmetrics
pandas
Pillow
glob2
scikit-learn
tqdm
ray
matplotlib
ruamel.yaml
torchvision
```

Use the following command to automatically install the required packages for the project and update to the latest version:
```bash
pip install -Ur requirements.txt
```
If there is a package dependency problem that prevents installation, please use the following command:
```bash
pip install -Ur requirements.txt --force-reinstall
```

## **Parameters**

All parameters of this project are controlled by `argparse`. To view all parameters and descriptions, please use the following command:
```bash
python main.py -h
```

The following are the necessary input parameters when executing any mode:
```bash
--mode
--data_path
--classes
--backbone_model
```

In addition to using `argparse`, it can also be controlled by `YAML`. The file location is `config/parameters.yaml`. **Note**. If both `argparse` and `YAML` are used at the same time, it will cause coverage problems, so the system will use ` YAML` overrides `argparse`.

## **Dataset format**

The dataset directory should follow the following format, where `train` is used to train the model, `val` is used to verify the model, and `test` is used to test the model:
```
example/
├── train
│   ├── class1
│   │   ├── image1.png
│   │   ├── image2.png
│   │   └── image3.png
│   └── class2
│       ├── image1.png
│       ├── image2.png
│       └── image3.png
├── val
│   ├── class1
│   │   ├── image1.png
│   │   ├── image2.png
│   │   └── image3.png
│   └── class2
│       ├── image1.png
│       ├── image2.png
│       └── image3.png
└── test
    ├── class1
    │   ├── image1.png
    │   ├── image2.png
    │   └── image3.png
    └── class2
        ├── image1.png
        ├── image2.png
        └── image3.png
```

If there are too many dataset classes, you can use `Text file` to save the name of the dataset class. **Note**, please follow the format below:
```
class1
class2
```

## **Classification model**

This project uses [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models) to create a classification model. The following command will list the classification models available in `--backbone_model`, or Go to [Model Summaries](https://rwightman.github.io/pytorch-image-models/models/) to query,
**Note** If the selected classification model provides an ImageNet pre-trained model, it will be automatically adopted, otherwise it will be initialized with  random initialization:
```bash
python main.py -h
```

In addition, a self-defined classification model is provided, please refer to `src/SelfDefinedModel.py`. If a self-defined model is provided, please **note** the following:
* `class` name must be the same as the file name
* Do not add `Softmax` to the self-defined model, so as not to double calculate the probability and cause the result to be lower than expected

## **Training**

Use the following command to train a custom dataset, where `$DATA_PATH` is the path of the custom dataset, `$CLASS1,$CLASS2` is the class name of the dataset, and `$BACKBONE_MODEL` is [PyTorch Image Models](https (://github.com/rwightman/pytorch-image-models) the selected classification model:
```bash
python main.py --mode train --data_path $DATA_PATH --classes $CLASS1,$CLASS2 --backbone_model $BACKBONE_MODEL
```

If you use `Text file` to save the dataset class name, set `$CLASS1,$CLASS2` to the file path of `Text file`, as in the following example:
```bash
python main.py --mode train --data_path $DATA_PATH --classes CLASSES.txt --backbone_model $BACKBONE_MODEL  #CLASSES.txt is the file path of the Text file
```

If you use a self-defined model, please set `$BACKBONE_MODEL` to the file path of the self-defined model, as in the following example:
```bash
python main.py --mode train --data_path $DATA_PATH --classes $CLASS1,$CLASS2 --backbone_model src/SelfDefinedModel.py #src/SelfDefinedModel.py is the file path of the self-defined model
```

This project provides default datasets `MNIST` and `CIFAR-10`. The datasets are provided by the `PyTorch` framework and will be automatically downloaded to the specified `$DATA_PATH`. Please execute the following command:
```bash
python main.py --mode train --data_path $DATA_PATH --predefined_dataset MNIST --classes None --in_chans 1 --backbone_model $BACKBONE_MODEL #use the MNIST dataset, there is no need to provide classes. Note that the MNIST dataset is a grayscale image, so the input channel needs to be set to 1

python main.py --mode train --data_path $DATA_PATH --predefined_dataset CIFAR10 --classes None --backbone_model $BACKBONE_MODEL #use CIFAR-10 dataset, no need to provide classes
```

This project provides transfer learning of the same classification model, so if you have the pre-trained model weights, please use the following command. **Note** If the number of input channels is different, an error will occur:
```bash
python main.py --mode train --data_path $DATA_PATH --classes $CLASS1,$CLASS2 --backbone_model $BACKBONE_MODEL --checkpoint_path $CHECKPOINT_PATH   #$CHECKPOINT_PATH is the path of the pre-trained model weight
```

During the training process, a working directory will be created to store the information of `TensorBoard` for users to confirm the training status, and store model weights according to the best validation set accuracy. The default storage location is `save/`, if you want to change the default save The location, please use the following command to change:
```bash
python main.py --mode train --data_path $DATA_PATH --classes $CLASS1,$CLASS2 --backbone_model $BACKBONE_MODEL --save_path $SAVE_PATH   #$SAVE_PATH is the storage path
```

The creation name of the working directory is `$version_*`, `*` is the number of times to create training, if it is 0 for the first time, it is 1 for the second time, and so on. In addition, to read the information of `TensorBoard`, please follow the command below:
```bash
tensorboard --logdir=$SAVE_PATH/lightning_logs/$version_*/ #$SAVE_PATH is the storage path, $version_* is the number of times to create training
```

After the training is completed, the weight storage location of the model is:
```bash
$SAVE_PATH/lightning_logs/$version_*/checkpoints/  #$SAVE_PATH is the storage path, $version_* is the training process you want to confirm
```

## **Prediction**

Use the following commands to predict data based on the trained model:
```bash
python main.py --mode predict --data_path $DATA_PATH --classes $CLASS1,$CLASS2 --backbone_model $BACKBONE_MODEL --checkpoint_path $CHECKPOINT_PATH #$DATA_PATH is the single file or directory to be predicted, and $CHECKPOINT_PATH is the path of the pre-trained model weight
```

The dataset directory for predicting should follow the following format:
```bash
predict/
└── predict
    ├── image1.png
    ├── image2.png
    └── image3.png
```

This project provides a GUI for users to make predictions, which can be used through the following commands:
```bash
python main.py --mode predict --data_path $DATA_PATH --classes $CLASS1,$CLASS2 --backbone_model $BACKBONE_MODEL --checkpoint_path $CHECKPOINT_PATH --gui #$DATA_PATH is the single file or directory to be predicted
```

## **Evaluation**

Use the following command to evaluate the model for k-fold cross-validation. **Note** This command will create a dataset named `k_fold_dataset_*` in the project directory. Please do not delete the dataset and ensure sufficient storage space. It will be deleted automatically after completion:
```bash
python main.py --mode evaluate --data_path $DATA_PATH --classes $CLASS1,$CLASS2 --backbone_model $BACKBONE_MODEL --n_splits $N_SPLITS  #$N_SPLITS is the number of folds
```

## **Hyperparameter optimization**

Use the following commands to execute the hyperparameter optimization:
```bash
python main.py --mode tune --data_path $DATA_PATH --classes $CLASS1,$CLASS2 --backbone_model $BACKBONE_MODEL
```

## **Configuration files**

This project provides the following configuration files for users to quickly modify parameters:
```bash
parameters.yaml         #all the parameters of this project
transform.yaml          #methods used in data augmentation
optimizer.yaml          #optimizer for train the model
hyperparameter.yaml     #parameter space for hyperparameter optimization
```

### **parameters.yaml**

This project can modify all the parameters of this project by rewriting `parameters.yaml`. Please rewrite according to the `YAML` syntax. To use this configuration file, please use the following command:
```bash
python main.py --mode train --data_path $DATA_PATH --classes $CLASS1,$CLASS2 --backbone_model $BACKBONE_MODEL --parameters_config_path $PARAMETERS_CONFIG_PATH #$PARAMETERS_CONFIG_PATH is the file path
```

### **transform.yaml**

This project can be augmented by modifying the data used in training, verification, testing, and prediction by rewriting `transform.yaml`. Please rewrite it according to the `YAML` syntax and use it with the following commands. **Note** This project uses the settings of `config/transform.yaml` for data augmentation by default:
```bash
python main.py --mode train --data_path $DATA_PATH --classes $CLASS1,$CLASS2 --backbone_model $BACKBONE_MODEL --transform_config_path $TRANSFORM_CONFIG_PATH #$TRANSFORM_CONFIG_PATH is the file path
```

When modifying `transform.yaml`, please follow the following format. **Note** If the image is grayscale, in order to prevent errors in the input channel, you can add `Grayscale` to convert the image into a grayscale image. If the image itself is grayscale When you add it, it has no effect:
```yaml
train:
  #data augmentation for training

val:
  #data augmentation for validation

test:
  #data augmentation for test

predict:
  #data augmentation for prediction
```

### **optimizer.yaml**

This project can modify the optimizer used during training by rewriting `optimizer.yaml`. Please rewrite it according to the `YAML` syntax and use the following commands. **Note** This project uses `config/optimizer.yaml` by default. The settings to create an optimizer:
```bash
python main.py --mode train --data_path $DATA_PATH --classes $CLASS1,$CLASS2 --backbone_model $BACKBONE_MODEL --optimizer_config_path $OPTIMIZER_CONFIG_PATH #$OPTIMIZER_CONFIG_PATH is the file path
```

### **hyperparameter.yaml**

This project can modify the parameter space during hyperparameter optimization by rewriting `hyperparameter.yaml`. The parameters of the parameter space must exist in `src/project_parameters.py`, please rewrite it according to the `YAML` syntax, and use the following commands , **Note** This project uses the settings of `config/hyperparameter.yaml` for hyperparameter optimization by default:
```bash
python main.py --mode train --data_path $DATA_PATH --classes $CLASS1,$CLASS2 --backbone_model $BACKBONE_MODEL --hyperparameter_config_path $HYPERPARAMETER_CONFIG_PATH #$HYPERPARAMETER_CONFIG_PATH is the file path
```

When modifying `hyperparameter.yaml`, please follow the following format:
```yaml
int:    #integer space such as epochs, etc
  train_iter:
    - 10
    - 200
  step_size:
    - 1
    - 10
float:  #floating point number space such as learning rate, etc
  lr:
    - 1e-4
    - 1e-1
choice: #selection space such as learning rate scheduler, etc.
  lr_scheduler:
    - CosineAnnealingLR
    - StepLR
```

## **Pre-trained weight**

This project provides pre-trained models of `MNIST` and `CIFAR-10`. The classification model uses `mnasnet_small`. The model parameters are about `761 K`. The training parameters and results are listed below:
```bash
#MNIST, other parameters are default values
#training dataset accuracy: 93.6107878989%
#validation dataset accuracy: 93.072219193%
#test dataset accuracy: 97.1892234683%
python main.py --mode train --data_path data/ --predefined_dataset MNIST --classes None --backbone_model mnasnet_small --in_chans 1 --batch_size 512 --lr 0.019965407596926204 --lr_scheduler StepLR --step_size 9 --train_iter 112

#CIFAR10, other parameters are default values
#training dataset accuracy: 62.1069026899%
#validation dataset accuracy: 61.1735984683%
#test dataset accuracy: 61.9376149774%
python main.py --mode train --data_path data/ --predefined_dataset CIFAR10 --classes None --backbone_model mnasnet_small --batch_size 512 --lr 0.03188473363097435 --lr_scheduler CosineAnnealingLR --step_size 2 --train_iter 190
```

## **Tool**

This project provides tools to convert the default datasets `MNIST` and `CIFAR-10` into `png` image format, and use the following command to convert the specified dataset to `png` image format, where `data_path` is the dataset the path of `MNIST` and `CIFAR-10`, **Note** You must first execute the main program `main.py` and download the dataset before you can execute this tool:
```python
#convert MNIST to png image
from src.utils import pytorch_mnist_to_png
data_path = 'data/MNIST/'
pytorch_mnist_to_png(data_path=data_path)

#convert CIFAR-10 to png image
from src.utils import pytorch_cifar10_to_png
data_path = 'data/MNIST/'
pytorch_cifar10_to_png(data_path=data_path)
```

## **License**

The entire codebase is under [MIT license](LICENSE).
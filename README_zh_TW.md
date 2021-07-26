# ImageClassification

[English](README.md)

**ImageClassification**是一個通用影像分類的專案，幫助使用者快速完成影像分類任務減少開發時間，提供訓練、預測、評估、超參數優化等功能，使用者只需建立指定格式的資料集，即可快速完成影像分類任務。

本專案在macOS >=10.15.7, Ubuntu >=20.04, Python 64-bit >= 3.7下完成開發與測試，若使用上遇到任何問題，請在GitHub上[提出 issue](https://github.com/fastyangmh/ImageClassification/issues)。

## **專案功能**

本專案已提供下列功能：
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

## **安裝套件**

本專案使用以下套件：
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

透過以下指令，即可自動安裝專案所需套件並更新至最新版本：
```bash
pip install -Ur requirements.txt
```
若有套件相依問題導致無法安裝，請透過下列指令：
```bash
pip install -Ur requirements.txt --force-reinstall
```

## **專案參數**

本專案所有參數透過`argparse`控制，若需查看所有參數與說明，請透過以下指令：
```bash
python main.py -h
```

以下是執行任何模式時，必要的輸入參數：
```bash
--mode
--data_path
--classes
--backbone_model
```

除了使用`argparse`也可透過`YAML`控制，檔案位置為`config/parameters.yaml`，**注意**，若同時使用`argparse`與`YAML`會產生覆蓋問題，因此系統會以`YAML`覆蓋`argparse`。

## **資料集格式**

資料集目錄請遵循以下格式，其中`train`用於訓練模型，`val`用於驗證模型，`test`用於測試模型：
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

若資料集類別數量過多，可使用`Text file`儲存資料集類別名稱，**注意**，請遵循以下格式：
```
class1
class2
```

## **分類模型**

本專案利用[PyTorch Image Models](https://github.com/rwightman/pytorch-image-models)建立分類模型，透過以下指令將會列出在`--backbone_model`中可用的分類模型，或是前往[Model Summaries](https://rwightman.github.io/pytorch-image-models/models/)查詢，
**注意**若選用的分類模型有提供ImageNet的預訓練模型，將會自動採用，否則將以隨機初始化：
```bash
python main.py -h
```

此外提供自定義分類模型，請參考`src/SelfDefinedModel.py`，若提供自定義模型，請**注意**以下事項：
* `class`名稱請與檔名相同
* 請勿在自定義模型中加入`Softmax`，避免重覆計算機率值導致結果不如預期

## **訓練**

透過以下指令，訓練自定義的資料集，其中`$DATA_PATH`為自定義的資料集路徑，`$CLASS1,$CLASS2`為資料集的類別名稱，`$BACKBONE_MODEL`為[PyTorch Image Models](https://github.com/rwightman/pytorch-image-models)中選用的分類模型：
```bash
python main.py --mode train --data_path $DATA_PATH --classes $CLASS1,$CLASS2 --backbone_model $BACKBONE_MODEL
```

若透過`Text file`儲存資料集類別名稱，請將`$CLASS1,$CLASS2`設為`Text file`的檔案路徑，如以下範例：
```bash
python main.py --mode train --data_path $DATA_PATH --classes CLASSES.txt --backbone_model $BACKBONE_MODEL  #CLASSES.txt為Text file的檔案路徑
```

若使用自定義模型，請將`$BACKBONE_MODEL`設為自定義模型的檔案路徑，如以下範例：
```bash
python main.py --mode train --data_path $DATA_PATH --classes $CLASS1,$CLASS2 --backbone_model src/SelfDefinedModel.py #src/SelfDefinedModel.py為自定義模型的檔案路徑
```

本專案提供預設資料集`MNIST`與`CIFAR-10`，資料集由`PyTorch`框架提供，將會自動下載至指定的`$DATA_PATH`，請透過以下指令執行：
```bash
python main.py --mode train --data_path $DATA_PATH --predefined_dataset MNIST --classes None --in_chans 1 --backbone_model $BACKBONE_MODEL #使用MNIST資料集，無需提供classes，注意MNIST資料集為灰階圖片，故需將輸入通道設為1

python main.py --mode train --data_path $DATA_PATH --predefined_dataset CIFAR10 --classes None --backbone_model $BACKBONE_MODEL #使用CIFAR-10資料集，無需提供classes
```

本專案提供相同分類模型的遷移學習，因此若擁有訓練好的模型權重，請透過以下指令，**注意**若輸入通道數不同將會發生錯誤：
```bash
python main.py --mode train --data_path $DATA_PATH --classes $CLASS1,$CLASS2 --backbone_model $BACKBONE_MODEL --checkpoint_path $CHECKPOINT_PATH   #$CHECKPOINT_PATH為先前訓練好的模型權重路徑
```

訓練過程中，將會建立工作目錄儲存`TensorBoard`的資訊供使用者確認訓練狀態，並根據最佳驗證集準確率儲存模型權重，預設儲存位置為`save/`，若要變更預設儲存位置，請透過以下指令變更：
```bash
python main.py --mode train --data_path $DATA_PATH --classes $CLASS1,$CLASS2 --backbone_model $BACKBONE_MODEL --save_path $SAVE_PATH   #$SAVE_PATH為儲存路徑
```

工作目錄的建立名稱為`$version_*`，`*`為第幾次建立訓練，若第一次為0，若第二次則為1，依此類推。此外若要讀取`TensorBoard`的資訊請根據以下指令：
```bash
tensorboard --logdir=$SAVE_PATH/lightning_logs/$version_*/ #$SAVE_PATH為儲存路徑，$version_*為第幾次建立訓練
```

訓練完成後，模型的權重儲存位置為：
```bash
$SAVE_PATH/lightning_logs/$version_*/checkpoints/  #$SAVE_PATH為儲存路徑，$version_*為想確認的訓練過程
```

## **預測**

透過以下指令，根據訓練好的模型進行資料預測：
```bash
python main.py --mode predict --data_path $DATA_PATH --classes $CLASS1,$CLASS2 --backbone_model $BACKBONE_MODEL --checkpoint_path $CHECKPOINT_PATH #$DATA_PATH為待預測之單一檔案或目錄，$CHECKPOINT_PATH為先前訓練好的模型權重路徑
```

預測時的資料集目錄請遵循下列格式：
```bash
predict/
└── predict
    ├── image1.png
    ├── image2.png
    └── image3.png
```

本專案有提供GUI供使用者做預測，透過以下指令使用：
```bash
python main.py --mode predict --data_path $DATA_PATH --classes $CLASS1,$CLASS2 --backbone_model $BACKBONE_MODEL --checkpoint_path $CHECKPOINT_PATH --gui #$DATA_PATH為待預測之單一檔案或目錄
```

## **評估**

透過以下指令對模型做k-fold cross-validation的評估，**注意**該指令會在專案目錄下建立資料集名稱為`k_fold_dataset_*`，請勿刪除該資料集並確保儲存空間足夠，評估完成後將會自動刪除：
```bash
python main.py --mode evaluate --data_path $DATA_PATH --classes $CLASS1,$CLASS2 --backbone_model $BACKBONE_MODEL --n_splits $N_SPLITS  #$N_SPLITS為k折次數
```

## **超參數優化**

透過以下指令，進行超參數優化：
```bash
python main.py --mode tune --data_path $DATA_PATH --classes $CLASS1,$CLASS2 --backbone_model $BACKBONE_MODEL
```

## **設定檔**

本專案提供以下設定檔供使用者快速改寫參數：
```bash
parameters.yaml         #本專案所有的參數
transform.yaml          #資料擴增時用到的方法
optimizer.yaml          #用於模型訓練時的優化器
hyperparameter.yaml     #用於超參數優化時的參數空間
```

### **parameters.yaml**

本專案可透過改寫`parameters.yaml`修改本專案所有參數，請依據`YAML`語法做改寫，若要使用該設定檔，請透過以下指令：
```bash
python main.py --mode train --data_path $DATA_PATH --classes $CLASS1,$CLASS2 --backbone_model $BACKBONE_MODEL --parameters_config_path $PARAMETERS_CONFIG_PATH #$PARAMETERS_CONFIG_PATH為檔案路徑
```

### **transform.yaml**

本專案可透過改寫`transform.yaml`修改訓練、驗證、測試、預測時使用的資料擴增。請依據`YAML`語法做改寫，並透過以下指令使用，**注意**本專案預設使用`config/transform.yaml`的設定做資料擴增：
```bash
python main.py --mode train --data_path $DATA_PATH --classes $CLASS1,$CLASS2 --backbone_model $BACKBONE_MODEL --transform_config_path $TRANSFORM_CONFIG_PATH #$TRANSFORM_CONFIG_PATH為檔案路徑
```

修改`transform.yaml`時，請遵循下列格式，**注意**若圖片為灰階時，為防止輸入通道出錯，可加入`Grayscale`，將圖片轉成灰階圖，若本身圖片為灰階時，加入則無影響：
```yaml
train:
  #訓練用的資料擴增

val:
  #驗證用的資料擴增

test:
  #測試用的資料擴增

predict:
  #預測用的資料擴增
```

### **optimizer.yaml**

本專案可透過改寫`optimizer.yaml`修改訓練時使用的優化器，請依據`YAML`語法做改寫，並透過以下指令使用，**注意**本專案預設使用`config/optimizer.yaml`的設定建立優化器：
```bash
python main.py --mode train --data_path $DATA_PATH --classes $CLASS1,$CLASS2 --backbone_model $BACKBONE_MODEL --optimizer_config_path $OPTIMIZER_CONFIG_PATH #$OPTIMIZER_CONFIG_PATH為檔案路徑
```

### **hyperparameter.yaml**

本專案可透過改寫`hyperparameter.yaml`修改超參數優化時的參數空間，該參數空間的參數需存在於`src/project_parameters.py`中，請依據`YAML`語法做改寫，並透過以下指令使用，**注意**本專案預設使用`config/hyperparameter.yaml`的設定做超參數優化：
```bash
python main.py --mode train --data_path $DATA_PATH --classes $CLASS1,$CLASS2 --backbone_model $BACKBONE_MODEL --hyperparameter_config_path $HYPERPARAMETER_CONFIG_PATH #$HYPERPARAMETER_CONFIG_PATH為檔案路徑
```

修改`hyperparameter.yaml`時，請遵循下列格式：
```yaml
int:    #整數空間如訓練次數，學習率下降次數等
  train_iter:
    - 10
    - 200
  step_size:
    - 1
    - 10
float:  #浮點數空間如學習率等
  lr:
    - 1e-4
    - 1e-1
choice: #選擇空間如學習率下降器等
  lr_scheduler:
    - CosineAnnealingLR
    - StepLR
```

## **預訓練權重**

本專案提供`MNIST`與`CIFAR-10`的預訓練模型，分類模型使用`mnasnet_small`，該模型參數量約為`761 K`，以下列出訓練的參數與結果：
```bash
#MNIST，其餘參數皆為預設值
#訓練集準確率：93.6107878989%
#驗證集準確率：93.072219193%
#測試集準確率：97.1892234683%
python main.py --mode train --data_path data/ --predefined_dataset MNIST --classes None --backbone_model mnasnet_small --in_chans 1 --batch_size 512 --lr 0.019965407596926204 --lr_scheduler StepLR --step_size 9 --train_iter 112

#CIFAR10，其餘參數皆為預設值
#訓練集準確率：62.1069026899%
#驗證集準確率：61.1735984683%
#測試集準確率：61.9376149774%
python main.py --mode train --data_path data/ --predefined_dataset CIFAR10 --classes None --backbone_model mnasnet_small --batch_size 512 --lr 0.03188473363097435 --lr_scheduler CosineAnnealingLR --step_size 2 --train_iter 190
```

## **工具**

本專案提供工具將預設資料集`MNIST`與`CIFAR-10`轉換成`png`圖片格式，透過以下指令將指定的資料集轉換成`png`圖片格式，其中`data_path`為資料集`MNIST`與`CIFAR-10`的路徑，**注意**須先執行過主程式`main.py`下載過資料集才可執行該工具：
```python
#將MNIST轉換成png圖片
from src.utils import pytorch_mnist_to_png
data_path = 'data/MNIST/'
pytorch_mnist_to_png(data_path=data_path)

#將CIFAR-10轉換成png圖片
from src.utils import pytorch_cifar10_to_png
data_path = 'data/CIFAR10/'
pytorch_cifar10_to_png(data_path=data_path)
```

## **授權**

本專案遵循[MIT授權協議](LICENSE)。
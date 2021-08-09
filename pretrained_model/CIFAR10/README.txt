The CIFAR10_mnasnet_small_checkpoint.ckpt is used the following parameters to train.

python main.py --mode train --data_path data/ --predefined_dataset CIFAR10 --classes None --in_chans 3 --batch_size 128 --backbone_model mnasnet_small --num_workers 4 --train_iter 144 --step_size 2 --lr 0.0002921712874576203 --alpha 0.6796752976738497 --lr_scheduler CosineAnnealingLR --loss_function BCELoss

The CIFAR10_mnasnet_small_checkpoint.ckpt training result is the following.

the train dataset confusion matrix:
            airplane  automobile  bird   cat  deer   dog  frog  horse  ship  truck
airplane        2118         229   329   124    39   114    42    155   617    216
automobile        82        2547    59    92     8    53   126    132   162    702
bird             298         167  1785   295   325   288   367    297   142     49
cat              115         215   283  1475   131   714   484    291   180    128
deer             120          47   555   284  1408   202   660    518   136     53
dog               95         208   299   783   131  1559   213    417   170     87
frog              29          82   196   356   203   134  2812    133    53     61
horse            108         238   233   320   199   334   163   2185    94    155
ship             326         263   117   144    29    88    49     57  2703    226
truck            135         825    47   133    13    58   129    192   222   2236
--------------------------------------------------------------------------------
test accuracy: 0.520592052715655, test loss: 0.3128224863602331
--------------------------------------------------------------------------------

test the val dataset
the val dataset confusion matrix:
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
            airplane  automobile  bird  cat  deer  dog  frog  horse  ship  truck
airplane         552          52    87   29     8   26    14     37   174     38
automobile        27         642    12   23     3   12    31     29    56    202
bird              81          41   402   69    86   74    91     87    37     19
cat               27          49    84  328    29  185   129     76    53     24
deer              26          11   134   69   353   57   178    154    23     12
dog               23          46    82  186    43  418    60    109    44     27
frog               7          17    59   75    57   38   619     34    15     20
horse             26          56    65   59    52   75    44    528    25     41
ship              90          64    26   41     7   23    13     21   636     77
truck             20         240    14   44     2    7    30     56    59    538
--------------------------------------------------------------------------------
test accuracy: 0.5015822784810127, test loss: 0.3134946626952932
--------------------------------------------------------------------------------

test the test dataset
the test dataset confusion matrix:
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
            airplane  automobile  bird  cat  deer  dog  frog  horse  ship  truck
airplane         529          37   105   27     5   40     6     24   185     42
automobile        21         612    11   32     2   37    21     30    63    171
bird              77          30   422   69    86  120    88     63    35     10
cat               22          26    75  368    25  252   101     53    50     28
deer              33          10   114   79   419   70   126    110    28     11
dog               24          23    51  169    27  550    25     78    42     11
frog               1          15    59   84    48   54   689     28     7     15
horse             33          37    42   68    58  155    32    501    35     39
ship              58          51    27   32     4   42     9     15   717     45
truck             37         186     9   32     3   27    29     39    93    545
--------------------------------------------------------------------------------
test accuracy: 0.5361946202531646, test loss: 0.31253164252148397
--------------------------------------------------------------------------------
The CIFAR10_mnasnet_small_checkpoint.ckpt is used the following parameters to train.

python main.py --mode train --data_path data/ --predefined_dataset CIFAR10 --classes None --backbone_model mnasnet_small --batch_size 512 --num_workers 5 --train_iter 108 --step_size 2 --lr 0.006973673925118025 --lr_scheduler CosineAnnealingLR

The CIFAR10_mnasnet_small_checkpoint.ckpt training result is the following.

the train dataset confusion matrix:
            airplane  automobile  bird   cat  deer   dog  frog  horse  ship  truck
airplane        3154          70   140    40    64    42    20    117   242     94
automobile        59        3348    35    20     3    46    25     53    87    287
bird             253          25  2551   143   377   204   192    182    53     33
cat               96          48   250  1873   206   929   216    247    67     84
deer              71           3   265   113  2840   117   208    300    32     34
dog               50          34   121   371   171  2794    73    255    59     34
frog              25          14   136   177   157   100  3322     50    33     45
horse             54          24    90    98   247   188    36   3198    34     60
ship             257          98    56    32    23    41    27     45  3366     57
truck            100         442    32    32    17    78    34    142   106   3007
--------------------------------------------------------------------------------
test accuracy: 0.7356111550632911, test loss: 0.7510817699794528
--------------------------------------------------------------------------------

test the val dataset
the val dataset confusion matrix:
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
            airplane  automobile  bird  cat  deer  dog  frog  horse  ship  truck
airplane         796          18    40   12    27   15     6     14    67     22
automobile        20         862     9    3     2    8    11      8    27     87
bird              78           4   566   41    99   59    55     56    15     14
cat               21          15    63  433    44  242    64     78    10     14
deer              13           1    64   43   701   33    63     82    10      7
dog               18          12    42   83    52  707    21     82    14      7
frog               3           3    35   47    48   32   725     18     8     22
horse             12           6    18   26    58   53    15    763    11      9
ship              64          36    24   15     6   11     7      8   809     18
truck             17         111     8   10     6   17     8     40    25    768
--------------------------------------------------------------------------------
test accuracy: 0.7126608461141586, test loss: 0.838104110956192
--------------------------------------------------------------------------------

test the test dataset
the test dataset confusion matrix:
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
            airplane  automobile  bird  cat  deer  dog  frog  horse  ship  truck
airplane         761          15    51   14    19    8     4     27    81     20
automobile        19         835     5    4     0   14     7     10    18     88
bird              71          10   656   42    81   57    41     31     7      4
cat               27           6    73  479    55  229    55     47    14     15
deer              17           1    82   27   695   35    63     68     7      5
dog               15           5    35   96    40  719    18     47    12     13
frog               5           4    31   65    43   30   791     14     7     10
horse             22           3    23   24    61   72     9    763     8     15
ship              78          21    16    8     7   11     4      5   827     23
truck             32         117     7   13     0   17     7     18    37    752
--------------------------------------------------------------------------------
test accuracy: 0.7271139711141587, test loss: 0.7801244765520096
--------------------------------------------------------------------------------
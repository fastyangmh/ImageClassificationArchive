The MNIST_mnasnet_small_checkpoint.ckpt is used the following parameters to train.

python main.py --mode train --data_path data/ --predefined_dataset MNIST --classes None --backbone_model mnasnet_small --in_chans 1 --batch_size 512 --lr 0.019965407596926204 --lr_scheduler StepLR --step_size 9 --train_iter 112

The MNIST_mnasnet_small_checkpoint.ckpt training result is the following.

the train dataset confusion matrix:
           0 - zero  1 - one  2 - two  3 - three  4 - four  5 - five  6 - six  7 - seven  8 - eight  9 - nine
0 - zero       4683        1        6          5         1         2       19          3         14         6
1 - one           1     5307       17          1        19         2        8         34         11         2
2 - two          21        6     4103         50       125       181       85         43         55        55
3 - three        10        8       37       4712         7        52       10         16         66        10
4 - four          3       16      107          4      4299        14       18        117          7        83
5 - five         11        3      148         65        13      3966       66         31         29        38
6 - six          31       12       45         11        20        24     4421         53         28       118
7 - seven         6       25       30         13       142        19       43       4603          3       102
8 - eight        29        9       33         69        18        23       25         14       4409        34
9 - nine         19        0       32         10        46        30       85         70         32      4432
--------------------------------------------------------------------------------
training accuracy: 0.9361078789893617, training loss: 0.1924631104190299
--------------------------------------------------------------------------------

the val dataset confusion matrix:
           0 - zero  1 - one  2 - two  3 - three  4 - four  5 - five  6 - six  7 - seven  8 - eight  9 - nine
0 - zero       1165        0        4          2         0         0        5          0          1         6
1 - one           0     1314        3          0         9         2        3          7          2         0
2 - two           6        1     1062         11        32        53       30         15         11        13
3 - three         5        2        5       1142         1        19        2          5         19         3
4 - four          0        2       31          2      1086         6        6         22          2        17
5 - five          4        3       39         15        11       934       22         12          2         9
6 - six           9        2        7          1         7         4     1075         13          4        33
7 - seven         2        5        9          2        39         5       13       1175          1        28
8 - eight         8        2        8         13         3         9       12          5       1115        13
9 - nine          6        2       10          4        10        12       27         12         10      1100
--------------------------------------------------------------------------------
validation accuracy: 0.9307221919298172, validation loss: 0.21207198314368725
--------------------------------------------------------------------------------

the test dataset confusion matrix:
           0 - zero  1 - one  2 - two  3 - three  4 - four  5 - five  6 - six  7 - seven  8 - eight  9 - nine
0 - zero        975        0        1          0         0         0        2          0          0         2
1 - one           0     1126        1          0         3         0        1          4          0         0
2 - two           2        1      965          1        22        17       13          7          3         1
3 - three         0        1        0        997         0         4        1          3          4         0
4 - four          0        1       25          1       943         0        1          9          0         2
5 - five          1        0       22          3         1       855        6          3          1         0
6 - six           4        1       12          0         1         3      922          8          1         6
7 - seven         0        4        4          2        17         1        3        996          0         1
8 - eight         0        0        3          1         1         0        1          1        963         4
9 - nine          0        0        5          1         6         4        6          3          5       979
--------------------------------------------------------------------------------
test accuracy: 0.9718922346830368, test loss: 0.09054205305874348
--------------------------------------------------------------------------------
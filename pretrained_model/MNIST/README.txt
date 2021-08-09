The MNIST_mnasnet_small_checkpoint.ckpt is used the following parameters to train.

python main.py --mode train --data_path data/ --predefined_dataset MNIST --classes None --in_chans 1 --batch_size 128 --backbone_model mnasnet_small --num_workers 4 --train_iter 138 --step_size 5 --lr 0.0003105158274479257 --alpha 0.20573437708578535 --lr_scheduler CosineAnnealingLR --loss_function BCELoss

The MNIST_mnasnet_small_checkpoint.ckpt training result is the following.

the train dataset confusion matrix:
           0 - zero  1 - one  2 - two  3 - three  4 - four  5 - five  6 - six  7 - seven  8 - eight  9 - nine
0 - zero       4621        1        4          9         1         8       35          7         22        14
1 - one           1     5289       24         15        22         2        4         37          6         1
2 - two          31       12     3907         68       184       252       84         73        108        61
3 - three        20       11       56       4624         7        79       22         28        102        20
4 - four          3       29      153          7      4193        23       20        110         10        95
5 - five         22       24      171        101        27      3763       89         50         30        60
6 - six          39       34       39         21        39        36     4279         85         31       113
7 - seven         9       38       58         17       216        11       43       4448          7       126
8 - eight        34       25       37         61        31        45       47         13       4334        52
9 - nine         47       27       47         24        75        62      187         84         42      4185
--------------------------------------------------------------------------------
training accuracy: 0.9092291666666666, training loss: 0.1693627180258433
--------------------------------------------------------------------------------

the val dataset confusion matrix:
           0 - zero  1 - one  2 - two  3 - three  4 - four  5 - five  6 - six  7 - seven  8 - eight  9 - nine
0 - zero       1175        0        1          2         1         5        5          0          2        10
1 - one           0     1313        7          1         5         1        2         10          0         2
2 - two          11        4      959         19        44        69       25         18         20         9
3 - three         4        5       15       1074         2        21        6          8         22         5
4 - four          2        8       31          1      1076         9        5         30          2        35
5 - five          4        7       31         32         8       949       19         14          3        17
6 - six           8        9       11          4        11        15     1086         18          8        32
7 - seven         3       15       16          8        55         3       16       1147          0        29
8 - eight         8        5        7         13        11        14       14          1       1090         9
9 - nine         23        7        6         12        16        22       39         20          8      1016
--------------------------------------------------------------------------------
validation accuracy: 0.9069703012070758, validation loss: 0.16969262190321657
--------------------------------------------------------------------------------

the test dataset confusion matrix:
           0 - zero  1 - one  2 - two  3 - three  4 - four  5 - five  6 - six  7 - seven  8 - eight  9 - nine
0 - zero        971        0        1          0         0         1        3          0          0         4
1 - one           0     1124        1          1         5         1        0          3          0         0
2 - two           2        0      915          0        33        45       21          9          4         3
3 - three         0        0        4        994         0         5        0          4          3         0
4 - four          0        1       24          1       924         0        1         22          0         9
5 - five          3        0       27          7         4       834        8          6          0         3
6 - six           2        2        2          0         3         0      916         23          1         9
7 - seven         0        5        7          4        30         2        2        970          1         7
8 - eight         0        0        3          4         3         1        1          1        954         7
9 - nine          1        1        9          7        11        13       11          8          5       943
--------------------------------------------------------------------------------
test accuracy: 0.9550039556962026, test loss: 0.15762682259082794
--------------------------------------------------------------------------------
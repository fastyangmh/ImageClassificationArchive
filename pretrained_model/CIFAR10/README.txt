The CIFAR10_mnasnet_small_checkpoint.ckpt is used the following parameters to train.

python main.py --mode train --data_path data/ --predefined_dataset CIFAR10 --classes None --backbone_model mnasnet_small --batch_size 512 --lr 0.03188473363097435 --lr_scheduler CosineAnnealingLR --step_size 2 --train_iter 190

The CIFAR10_mnasnet_small_checkpoint.ckpt training result is the following.

the train dataset confusion matrix:
            airplane  automobile  bird   cat  deer   dog  frog  horse  ship  truck
airplane        2447         137   267   122    35    79    16     82   550    239
automobile        37        3021    39    60     9    71    59     35   175    553
bird             340          53  2065   324   428   239   217    118   112     54
cat               70          68   246  1635   276  1107   251    105   127     79
deer              83           6   494   225  2384   137   301    248    97     31
dog               38          46   241   720   238  2340    84    139   104     34
frog              16          30   225   465   299    79  2784     23    32     50
horse             68          75   182   241   479   387    43   2353    72    111
ship             212         120    75   111    32    86    25     18  3155    195
truck             63         681    32   132    24    85    77     75   222   2629
--------------------------------------------------------------------------------
test accuracy: 0.6210690268987342, test loss: 1.0582304566721372
--------------------------------------------------------------------------------

test the val dataset
the val dataset confusion matrix:
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
            airplane  automobile  bird  cat  deer  dog  frog  horse  ship  truck
airplane         645          21    73   27    13   21     5     21   133     67
automobile        12         689    10   15     1   25    10      8    29    142
bird              87          22   535   92   118   65    65     27    29     10
cat               16          19    85  445    57  273    57     33    28     23
deer              23           1   109   47   600   33    79     62    24     16
dog                9          12    76  187    63  573    24     34    30      8
frog               3           8    51  126    79   24   677      5    17      7
horse             14          14    54   76   114  104    11    553    20     29
ship              47          26    22   24    12   21     7      5   762     45
truck             17         158     9   31     7   29    18     15    52    644
--------------------------------------------------------------------------------
test accuracy: 0.6117359846830368, test loss: 1.09590602517128
--------------------------------------------------------------------------------

test the test dataset
the test dataset confusion matrix:
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
            airplane  automobile  bird  cat  deer  dog  frog  horse  ship  truck
airplane         568          52    66   28    10   31     1     26   154     64
automobile         8         738     5    8     2   35     5      7    37    155
bird              88          21   521   86   100   81    52     18    26      7
cat               16          15    69  400    59  312    59     23    30     17
deer              19           3   141   48   604   45    71     39    26      4
dog                8           9    59  129    44  684    18     21    21      7
frog               3           5    59  127    53   31   702      9     5      6
horse             15           4    51   48   120  148    10    539    21     44
ship              37          30    23   31     7   40     4      5   776     47
truck             11         169     6   33     5   26    10     18    51    671
--------------------------------------------------------------------------------
test accuracy: 0.6193761497735977, test loss: 1.0684717178344727
--------------------------------------------------------------------------------
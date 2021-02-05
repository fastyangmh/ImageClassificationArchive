# import
import argparse
from os import listdir
from os.path import join
import torch

# class


class ProjectPrameters():
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # base
        self.parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict', 'tune', 'evaluate'],
                                 help='if the mode equals train, will train the model. if the mode equals predict, will use the pre-trained model to predict. if the mode equals tune, will hyperparameter tuning the model. if the mode equals evaluate, will evaluate the model by the k-fold validation.')
        self.parser.add_argument(
            '--dataPath', type=str, default='data/dogs_vs_cats/', help='the data path.')
        self.parser.add_argument('--predefinedTask', type=str, default=None, choices=[
                                 'mnist', 'cifar10'], help='the predefined task that provided the mnist and cifar10 tasks.')
        self.parser.add_argument('--dataType', type=self.str_to_str_list, default='none',
                                 help='the categories of data. if "none" then will automatically get the dataType from the train directory.')
        self.parser.add_argument(
            '--randomSeed', type=self.str_to_int, default=0, help='the random seed.')
        self.parser.add_argument(
            '--savePath', type=str, default='save/', help='the path which store the models, optimizers, epoch, loss.')
        self.parser.add_argument('--valSize', type=float, default=0.1,
                                 help='the validation data size used for the predefined task.')
        self.parser.add_argument('--noCuda', action='store_true', default=False,
                                 help='whether to use Cuda to train the model. if True which will train the model on CPU. if False which will train on GPU.')

        # feature
        self.parser.add_argument(
            '--maxImageSize', type=int, default=200, help='the maximum image size.')

        # train
        self.parser.add_argument(
            '--batchSize', type=int, default=32, help='how many samples per batch to load.')

        # model
        self.parser.add_argument('--modelType', type=str, default='single', choices=[
                                 'single', 'multiple'], help='the model type. the single type is a classifier with multiclass. the multiple type is a binary classifier.')

        # debug
        self.parser.add_argument(
            '--maxFiles', type=self.str_to_int, default=None, help='the maximum number of files.')

    def str_to_int(self, s):
        if s == 'None' or s == 'none':
            return None
        else:
            return int(s)

    def str_to_str_list(self, s):
        return [str(v) for v in s.split(',') if len(v) > 0]

    def parse(self):
        projectParams = self.parser.parse_args()

        # base
        if projectParams.predefinedTask is not None:
            projectParams.dataPath = join(
                './data/', projectParams.predefinedTask)

        if projectParams.dataType == ['none'] and projectParams.predefinedTask is None:
            try:
                dirs = listdir(join(projectParams.dataPath, 'train'))
                projectParams.dataType = {dirname: idx for idx,
                                          dirname in enumerate(dirs)}
            except:
                assert False, 'the dataPath does not exist the data.'
        projectParams.useCuda = torch.cuda.is_available() and not projectParams.noCuda

        return projectParams


if __name__ == "__main__":
    # project parameters
    projectParams = ProjectPrameters().parse()

    # display each parameter
    for k, v in vars(projectParams).items():
        print('{:<12}= {}'.format(k, v))

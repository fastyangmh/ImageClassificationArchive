# import
import argparse
from os.path import join, basename, abspath
import torch
from datetime import datetime
from glob import glob

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
        self.parser.add_argument('--dataType', type=self.str_to_str_list, default=None,
                                 help='the categories of data. if the value equals None then will automatically get the dataType from the train directory.')
        self.parser.add_argument(
            '--randomSeed', type=self.str_to_int, default=0, help='the random seed.')
        self.parser.add_argument(
            '--savePath', type=str, default='save/', help='the path which store the models, optimizers, epoch, loss.')
        self.parser.add_argument('--valSize', type=float, default=0.1,
                                 help='the validation data size used for the predefined task.')
        self.parser.add_argument('--noCuda', action='store_true', default=False,
                                 help='whether to use Cuda to train the model. if True which will train the model on CPU. if False which will train on GPU.')
        self.parser.add_argument('--valIter', type=self.str_to_int,
                                 default=None, help='the number of validation iteration.')
        self.parser.add_argument(
            '--checkpointPath', type=str, default=None, help='the path which store the checkpoint.')
        self.parser.add_argument('--numWorkers', type=int, default=torch.get_num_threads(
        ), help='how many subprocesses to use for data loading.')
        self.parser.add_argument('--noBalance', action='store_true',
                                 default=False, help='whether to balance the data.')

        # feature
        self.parser.add_argument(
            '--maxImageSize', type=self.str_to_int_list, default=[224], help='the maximum image size(height, width).')

        # train
        self.parser.add_argument(
            '--batchSize', type=int, default=32, help='how many samples per batch to load.')
        self.parser.add_argument('--optimizer', type=str, default='adam', choices=[
                                 'adam', 'sgd'], help='the optimizer while training model.')
        self.parser.add_argument(
            '--lr', type=float, default=1e-3, help='the learning rate.')
        self.parser.add_argument(
            '--weightDecay', type=float, default=0., help='the weight decay of optimizer (L2 penalty).')
        self.parser.add_argument(
            '--momentum', type=float, default=0.1, help='the momentum factor of the SGD optimizer.')
        self.parser.add_argument(
            '--trainIter', type=int, default=100, help='the number of training iteration.')
        self.parser.add_argument('--lrScheduler', type=str, default='cosine', choices=[
                                 'cosine', 'step'], help='the lr scheduler while training model.')
        self.parser.add_argument(
            '--lrSchedulerStepSize', type=int, default=10, help='period of learning rate decay.')
        self.parser.add_argument('--lrSchedulerGamma', type=int, default=0.1,
                                 help='multiplicative factor of learning rate decay.')
        self.parser.add_argument('--noEarlyStopping', action='store_true',
                                 default=False, help='whether to use early stopping while training.')
        self.parser.add_argument('--earlyStoppingPatience', type=int, default=3,
                                 help='number of validation checks with no improvement after which training will be stopped.')

        # model
        self.parser.add_argument('--backboneModel', type=str, default='mobilenetv2', choices=[
                                 'resnet18', 'wideresnet50', 'resnext50', 'vgg11bn', 'mobilenetv2', 'ghostnet'], help='the backbone model used for classification.')

        # evaluate
        self.parser.add_argument(
            '--kFoldValue', type=int, default=5, help='the value of k-fold validation.')

        # tune
        self.parser.add_argument(
            '--tuneIter', type=int, default=100, help='the number of tuning iteration.')
        self.parser.add_argument(
            '--tuneCPU', type=int, default=2, help='CPU resources to allocate per trial.')
        self.parser.add_argument(
            '--tuneGPU', type=float, default=None, help='GPU resources to allocate per trial.')

        # debug
        self.parser.add_argument(
            '--maxFiles', type=self.str_to_int, default=None, help='the maximum number of files.')
        self.parser.add_argument('--report', type=str, default=None, choices=[
                                 'simple', 'advanced'], help='whether to report the bottleneck.')
        self.parser.add_argument('--weightsSummary', type=str, default=None, choices=[
                                 'top', 'full'], help='whether to report the weight of the model.')
        self.parser.add_argument('--tuneDebug', action='store_true',
                                 default=False, help='whether to use debug mode while tuning.')

    def str_to_str_list(self, s):
        return [str(v) for v in s.split(',') if len(v) > 0]

    def str_to_int(self, s):
        if s == 'None' or s == 'none':
            return None
        else:
            return int(s)

    def str_to_int_list(self, s):
        return [int(v) for v in s.split(',') if len(v) > 0]

    def parse(self):
        projectParams = self.parser.parse_args()

        # base
        projectParams.dataPath = abspath(projectParams.dataPath)
        if projectParams.predefinedTask is not None:
            # the dataType of predefinedTask will automatically get from data_preparation
            projectParams.dataPath = join(
                './data/', projectParams.predefinedTask)
            projectParams.numClasses = 10
        elif projectParams.dataType is None:
            projectParams.dataType = {dType: idx for idx, dType in enumerate(sorted(
                [basename(dType[:-1]) for dType in glob(join(projectParams.dataPath, 'train/*/'))]))}
            projectParams.numClasses = len(projectParams.dataType)
            assert projectParams.numClasses, 'there does not get any dataType.'
        else:
            projectParams.dataType = {
                dType: idx for idx, dType in enumerate(sorted(projectParams.dataType))}
            projectParams.numClasses = len(projectParams.dataType)
        projectParams.useCuda = torch.cuda.is_available() and not projectParams.noCuda
        projectParams.gpus = -1 if projectParams.useCuda else 0
        if projectParams.valIter is None:
            projectParams.valIter = projectParams.trainIter
        projectParams.useBalance = not projectParams.noBalance
        if projectParams.useBalance and projectParams.mode == 'train' and projectParams.predefinedTask is None:
            dataWeight = {}
            for dType in projectParams.dataType.keys():
                dataWeight[dType] = len(
                    glob(join(projectParams.dataPath, 'train/{}/*.png'.format(dType))))
            projectParams.dataWeight = {
                dType: 1-(dataWeight[dType]/sum(dataWeight.values())) for dType in dataWeight.keys()}
        if projectParams.mode == 'tune':
            projectParams.numWorkers = projectParams.tuneCPU

        # feature
        if len(projectParams.maxImageSize) == 1:
            projectParams.maxImageSize = projectParams.maxImageSize*2
        else:
            # the PIL image size is (width, height), then the maxImageSize needs to swap.
            projectParams.maxImageSize.reverse()

        # train
        projectParams.useEarlyStopping = not projectParams.noEarlyStopping
        if projectParams.useEarlyStopping:
            # because the PyTorch lightning needs to get validation loss in every training epoch.
            projectParams.valIter = 1

        # evaluate
        if projectParams.mode == 'evaluate':
            projectParams.kFoldDataPath = './kFoldDataset{}'.format(
                datetime.now().strftime('%Y%m%d%H%M%S'))

        # tune
        if projectParams.tuneGPU is None:
            projectParams.tuneGPU = torch.cuda.device_count()/projectParams.tuneCPU

        return projectParams


if __name__ == "__main__":
    # project parameters
    projectParams = ProjectPrameters().parse()

    # display each parameter
    for k, v in vars(projectParams).items():
        print('{:<12}= {}'.format(k, v))

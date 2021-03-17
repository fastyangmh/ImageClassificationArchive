# import
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import torch
from src.project_parameters import ProjectPrameters
import pytorch_lightning as pl
from torchvision.models import resnet18, wide_resnet50_2, resnext50_32x4d, vgg11_bn, mobilenet_v2
import torch.nn as nn
import numpy as np
import torch.optim as optim
import pandas as pd
from src.utils import load_checkpoint

# def


def get_classifier(projectParams):
    officialModelDict = {'resnet18': 'resnet18', 'wideresnet50': 'wide_resnet50_2',
                         'resnext50': 'resnext50_32x4d', 'vgg11bn': 'vgg11_bn', 'mobilenetv2': 'mobilenet_v2'}
    pytorchHub = {'ghostnet': ['huawei-noah/ghostnet', 'ghostnet_1x']}
    if projectParams.backboneModel in officialModelDict:
        classifier = eval('{}(pretrained=True, progress=False)'.format(
            officialModelDict[projectParams.backboneModel]))
    else:
        repo, model = pytorchHub[projectParams.backboneModel]
        classifier = torch.hub.load(repo, model, pretrained=True)

    # change the number of output feature
    if projectParams.backboneModel in ['vgg11bn', 'mobilenetv2']:
        classifier.classifier[-1] = nn.Linear(
            in_features=classifier.classifier[-1].in_features, out_features=projectParams.numClasses)
    elif projectParams.backboneModel in ['ghostnet']:
        classifier.classifier = nn.Linear(
            in_features=classifier.classifier.in_features, out_features=projectParams.numClasses)
    else:
        classifier.fc = nn.Linear(
            in_features=classifier.fc.in_features, out_features=projectParams.numClasses)

    # change the number of input channels
    if projectParams.predefinedTask == 'mnist':
        if projectParams.backboneModel == 'vgg11bn':
            classifier.features[0] = nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        elif projectParams.backboneModel == 'mobilenetv2':
            classifier.features[0][0] = nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)
        elif projectParams.backboneModel == 'ghostnet':
            classifier.conv_stem = nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False)
        else:
            classifier.conv1 = nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
    return classifier


def get_optimizer(model_parameters, projectParams):
    if projectParams.optimizer == 'adam':
        optimizer = optim.Adam(
            params=model_parameters, lr=projectParams.lr, weight_decay=projectParams.weightDecay)
    elif projectParams.optimizer == 'sgd':
        optimizer = optim.SGD(
            params=model_parameters, lr=projectParams.lr, momentum=projectParams.momentum)
    return optimizer


def get_lr_scheduler(projectParams, optimizer):
    if projectParams.lrScheduler == 'cosine':
        lrScheduler = CosineAnnealingLR(
            optimizer=optimizer, T_max=projectParams.lrSchedulerStepSize)
    elif projectParams.lrScheduler == 'step':
        lrScheduler = StepLR(
            optimizer=optimizer, step_size=projectParams.lrSchedulerStepSize, gamma=projectParams.lrSchedulerGamma)
    return lrScheduler


def get_criterion(projectParams):
    if 'dataWeight' in projectParams:
        # in order to prevent error while predict stage or predefinedTask, the condition use it
        weight = torch.Tensor(list(projectParams.dataWeight.values()))
    else:
        weight = None
    criterion = nn.CrossEntropyLoss(weight=weight)
    return criterion


def create_model(projectParams):
    model = Net(projectParams=projectParams)
    if projectParams.checkpointPath is not None:
        model = load_checkpoint(model=model, projectParams=projectParams)
    return model

# class


class Net(pl.LightningModule):
    def __init__(self, projectParams):
        super().__init__()
        self.classifier = get_classifier(projectParams=projectParams)
        self.activation = nn.Softmax(dim=-1)
        self.criterion = get_criterion(projectParams=projectParams)
        self.accuracy = pl.metrics.Accuracy()
        self.confMat = pl.metrics.ConfusionMatrix(
            num_classes=projectParams.numClasses)
        self.projectParams = projectParams

    def forward(self, x):
        return self.activation(self.classifier(x))

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop('loss', None)
        return items

    def training_step(self, batch, batchIndex):
        x, y = batch
        yhat = self.classifier(x)
        loss = self.criterion(yhat, y)
        trainACC = self.accuracy(yhat, y)
        return {'loss': loss, 'accuracy': trainACC}

    def training_epoch_end(self, outputs) -> None:
        epochLoss = []
        epochAcc = []
        for stepDict in outputs:
            epochLoss.append(stepDict['loss'].item())
            epochAcc.append(stepDict['accuracy'].item())
        self.log('train epoch loss', np.mean(
            epochLoss), on_epoch=True, prog_bar=True)
        self.log('train epoch accuracy', np.mean(epochAcc))

    def validation_step(self, batch, batchIndex):
        x, y = batch
        yhat = self.forward(x)
        loss = self.criterion(yhat, y)
        valAcc = self.accuracy(yhat, y)
        return {'loss': loss, 'accuracy': valAcc}

    def validation_epoch_end(self, outputs) -> None:
        epochLoss = []
        epochAcc = []
        for stepDict in outputs:
            epochLoss.append(stepDict['loss'].item())
            epochAcc.append(stepDict['accuracy'].item())
        self.log('validation epoch loss', np.mean(
            epochLoss), on_epoch=True, prog_bar=True)
        self.log('validation epoch accuracy', np.mean(epochAcc))

    def test_step(self, batch, batchIndex):
        x, y = batch
        yhat = self.forward(x)
        loss = self.criterion(yhat, y)
        testAcc = self.accuracy(yhat, y)
        return {'loss': loss, 'accuracy': testAcc, 'yPred': yhat, 'yTrue': y}

    def test_epoch_end(self, outputs) -> None:
        epochLoss = []
        epochAcc = []
        yPred = []
        yTrue = []
        for stepDict in outputs:
            epochLoss.append(stepDict['loss'].item())
            epochAcc.append(stepDict['accuracy'].item())
            yPred.append(stepDict['yPred'])
            yTrue.append(stepDict['yTrue'])
        self.log('test epoch loss', np.mean(epochLoss))
        self.log('test epoch accuracy', np.mean(epochAcc))
        yPred = torch.cat(yPred, 0)
        yTrue = torch.cat(yTrue, 0)
        confMat = pd.DataFrame(self.confMat(yPred, yTrue).tolist(), columns=self.projectParams.dataType.keys(
        ), index=self.projectParams.dataType.keys()).astype(int)
        print(confMat)

    def configure_optimizers(self):
        optimizer = get_optimizer(
            model_parameters=self.parameters(), projectParams=self.projectParams)
        if self.projectParams.lrSchedulerStepSize > 0:
            lrScheduler = get_lr_scheduler(
                projectParams=self.projectParams, optimizer=optimizer)
            return [optimizer], [lrScheduler]
        else:
            return optimizer


if __name__ == '__main__':
    # project parameters
    projectParams = ProjectPrameters().parse()
    projectParams.numClasses = 2
    channel = 1 if projectParams.predefinedTask == 'mnist' else 3

    # create model
    model = create_model(projectParams=projectParams)

    # create input data
    x = torch.ones(projectParams.batchSize, channel,
                   projectParams.maxImageSize[0], projectParams.maxImageSize[1])

    # get model output
    y = model(x)

    # display the dimension of input and output
    print(x.shape)
    print(y.shape)

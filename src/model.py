# import
import torch
from torch.optim import optimizer
from src.project_parameters import ProjectPrameters
import pytorch_lightning as pl
from torchvision.models import resnet18, wide_resnet50_2, resnext50_32x4d, vgg11_bn, mobilenet_v2
import torch.nn as nn
import numpy as np
import torch.optim as optim

# def


def get_classifier(projectParams):
    modelDict = {'resnet18': 'resnet18', 'wideresnet50': 'wide_resnet50_2',
                 'resnext50': 'resnext50_32x4d', 'vgg11bn': 'vgg11_bn', 'mobilenetv2': 'mobilenet_v2'}
    classifier = eval('{}(pretrained=True, progress=False)'.format(
        modelDict[projectParams.backboneModel]))
    # change the number of output feature
    if projectParams.backboneModel in ['vgg11bn', 'mobilenetv2']:
        classifier.classifier[-1] = nn.Linear(
            in_features=classifier.classifier[-1].in_features, out_features=projectParams.numClasses)
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


# class


class Net(pl.LightningModule):
    def __init__(self, projectParams):
        super().__init__()
        self.classifier = get_classifier(projectParams=projectParams)
        self.activation = nn.Softmax(dim=-1)
        self.criterion = nn.CrossEntropyLoss()
        self.projectParams = projectParams
        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        return self.activation(self.classifier(x))

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
        self.log('train epoch accuracy', np.mean(
            epochAcc), on_epoch=True, prog_bar=True)

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
        self.log('validation epoch accuracy', np.mean(
            epochAcc), on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = get_optimizer(
            model_parameters=self.parameters(), projectParams=self.projectParams)
        return optimizer


if __name__ == '__main__':
    # project parameters
    projectParams = ProjectPrameters().parse()
    projectParams.numClasses = 2
    channel = 1 if projectParams.predefinedTask == 'mnist' else 3

    # create model
    model = Net(projectParams=projectParams)

    # create input data
    x = torch.ones(projectParams.batchSize, channel,
                   projectParams.maxImageSize, projectParams.maxImageSize)

    # get model output
    y = model(x)

    # display the dimension of input and output
    print(x.shape)
    print(y.shape)

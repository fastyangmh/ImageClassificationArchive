# import
import timm
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from src.project_parameters import ProjectParameters
from pytorch_lightning import LightningModule
import torch.nn as nn
from torchmetrics import Accuracy, ConfusionMatrix
import pandas as pd
import numpy as np
from src.utils import load_checkpoint, load_yaml
import torch.optim as optim
from os.path import dirname, basename
import torch.nn.functional as F
import sys

# def


def _get_backbone_model_from_file(filepath, in_chans, num_classes):
    """Load a BackboneModel from a file .

    Args:
        filepath (str): the file path of the backbone model.
        in_chans (int): number of input channels / colors.
        num_classes (int): the number of classes.

    Returns:
        nn.Module: the self-defined backbone model.
    """
    sys.path.append('{}'.format(dirname(filepath)))
    class_name = basename(filepath).split('.')[0]
    exec('from {} import {}'.format(*[class_name]*2))
    return eval('{}(in_chans={}, num_classes={})'.format(class_name, in_chans, num_classes))


def _get_backbone_model(project_parameters):
    """Get the backbone model .

    Args:
        project_parameters (argparse.Namespace): the parameters for the project.

    Returns:
        timm.models or nn.Module: the timm model or the self-defined backbone model.
    """
    if project_parameters.backbone_model in timm.list_models():
        backbone_model = timm.create_model(model_name=project_parameters.backbone_model, pretrained=True,
                                           num_classes=project_parameters.num_classes, in_chans=project_parameters.in_chans)
    elif '.py' in project_parameters.backbone_model:
        backbone_model = _get_backbone_model_from_file(
            filepath=project_parameters.backbone_model, in_chans=project_parameters.in_chans, num_classes=project_parameters.num_classes)
    else:
        assert False, 'please check the backbone model. the backbone model: {}'.format(
            project_parameters.backbone_model)
    return backbone_model


def _get_loss_function(project_parameters):
    """Get loss function .

    Args:
        project_parameters (argparse.Namespace): the parameters for the project.

    Returns:
        torch.nn.modules.loss.CrossEntropyLoss: It is useful when training a classification problem with `C` classes.
    """
    if 'data_weight' in project_parameters:
        weight = torch.Tensor(list(project_parameters.data_weight.values()))
    else:
        weight = None
    return nn.CrossEntropyLoss(weight=weight)


def _get_optimizer(model_parameters, project_parameters):
    """Get optimizer .

    Args:
        model_parameters (iterable): iterable of parameters to optimize or dicts defining parameter groups
        project_parameters (argparse.Namespace): the parameters for the project.

    Returns:
        torch.optim: the optimization algorithm.
    """
    optimizer_config = load_yaml(
        filepath=project_parameters.optimizer_config_path)
    optimizer_name = list(optimizer_config.keys())[0]
    if optimizer_name in dir(optim):
        for name, value in optimizer_config.items():
            if value is None:
                optimizer = eval('optim.{}(params=model_parameters, lr={})'.format(
                    optimizer_name, project_parameters.lr))
            elif type(value) is dict:
                value = ('{},'*len(value)).format(*['{}={}'.format(a, b)
                                                    for a, b in value.items()])
                optimizer = eval('optim.{}(params=model_parameters, lr={}, {})'.format(
                    optimizer_name, project_parameters.lr, value))
            else:
                assert False, '{}: {}'.format(name, value)
        return optimizer
    else:
        assert False, 'please check the optimizer. the optimizer config: {}'.format(
            optimizer_config)


def _get_lr_scheduler(project_parameters, optimizer):
    """Returns the LR scheduler .

    Args:
        project_parameters (argparse.Namespace): the parameters for the project.
        optimizer (Optimizer): Wrapped optimizer.

    Returns:
        torch.optim.lr_scheduler: the LR scheduler. 
    """
    if project_parameters.lr_scheduler == 'StepLR':
        lr_scheduler = StepLR(optimizer=optimizer,
                              step_size=project_parameters.step_size, gamma=project_parameters.gamma)
    elif project_parameters.lr_scheduler == 'CosineAnnealingLR':
        lr_scheduler = CosineAnnealingLR(
            optimizer=optimizer, T_max=project_parameters.step_size)
    return lr_scheduler


def create_model(project_parameters):
    """Create a neural network model .

    Args:
        project_parameters (argparse.Namespace): the parameters for the project.

    Returns:
        LightningModule: a neural network model.
    """
    model = Net(project_parameters=project_parameters)
    if project_parameters.checkpoint_path is not None:
        model = load_checkpoint(model=model, num_classes=project_parameters.num_classes,
                                use_cuda=project_parameters.use_cuda, checkpoint_path=project_parameters.checkpoint_path)
    return model

# class


class Net(LightningModule):
    """Constructs a LightningModule class .
    """

    def __init__(self, project_parameters):
        """Initialize the class.

        Args:
            project_parameters (argparse.Namespace): the parameters for the project.
        """
        super().__init__()
        self.project_parameters = project_parameters
        self.backbone_model = _get_backbone_model(
            project_parameters=project_parameters)
        self.activation_function = nn.Softmax(dim=-1)
        self.loss_function = _get_loss_function(
            project_parameters=project_parameters)
        self.accuracy = Accuracy()
        self.confusion_matrix = ConfusionMatrix(
            num_classes=project_parameters.num_classes)

    def training_forward(self, x):
        """Defines the computation performed at every call in training.

        Args:
            x (torch.Tensor): the input data.

        Returns:
            torch.Tensor: the predict of neural network model.
        """
        return self.backbone_model(x)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): the input data.

        Returns:
            torch.Tensor: the predict of neural network model.
        """
        return self.activation_function(self.backbone_model(x))

    def get_progress_bar_dict(self):
        """Remove the step loss information from the progress bar .

        Returns:
            dict: Dictionary with the items to be displayed in the progress bar.
        """
        # don't show the loss value
        items = super().get_progress_bar_dict()
        items.pop('loss', None)
        return items

    def _parse_outputs(self, outputs, calculate_confusion_matrix):
        """Parse the outputs to get the epoch of loss and accuracy .

        Args:
            outputs (dict): the output contains loss and accuracy.
            calculate_confusion_matrix (bool): whether to calculate the confusion matrix.

        Returns:
            tuple: the tuple contains the epoch of loss and accuracy. And if calculate_confusion_matrix is True, the tuple will contain a confusion matrix.
        """
        epoch_loss = []
        epoch_accuracy = []
        if calculate_confusion_matrix:
            y_true = []
            y_pred = []
        for step in outputs:
            epoch_loss.append(step['loss'].item())
            epoch_accuracy.append(step['accuracy'].item())
            if calculate_confusion_matrix:
                y_pred.append(step['y_hat'])
                y_true.append(step['y'])
        if calculate_confusion_matrix:
            y_pred = torch.cat(y_pred, 0)
            y_true = torch.cat(y_true, 0)
            confmat = pd.DataFrame(self.confusion_matrix(y_pred, y_true).tolist(
            ), columns=self.project_parameters.classes, index=self.project_parameters.classes).astype(int)
            return epoch_loss, epoch_accuracy, confmat
        else:
            return epoch_loss, epoch_accuracy

    def training_step(self, batch, batch_idx):
        """Compute and return the training loss and accuracy.

        Args:
            batch (:class:`~torch.Tensor` | (:class:`~torch.Tensor`, ...) | [:class:`~torch.Tensor`, ...]): The output of :class:`~torch.utils.data.DataLoader`. A tensor, tuple or list.
            batch_idx (int): Integer displaying index of this batch.

        Returns:
            dict: the dictionary contains loss and accuracy.
        """
        x, y = batch
        y_hat = self.training_forward(x)
        loss = self.loss_function(y_hat, y)
        train_step_accuracy = self.accuracy(F.softmax(y_hat, dim=-1), y)
        return {'loss': loss, 'accuracy': train_step_accuracy}

    def training_epoch_end(self, outputs):
        """Called at the end of the training epoch with the outputs of all training steps.

        Args:
            outputs (list): List of outputs defined in :meth:`training_step`.
        """
        epoch_loss, epoch_accuracy = self._parse_outputs(
            outputs=outputs, calculate_confusion_matrix=False)
        self.log('training loss', np.mean(epoch_loss),
                 on_epoch=True, prog_bar=True)
        self.log('training accuracy', np.mean(epoch_accuracy))

    def validation_step(self, batch, batch_idx):
        """Compute and return the validation loss and accuracy.

        Args:
            batch (:class:`~torch.Tensor` | (:class:`~torch.Tensor`, ...) | [:class:`~torch.Tensor`, ...]): The output of :class:`~torch.utils.data.DataLoader`. A tensor, tuple or list.
            batch_idx (int): Integer displaying index of this batch.

        Returns:
            dict: the dictionary contains loss and accuracy.
        """
        x, y = batch
        y_hat = self.training_forward(x)
        loss = self.loss_function(y_hat, y)
        val_step_accuracy = self.accuracy(F.softmax(y_hat, dim=-1), y)
        return {'loss': loss, 'accuracy': val_step_accuracy}

    def validation_epoch_end(self, outputs):
        """Called at the end of the validation epoch with the outputs of all validation steps.

        Args:
            outputs (list): List of outputs defined in :meth:`validation_step`.
        """
        epoch_loss, epoch_accuracy = self._parse_outputs(
            outputs=outputs, calculate_confusion_matrix=False)
        self.log('validation loss', np.mean(epoch_loss),
                 on_epoch=True, prog_bar=True)
        self.log('validation accuracy', np.mean(epoch_accuracy))

    def test_step(self, batch, batch_idx):
        """Operates on a single batch of data from the test set.

        Args:
            batch (:class:`~torch.Tensor` | (:class:`~torch.Tensor`, ...) | [:class:`~torch.Tensor`, ...]): The output of :class:`~torch.utils.data.DataLoader`. A tensor, tuple or list.
            batch_idx (int): Integer displaying index of this batch.

        Returns:
            dict: the dictionary contains loss, accuracy, predicted, and ground truth.
        """
        x, y = batch
        y_hat = self.training_forward(x)
        loss = self.loss_function(y_hat, y)
        test_step_accuracy = self.accuracy(F.softmax(y_hat, dim=-1), y)
        return {'loss': loss, 'accuracy': test_step_accuracy, 'y_hat': F.softmax(y_hat, dim=-1), 'y': y}

    def test_epoch_end(self, outputs):
        """Called at the end of a test epoch with the output of all test steps.

        Args:
            outputs (list): List of outputs defined in :meth:`test_step`.
        """
        epoch_loss, epoch_accuracy, confmat = self._parse_outputs(
            outputs=outputs, calculate_confusion_matrix=True)
        self.log('test loss', np.mean(epoch_loss))
        self.log('test accuracy', np.mean(epoch_accuracy))
        print(confmat)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Returns:
            **Single optimizer** or **List of optimizers and LR schedulers.: the optimizer or the optimizer and LR schedulers.
        """
        optimizer = _get_optimizer(model_parameters=self.parameters(
        ), project_parameters=self.project_parameters)
        if self.project_parameters.step_size > 0:
            lr_scheduler = _get_lr_scheduler(
                project_parameters=self.project_parameters, optimizer=optimizer)
            return [optimizer], [lr_scheduler]
        else:
            return optimizer


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # create model
    model = create_model(project_parameters=project_parameters)

    # display model information
    model.summarize()

    # create input data
    x = torch.ones(project_parameters.batch_size,
                   project_parameters.in_chans, 224, 224)

    # get model output
    y = model.forward(x)

    # display the dimension of input and output
    print(x.shape)
    print(y.shape)

# import
from os.path import join
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data.dataset import random_split
from torchvision.datasets import ImageFolder
from src.project_parameters import ProjectParameters
from pytorch_lightning import LightningDataModule
from src.utils import get_transform_from_file
from typing import Optional, Callable
import numpy as np
from PIL import Image


# class


class ImageFolder(ImageFolder):
    """ImageFolder class for training, evaluation, tuning .
    """

    def __init__(self, root: str, transform: Optional[Callable], in_chans, loss_function, num_classes, alpha):
        super().__init__(root, transform=transform, loader=Image.open)
        self.in_chans = in_chans
        self.loss_function = loss_function
        self.num_classes = num_classes
        self.alpha = alpha

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        data = self.loader(path)
        assert self.in_chans == len(
            data.getbands()), 'please check the channels of image.'
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            label = self.target_transform(label)
        if self.loss_function == 'BCELoss':
            # one-hot encoding
            label = np.eye(self.num_classes)[label]
            # label smoothing
            label = label*(1-self.alpha) + (self.alpha/self.num_classes)
            label = label.astype(np.float32)
        return data, label


class MNIST(MNIST):
    """Special MNIT class .
    """

    def __init__(self, root: str, train: bool, transform: Optional[Callable], download: bool, loss_function, num_classes, alpha):
        super().__init__(root, train=train, transform=transform, download=download)
        self.loss_function = loss_function
        self.num_classes = num_classes
        self.alpha = alpha

    def __getitem__(self, index: int):
        data, label = super().__getitem__(index)
        if self.loss_function == 'BCELoss':
            # one-hot encoding
            label = np.eye(self.num_classes)[label]
            # label smoothing
            label = label*(1-self.alpha) + (self.alpha/self.num_classes)
            label = label.astype(np.float32)
        return data, label


class CIFAR10(CIFAR10):
    """Special CIFAR10 class .
    """

    def __init__(self, root: str, train: bool, transform: Optional[Callable], download: bool, loss_function, num_classes, alpha):
        super().__init__(root, train=train, transform=transform, download=download)
        self.loss_function = loss_function
        self.num_classes = num_classes
        self.alpha = alpha

    def __getitem__(self, index: int):
        data, label = super().__getitem__(index)
        if self.loss_function == 'BCELoss':
            # one-hot encoding
            label = np.eye(self.num_classes)[label]
            # label smoothing
            label = label*(1-self.alpha) + (self.alpha/self.num_classes)
            label = label.astype(np.float32)
        return data, label


class DataModule(LightningDataModule):
    """Constructs a LightningDataModule class .
    """

    def __init__(self, project_parameters):
        """Initialize the class .

        Args:
            project_parameters (argparse.Namespace): the parameters for this project.
        """
        super().__init__()
        self.project_parameters = project_parameters
        self.transform_dict = get_transform_from_file(
            filepath=project_parameters.transform_config_path)

    def prepare_data(self):
        """Prepare data for training, validation, and test .
        """
        if self.project_parameters.predefined_dataset is None:
            self.dataset = {}
            for stage in ['train', 'val', 'test']:
                self.dataset[stage] = ImageFolder(root=join(self.project_parameters.data_path, stage), transform=self.transform_dict[stage], in_chans=self.project_parameters.in_chans,
                                                  loss_function=self.project_parameters.loss_function, num_classes=self.project_parameters.num_classes, alpha=self.project_parameters.alpha)
                # modify the maximum number of files
                if self.project_parameters.max_files is not None:
                    lengths = (self.project_parameters.max_files, len(
                        self.dataset[stage])-self.project_parameters.max_files)
                    self.dataset[stage] = random_split(
                        dataset=self.dataset[stage], lengths=lengths)[0]
            if self.project_parameters.max_files is not None:
                assert self.dataset['train'].dataset.class_to_idx == self.project_parameters.class_to_idx, 'the classes is not the same. please check the classes of data. from ImageFolder: {} from argparse: {}'.format(
                    self.dataset['train'].dataset.class_to_idx, self.project_parameters.class_to_idx)
            else:
                assert self.dataset['train'].class_to_idx == self.project_parameters.class_to_idx, 'the classes is not the same. please check the classes of data. from ImageFolder: {} from argparse: {}'.format(
                    self.dataset['train'].class_to_idx, self.project_parameters.class_to_idx)
        else:
            train_set = eval('{}(root=self.project_parameters.data_path, train=True, download=True, transform=self.transform_dict["train"], loss_function=self.project_parameters.loss_function, num_classes=self.project_parameters.num_classes, alpha=self.project_parameters.alpha)'.format(
                self.project_parameters.predefined_dataset))
            test_set = eval('{}(root=self.project_parameters.data_path, train=False, download=True, transform=self.transform_dict["test"], loss_function=self.project_parameters.loss_function, num_classes=self.project_parameters.num_classes, alpha=self.project_parameters.alpha)'.format(
                self.project_parameters.predefined_dataset))
            # modify the maximum number of files
            if self.project_parameters.max_files is not None:
                for v in [train_set, test_set]:
                    v.data = v.data[:self.project_parameters.max_files]
                    v.targets = v.targets[:self.project_parameters.max_files]
            train_val_lengths = [round((1-self.project_parameters.val_size)*len(train_set)),
                                 round(self.project_parameters.val_size*len(train_set))]
            train_set, val_set = random_split(
                dataset=train_set, lengths=train_val_lengths)
            self.dataset = {'train': train_set,
                            'val': val_set, 'test': test_set}
            assert self.dataset['train'].dataset.class_to_idx == self.project_parameters.class_to_idx, 'the classes is not the same. please check the classes of data. from ImageFolder: {} from argparse: {}'.format(
                self.dataset['train'].dataset.class_to_idx, self.project_parameters.class_to_idx)

    def train_dataloader(self):
        """Returns a DataLoader for the train dataset .

        Returns:
            torch.utils.data.dataloader.DataLoader: Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        """
        return DataLoader(dataset=self.dataset['train'], batch_size=self.project_parameters.batch_size, shuffle=True, pin_memory=self.project_parameters.use_cuda, num_workers=self.project_parameters.num_workers)

    def val_dataloader(self):
        """Return a DataLoader for the validation dataset .

        Returns:
            torch.utils.data.dataloader.DataLoader: Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        """
        return DataLoader(dataset=self.dataset['val'], batch_size=self.project_parameters.batch_size, shuffle=False, pin_memory=self.project_parameters.use_cuda, num_workers=self.project_parameters.num_workers)

    def test_dataloader(self):
        """Returns a DataLoader for the test dataset .

        Returns:
            torch.utils.data.dataloader.DataLoader: Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        """
        return DataLoader(dataset=self.dataset['test'], batch_size=self.project_parameters.batch_size, shuffle=False, pin_memory=self.project_parameters.use_cuda, num_workers=self.project_parameters.num_workers)

    def get_data_loaders(self):
        """Returns a dictionnary containing all data loaders .

        Returns:
            dict: the dictionary contains the data loaders of train, validation, and test.
        """
        return {'train': self.train_dataloader(),
                'val': self.val_dataloader(),
                'test': self.test_dataloader()}


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # get data_module
    data_module = DataModule(project_parameters=project_parameters)
    data_module.prepare_data()

    # display the dataset information
    for stage in ['train', 'val', 'test']:
        print(stage, data_module.dataset[stage])

    # get data loaders
    data_loaders = data_module.get_data_loaders()

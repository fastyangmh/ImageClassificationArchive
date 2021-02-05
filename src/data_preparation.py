# import
from src.project_parameters import ProjectPrameters
from torchvision.datasets import CIFAR10, MNIST
import pytorch_lightning as pl
from torchvision.datasets import ImageFolder
from os.path import join
from torchvision import transforms
from torch.utils.data import random_split, DataLoader

# def


def get_transform_dictionary(projectParams):
    trainTransform = transforms.Compose([transforms.Resize(projectParams.maxImageSize),
                                         transforms.ColorJitter(),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomVerticalFlip(),
                                         transforms.ToTensor()])
    valTransform = transforms.Compose([transforms.Resize(projectParams.maxImageSize),
                                       transforms.ToTensor()])
    testTransform = transforms.Compose([transforms.Resize(projectParams.maxImageSize),
                                        transforms.ToTensor()])
    return {'train': trainTransform, 'val': valTransform, 'test': testTransform}

# class


class MyDataModule(pl.LightningDataModule):
    def __init__(self, projectParams, transformDict):
        super().__init__()
        self.projectParams = projectParams
        self.transformDict = transformDict

    def prepare_data(self):
        if self.projectParams.predefinedTask is None:
            self.dataset = {stage: ImageFolder(root=join(
                self.projectParams.dataPath, stage), transform=self.transformDict[stage]) for stage in ['train', 'val', 'test']}
        else:
            taskDict = {'cifar10': 'CIFAR10', 'mnist': 'MNIST'}
            trainSet = eval('{}(root=self.projectParams.dataPath, train=True, download=True, transform=self.transformDict["train"])'.format(
                taskDict[self.projectParams.predefinedTask]))
            testSet = eval('{}(root=self.projectParams.dataPath, train=False, download=True, transform=self.transformDict["test"])'.format(
                taskDict[self.projectParams.predefinedTask]))
            trainValSize = [int((1-self.projectParams.valSize)*len(trainSet)),
                            int(self.projectParams.valSize*len(trainSet))]
            trainSet, valSet = random_split(
                dataset=trainSet, lengths=trainValSize)
            self.dataset = {'train': trainSet,
                            'val': valSet, 'test': testSet}

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.dataset['train'], batch_size=self.projectParams.batchSize, shuffle=True, pin_memory=self.projectParams.useCuda)

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(dataset=self.dataset['val'], batch_size=self.projectParams.batchSize, shuffle=False, pin_memory=self.projectParams.useCuda)

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(dataset=self.dataset['test'], batch_size=self.projectParams.batchSize, shuffle=False, pin_memory=self.projectParams.useCuda)


if __name__ == "__main__":
    # parameters
    projectParams = ProjectPrameters().parse()

    # get dataset
    dataset = MyDataModule(
        projectParams=projectParams, transformDict=get_transform_dictionary(projectParams=projectParams))

    # get data loader
    dataset.prepare_data()
    trainLoader = dataset.train_dataloader()
    valLoader = dataset.val_dataloader()
    testLoader = dataset.test_dataloader()

    # display the dataset information
    for stage in ['train', 'val', 'test']:
        print(stage, dataset.dataset[stage])

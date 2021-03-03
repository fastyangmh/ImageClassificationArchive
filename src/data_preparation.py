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
    trainTransform = transforms.Compose([transforms.Resize(size=(projectParams.maxImageSize[0], projectParams.maxImageSize[1])),
                                         transforms.ColorJitter(),
                                         transforms.RandomRotation(degrees=90),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomVerticalFlip(),
                                         transforms.ToTensor(),
                                         transforms.RandomErasing()])
    valTransform = transforms.Compose([transforms.Resize((projectParams.maxImageSize[0], projectParams.maxImageSize[1])),
                                       transforms.ToTensor()])
    testTransform = transforms.Compose([transforms.Resize((projectParams.maxImageSize[0], projectParams.maxImageSize[1])),
                                        transforms.ToTensor()])
    return {'train': trainTransform, 'val': valTransform, 'test': testTransform}

# class


class MyDataModule(pl.LightningDataModule):
    def __init__(self, projectParams):
        super().__init__()
        self.projectParams = projectParams
        self.transformDict = get_transform_dictionary(
            projectParams=projectParams)

    def prepare_data(self):
        if self.projectParams.predefinedTask is None:
            self.dataset = {}
            # modify the maximum number of files
            for stage in ['train', 'val', 'test']:
                self.dataset[stage] = ImageFolder(root=join(
                    self.projectParams.dataPath, stage), transform=self.transformDict[stage])
                if self.projectParams.maxFiles is not None:
                    self.dataset[stage] = random_split(dataset=self.dataset[stage], lengths=(
                        self.projectParams.maxFiles, len(self.dataset[stage])-self.projectParams.maxFiles))[0]
        else:
            taskDict = {'cifar10': 'CIFAR10', 'mnist': 'MNIST'}
            trainSet = eval('{}(root=self.projectParams.dataPath, train=True, download=True, transform=self.transformDict["train"])'.format(
                taskDict[self.projectParams.predefinedTask]))
            testSet = eval('{}(root=self.projectParams.dataPath, train=False, download=True, transform=self.transformDict["test"])'.format(
                taskDict[self.projectParams.predefinedTask]))
            # modify the maximum number of files
            for dSet in [trainSet, testSet]:
                dSet.data = dSet.data[:self.projectParams.maxFiles]
                dSet.targets = dSet.targets[:self.projectParams.maxFiles]
            trainValSize = [int((1-self.projectParams.valSize)*len(trainSet)),
                            int(self.projectParams.valSize*len(trainSet))]
            trainSet, valSet = random_split(
                dataset=trainSet, lengths=trainValSize)
            self.dataset = {'train': trainSet,
                            'val': valSet, 'test': testSet}
            # get the dataType from the testSet
            self.projectParams.dataType = testSet.class_to_idx

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.dataset['train'], batch_size=self.projectParams.batchSize, shuffle=True, pin_memory=self.projectParams.useCuda, num_workers=projectParams.numWorkers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.dataset['val'], batch_size=self.projectParams.batchSize, shuffle=False, pin_memory=self.projectParams.useCuda, num_workers=projectParams.numWorkers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.dataset['test'], batch_size=self.projectParams.batchSize, shuffle=False, pin_memory=self.projectParams.useCuda, num_workers=projectParams.numWorkers)


if __name__ == "__main__":
    # project parameters
    projectParams = ProjectPrameters().parse()

    # get dataset
    dataset = MyDataModule(projectParams=projectParams)

    # get data loader
    dataset.prepare_data()
    trainLoader = dataset.train_dataloader()
    valLoader = dataset.val_dataloader()
    testLoader = dataset.test_dataloader()

    # display the dataset information
    for stage in ['train', 'val', 'test']:
        print(stage, dataset.dataset[stage])

# import
import torch
from src.project_parameters import ProjectPrameters
from src.model import create_model
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

# class


class Predict:
    def __init__(self, projectParams):
        self.projectParams = projectParams
        self.model = create_model(projectParams=projectParams).eval()
        self.transform = transforms.Compose([transforms.Resize((projectParams.maxImageSize[0], projectParams.maxImageSize[1])),
                                             transforms.ToTensor()])

    def get_result(self, dataPath):
        result = []
        if '.png' in dataPath or '.jpg' in dataPath:
            img = Image.open(dataPath).convert('RGB')
            img = self.transform(img)[None, :]
            with torch.no_grad():
                result.append(self.model(img).tolist()[0])
        else:
            dataset = ImageFolder(dataPath, transform=self.transform)
            dataLoader = DataLoader(dataset, batch_size=self.projectParams.batchSize,
                                    pin_memory=self.projectParams.useCuda, num_workers=self.projectParams.numWorkers)
            with torch.no_grad():
                for img, _ in dataLoader:
                    result.append(self.model(img).tolist())
        return np.concatenate(result, 0).round(2)


if __name__ == '__main__':
    # project parameters
    projectParams = ProjectPrameters().parse()

    # predict
    result = Predict(projectParams=projectParams).get_result(
        dataPath=projectParams.dataPath)
    # use [:-1] to remove the latest comma
    print(('{},'*projectParams.numClasses).format(*
                                                  projectParams.classes.keys())[:-1])
    print(result)

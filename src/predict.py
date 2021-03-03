# import
import torch
from src.project_parameters import ProjectPrameters
from src.model import create_model
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

# def


def predict(projectParams):
    # get model
    model = create_model(projectParams=projectParams)
    model.eval()

    # load image and transfrom it
    result = []
    transform = transforms.Compose([transforms.Resize((projectParams.maxImageSize[0], projectParams.maxImageSize[1])),
                                    transforms.ToTensor()])
    if '.png' in projectParams.dataPath:
        img = Image.open(projectParams.dataPath).convert('RGB')
        img = transform(img)[None, :]
        with torch.no_grad():
            result.append(model(img).tolist()[0])
    else:
        dataset = ImageFolder(projectParams.dataPath, transform=transform)
        dataLoader = DataLoader(dataset, batch_size=projectParams.batchSize,
                                pin_memory=projectParams.useCuda, num_workers=projectParams.numWorkers)
        with torch.no_grad():
            for img, _ in dataLoader:
                result.append(model(img).tolist())
    return np.concatenate(result, 0).round(2)


if __name__ == '__main__':
    # project parameters
    projectParams = ProjectPrameters().parse()

    # predict
    result = predict(projectParams=projectParams)
    print(('{},'*projectParams.numClasses).format(*projectParams.dataType.keys()))
    print(result)

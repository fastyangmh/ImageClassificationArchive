# import
from PIL import Image
import torch
from src.project_parameters import ProjectPrameters
from src.model import Net, create_model
from torchvision import transforms

# def


def predict(projectParams):
    # get model
    model = create_model(projectParams=projectParams)
    model.eval()

    # load image and transfrom it
    transform = transforms.Compose([transforms.Resize((projectParams.maxImageSize[0], projectParams.maxImageSize[1])),
                                    transforms.ToTensor()])
    img = Image.open(projectParams.dataPath).convert('RGB')
    img = transform(img)[None, :]
    with torch.no_grad():
        result = model(img).tolist()[0]
    return result


if __name__ == '__main__':
    # project parameters
    projectParams = ProjectPrameters().parse()

    # predict
    result = predict(projectParams=projectParams)
    print(('{},'*projectParams.numClasses).format(*projectParams.dataType.keys()))
    print(result)

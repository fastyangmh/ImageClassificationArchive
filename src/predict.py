# import
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets.folder import ImageFolder
from src.utils import get_transform_from_file
from src.model import create_model
from src.project_parameters import ProjectParameters
from PIL import Image
import numpy as np

# class


class Predict:
    """Class method for Predict .
    """

    def __init__(self, project_parameters):
        """Initialize the class.

        Args:
            project_parameters (argparse.Namespace): the parameters for the project.
        """
        self.project_parameters = project_parameters
        self.model = create_model(project_parameters=project_parameters).eval()
        self.transform = get_transform_from_file(
            filepath=project_parameters.transform_config_path)['predict']

    def __call__(self, data_path):
        """Predict the given the data_path of data.

        Args:
            data_path (str): the data path.

        Returns:
            numpy.ndarray: the model predicted result.
        """
        result = []
        if '.png' in data_path or '.jpg' in data_path:
            image = Image.open(fp=data_path).convert('RGB')
            image = self.transform(image)[None, :]
            with torch.no_grad():
                result.append(self.model(image).tolist()[0])
        else:
            dataset = ImageFolder(root=data_path, transform=self.transform)
            data_loader = DataLoader(dataset=dataset, batch_size=self.project_parameters.batch_size,
                                     pin_memory=self.project_parameters.use_cuda, num_workers=self.project_parameters.num_workers)
            with torch.no_grad():
                for image, _ in data_loader:
                    result.append(self.model(image).tolist())
        return np.concatenate(result, 0).round(2)


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # predict the data path
    result = Predict(project_parameters=project_parameters)(
        data_path=project_parameters.data_path)
    # use [:-1] to remove the latest comma
    print(('{},'*project_parameters.num_classes).format(*
                                                        project_parameters.classes)[:-1])
    print(result)

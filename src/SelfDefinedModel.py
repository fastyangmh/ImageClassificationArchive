# import
import torch
import torch.nn as nn
from torch.hub import load

# class


class SelfDefinedModel(nn.Module):
    """Creates a self-defined model .
    """

    def __init__(self, in_chans, num_classes):
        """Initialize the class.

        Args:
            in_chans (int): number of input channels / colors.
            num_classes (int): the number of classes.
        """
        super(SelfDefinedModel, self).__init__()
        self.classifier = load(
            repo_or_dir='huawei-noah/ghostnet', model='ghostnet_1x', pretrained=True, progress=True)
        self.classifier.conv_stem = nn.Conv2d(in_channels=in_chans, out_channels=self.classifier.conv_stem.out_channels, kernel_size=self.classifier.conv_stem.kernel_size,
                                              stride=self.classifier.conv_stem.stride, padding=self.classifier.conv_stem.padding, bias=self.classifier.conv_stem.bias)
        self.classifier.classifier = nn.Linear(
            in_features=self.classifier.classifier.in_features, out_features=num_classes)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): the input data.

        Returns:
            torch.Tensor: the predict of neural network model.
        """
        return self.classifier(x)


if __name__ == '__main__':
    # parameters
    in_chans = 3
    num_classes = 10

    # create model
    model = SelfDefinedModel(in_chans=in_chans, num_classes=num_classes)

    # create input data
    x = torch.ones(1, in_chans, 224, 224)

    # get model output
    y = model(x)

    # display the dimension of input and output
    print(x.shape)
    print(y.shape)

# import
import torch
import torch.nn as nn
from torch.hub import load

# class


class GhostNet(nn.Module):
    def __init__(self):
        super(GhostNet, self).__init__()
        self.classifier = load(
            repo_or_dir='huawei-noah/ghostnet', model='ghostnet_1x', pretrained=True, progress=True)
        self.classifier.classifier = nn.Linear(
            in_features=self.classifier.classifier.in_features, out_features=2)

    def forward(self, x):
        return self.classifier(x)


if __name__ == '__main__':
    # create model
    model = GhostNet()

    # create input data
    x = torch.ones(1, 3, 224, 224)

    # get model output
    y = model(x)

    # display the dimension of input and output
    print(x.shape)
    print(y.shape)

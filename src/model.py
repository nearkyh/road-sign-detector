import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        resnet = resnet34(pretrained=True)
        layers = list(resnet.children())[:8]
        self.features1 = nn.Sequential(*layers[:6])
        self.features2 = nn.Sequential(*layers[6:])

        self.clf = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))
        self.reg = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = F.relu(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = x.view(x.shape[0], -1)

        return self.clf(x), self.reg(x)

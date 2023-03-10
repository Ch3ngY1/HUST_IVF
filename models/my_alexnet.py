from torchvision.models import alexnet
import torch
import torch.nn as nn

class Alex_(nn.Module):
    def __init__(self, num_classes=1):
        super(Alex_, self).__init__()
        # self.first = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
        self.basemodel = alexnet(pretrained=True)
        # self.features = nn.Sequential(self.first, *list(self.basemodel.features.children())[1:])
        self.features = self.basemodel.features
        self.classifier = nn.Sequential(*list(self.basemodel.classifier.children())[:-1], nn.Linear(4096, num_classes))
        self.classifier[0] = nn.Dropout(p=0.4, inplace=False)
        self.classifier[3] = nn.Dropout(p=0.4, inplace=False)
        # self.classifier
        # self.classifier = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(256 * 6 * 6, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, num_classes),
        # )

    def forward(self, input):
        # x = self.first(input)
        x = self.features(input)
        x = x.view(-1, 256*6*6)
        x = self.classifier(x)
        return x

import torch

import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes, head):
        super(VGG, self).__init__()
        self.features = features
        self._head = head
        if self._head:
            self.classifier = nn.Sequential(
                # nn.Linear(512 * 7 * 7, 4096),
                # nn.ReLU(inplace=True),
                # nn.Dropout(),
                # nn.Linear(4096, 4096),
                # nn.ReLU(inplace=True),
                # nn.Dropout(),
                # nn.Linear(4096, num_classes),

                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(inplace=True),
                # nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                # nn.Dropout(),
                nn.Linear(4096, num_classes),
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if self._head:
            x = self.classifier(x)
        return x

    def fix_model(self):
        # first block is frozen
        # warning: only useful for vgg16
        fix_modules = [self.features[x] for x in range(4)]
        for md in fix_modules:
            for name, param in md.named_parameters():
                param.requires_grad = False

    def froze_bn(self):
        froze_modules = self.modules()
        self._freeze_bn(froze_modules)

    def _freeze_bn(self, froze_modules):
        for md in froze_modules:
            if isinstance(md, nn.BatchNorm2d):
                # print(md)
                md.eval()
                for param in md.parameters():
                    param.requires_grad = False
            elif isinstance(md, nn.Sequential):
                self._freeze_bn(md)


    def load_state_dict(self, state_dict):
        from collections import OrderedDict
        own_state = self.state_dict()
        third_state_old = state_dict
        third_state = OrderedDict()  # remove all 'group' string
        add_num = 0
        last_part2 = '0'
        now_part2 = '0'

        for name, param in third_state_old.items():
            part1, part2, part3 = name.split('.')
            now_part2 = part2
            if part1 != 'classifier':
                if now_part2 != last_part2:
                    add_num += 1
                new_num_part2 = int(part2) + add_num
                new_tmp_part2 = str(new_num_part2)
                new_name = name.replace(part2, new_tmp_part2)
            else:
                new_name = name
            last_part2 = now_part2
            third_state[new_name] = param

        for name, param in third_state.items():
            if name not in own_state:
                raise KeyError('unexpected key "{}" in state_dict'.format(name))

            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
            print('{} has been loaded!'.format(name))

    def load_model(self, model_path):
        pre_model = torch.load(model_path)
        pre_model_dict = {k: v for k, v in pre_model.items() if ('classifier' not in k)}
        self.load_state_dict(pre_model_dict)

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'A':
            layers += [nn.AdaptiveMaxPool2d((7, 7))]
            # layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'A'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'A'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'A'],
}

def vgg11(num_classes=1000, init_function=None, head=True):
    """VGG 11-layer model (configuration "A")"""
    model = VGG(make_layers(cfg['A'], batch_norm=False), num_classes, head)
    model.apply(init_function)
    return model


def vgg11_bn(num_classes=1000, init_function=None, head=True):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG(make_layers(cfg['A'], batch_norm=True), num_classes, head)
    model.apply(init_function)
    return model

def vgg13(num_classes=1000, init_function=None, head=True):
    """VGG 13-layer model (configuration "B")"""
    model = VGG(make_layers(cfg['B'], batch_norm=False), num_classes, head)
    model.apply(init_function)
    return model


def vgg13_bn(num_classes=1000, init_function=None, head=True):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG(make_layers(cfg['B'], batch_norm=True), num_classes, head)
    model.apply(init_function)
    return model


def vgg16(num_classes=1000, init_function=None, head=True):
    """VGG 16-layer model (configuration "D")"""
    model = VGG(make_layers(cfg['D'], batch_norm=False), num_classes, head)
    model.apply(init_function)
    return model


def vgg16_bn(num_classes=1000, init_function=None, head=True):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(make_layers(cfg['D'], batch_norm=True), num_classes, head)
    model.apply(init_function)
    return model

def vgg19(num_classes=1000, init_function=None, head=True):
    """VGG 19-layer model (configuration "E")"""
    model = VGG(make_layers(cfg['E'], batch_norm=False), num_classes, head)
    model.apply(init_function)
    return model


def vgg19_bn(num_classes=1000, init_function=None, head=True):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG(make_layers(cfg['E'], batch_norm=True), num_classes, head)
    model.apply(init_function)
    return model
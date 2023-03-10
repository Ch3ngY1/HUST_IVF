import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-241335ed.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-6f0f7f60.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-4c113574.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-17b70270.pth',
}


def densenet121(num_classes=1000, init_function=None, head=True):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_classes, head, num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16))
    model.apply(init_function)
    return model


def densenet169(num_classes=1000, init_function=None, head=True):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_classes, head, num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32))
    model.apply(init_function)
    return model

def densenet201(num_classes=1000, init_function=None, head=True):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_classes, head, num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32))
    model.apply(init_function)
    return model

def densenet161(num_classes=1000, init_function=None, head=True):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_classes, head, num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24))
    model.apply(init_function)
    return model


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, num_classes, head, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0):

        super(DenseNet, self).__init__()
        self.channels_per_layer = []
        self._head = head

        # First convolution
        m = OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ])
        # self.features = nn.Sequential(m)
        self.group1 = nn.Sequential(m)

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            setattr(self, 'layer{:s}'.format(str(i + 1)), nn.Sequential(OrderedDict()))
            getattr(self, 'layer{:s}'.format(str(i + 1))).add_module('denseblock%d' % (i + 1), block)
            # self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                # self.features.add_module('transition%d' % (i + 1), trans)
                getattr(self, 'layer{:s}'.format(str(i + 1))).add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
            self.channels_per_layer.append(num_features)

        # Final batch norm
        # self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.layer4.add_module('norm5', nn.BatchNorm2d(num_features))
        self.layer4.add_module('final_relu', nn.ReLU(inplace=True))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.out_channels = num_features
        self.forward_list = ['group1', 'layer1', 'layer2', 'layer3', 'layer4']

        # Linear layer
        if self._head:
            self.classifier = nn.Linear(num_features, num_classes)


    def forward(self, x):
        # for i in range(10):
        #     x = self.features[i](x)
        #     print(x.size())
        # assert 1==0

        # features = self.features(x)
        features = self.group1(x)
        for i in range(4):
            features = getattr(self, 'layer{:s}'.format(str(i + 1)))(features)
        out = self.avg_pool(features).view(features.size(0), -1)
        if self._head:
            out = self.classifier(out)
        return out

    def froze_bn(self):
        froze_modules = [self.group1, self.layer1, self.layer2,
                         self.layer3, self.layer4]
        self._freeze_bn(froze_modules)

    def _freeze_bn(self, froze_modules):
        for md in froze_modules:
            if isinstance(md, nn.BatchNorm2d):
                md.eval()
                for param in md.parameters():
                    param.requires_grad = False
            elif isinstance(md, nn.Sequential):
                self._freeze_bn(md)

    def fix_model(self):
        # fix first layer and block1
        fix_modules = [self.group1, self.layer1]
        for md in fix_modules:
            for param in md.parameters():
                param.requires_grad = False

    def load_official_state_dict(self, state_dict):
        import re
        from collections import OrderedDict
        own_state_old = self.state_dict()
        own_state = OrderedDict()  # remove all 'group' string
        new_state_dict = OrderedDict()
        for k, v in own_state_old.items():
            k = re.sub('group\d+\.', '', k)
            k = re.sub('layer\d+\.', '', k, count=1)
            own_state[k] = v

        for k, v in state_dict.items():
            k = re.sub('features+\.', '', k)
            new_state_dict[k] = v

        for name, param in new_state_dict.items():
            if name not in own_state:
                print(name)
                raise ValueError

            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

    def load_state_dict(self, state_dict):
        import re
        from collections import OrderedDict
        own_state = self.state_dict()
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            k = re.sub('model+\.', '', k)
            new_state_dict[k] = v

        for name, param in new_state_dict.items():
            if name not in own_state:
                print(name)
                raise ValueError

            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

    def load_model(self, model_path):
        pre_model = torch.load(model_path)
        if model_path.count('.torch') != 0:
            pass
        else:
            pre_model = pre_model['state_dict']
        pre_model_dict = {k: v for k, v in pre_model.items() if ('classifier' not in k)}
        if model_path.count('.torch') != 0:
            self.load_official_state_dict(pre_model_dict)
        else:
            self.load_state_dict(pre_model_dict)
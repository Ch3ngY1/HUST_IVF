from models.p3d_model import P3D199
import torch.nn as nn
from models.resnet import resnet50
import torch
import torch.nn.functional as F

'''
1. Input = Image + position = N * [128*224*224 + 512*128]
2. Net = ResNet + LSTM
3. concat(ResNet(Image), position) --> LSTM --> 4FC
4. Final frame --> prediction
5. Loss = Loss_pred + Loss_utility - Loss_Reward
'''


# init_func = {'weights_init_xavier': weights_init_xavier,
#              'weights_init_he': weights_init_he,
#              'weights_init_normal': weights_init_normal}


class myLSTM(nn.Module):
    def __init__(self, num_frame=36, position_mode='cat'):
        super(myLSTM, self).__init__()
        self.num_class = 2
        self.num_frame = num_frame
        self.feature_len = 2048
        # TODO: positional encoding usage
        if position_mode == 'add':
            self.inshape = self.feature_len
        elif position_mode == 'cat':
            self.inshape = self.feature_len * 2
        else:
            raise KeyError('Unknown mode: {}'.format(position_mode))
        self.lstm = nn.GRUCell(self.inshape, self.feature_len)
        self.fc_pred = nn.Linear(self.feature_len, self.num_class, bias=True)
        self.fc_utility = nn.Linear(self.feature_len, 1, bias=True)
        self.fc_use = nn.Linear(self.feature_len, 2, bias=True)


    def forward(self, feature):
        # feature : N * 128 * 4096
        feature = feature.transpose(1, 0)

        # --> 128 * N * 4096
        # TODO: use cell as output: slow change, more long-time information
        hidden = []
        utility = []
        watch = []
        for i in range(self.num_frame):
            lstm_in = feature[i]
            if i == 0:
                h_x = self.lstm(lstm_in)  # h_x: Nxhidden, c_x: Nxhidden
            else:
                previous_state = h_x
                h_x = self.lstm(lstm_in, h_x)  # h_x: Nxhidden, c_x: Nxhidden
                use = self.fc_use(h_x)
                use = F.gumbel_softmax(use, tau=1, hard=False)
                # matrix multiple, use gumbel softmax

                watch.append(torch.argmax(use))
                # TODO: test not use h_x choice
                # h_x = torch.bmm(torch.stack([previous_state, h_x], dim=-1), use.unsqueeze(dim=-1)).squeeze(-1)

                # h_x = [h_x_previous, h_x] * use
            hidden.append(self.fc_pred(h_x))
            utility.append(self.fc_utility(h_x))

        return hidden, utility


class Ada(nn.Module):
    def __init__(self, init_function, num_classes=2, num_frame=96, position_mode='add'):
        super(Ada, self).__init__()
        self.group1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.basemodel = resnet50(num_classes=2, init_function=init_function)
        self.postition_mode = position_mode
        self.lstm = myLSTM(num_frame=num_frame, position_mode=position_mode)
        self.num_classs = num_classes

    def forward(self, input):
        img = input[0]

        position = input[1]
        feature = self.feature_extraction(self.feature, img)
        # shape =
        if self.postition_mode == 'cat':
            lstmin = torch.cat([feature, position], dim=2)
        elif self.postition_mode == 'add':
            lstmin = feature + position.cuda()
        # if self.c_out:
        #     pred, utility, pred_c = self.lstm(lstmin)
        #     return pred, utility, pred_c
        hidden, utility = self.lstm(lstmin)
        return hidden, utility

    def load_model(self, model_path):
        self.basemodel.load_model(model_path)
        self.feature = nn.Sequential(self.group1, *list(self.basemodel.children())[1:-1])

    def load_state_dict(self, state_dict):
        from torch.utils import model_zoo
        from torch import nn
        import re
        from collections import OrderedDict
        own_state_old = self.state_dict()
        own_state = OrderedDict()  # remove all 'group' string
        new_state = OrderedDict()  # remove all 'group' string
        for k, v in own_state_old.items():
            k = re.sub('group\d+\.', '', k)
            own_state[k] = v

        for k, v in state_dict.items():
            k = re.sub('module\.', '', k)
            new_state[k] = v

        for name, param in new_state.items():
            if name not in own_state or name.count('fc') != 0:
                print(name)
                continue

            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

        missing = (set(new_state.keys()) - set(own_state.keys())) | (set(own_state.keys()) - set(new_state.keys()))
        print('missing keys in state_dict: '.format(missing))

    def feature_extraction(self, model, img):
        # image = [N, 128, 1 ,224, 224]
        # net: 2D CNN
        b, f, c, h, w = img.shape
        img = img.view(b * f, c, h, w)  # --> (128 * N) * 1 * 224 * 224

        feature = model(img)
        feature = feature.view(b, f, -1)
        return feature

    def parallel(self, gpus):
        self.feature = torch.nn.DataParallel(self.feature, device_ids=list(range(len(gpus))))

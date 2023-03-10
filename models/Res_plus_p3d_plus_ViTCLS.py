from models.p3d_model import P3D199
import torch.nn as nn
from models.resnet import resnet50
import torch
import torch.nn.functional as F
from models.ViT import ViT_encode, ViT_cls
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
    def __init__(self, num_frame=32, c_trans=False):
        super(myLSTM, self).__init__()
        self.num_class = 2
        self.num_frame = num_frame
        self.feature_len = 2048
        self.inshape = self.feature_len
        # TODO: positional encoding usage

        self.lstm = nn.LSTMCell(self.inshape, self.feature_len)
        self.fc_pred = nn.Linear(self.feature_len, self.num_class, bias=True)
        self.fc_pred_c = nn.Linear(self.feature_len, self.num_class, bias=True)
        self.fc_utility = nn.Linear(self.feature_len, 1, bias=True)
        self.fc_use = nn.Linear(self.feature_len, 2, bias=True)

        self.c_trans = c_trans

    def forward(self, feature):
        # feature : N * 128 * 4096
        feature = feature.transpose(1, 0)

        # --> 128 * N * 4096
        # TODO: use cell as output: slow change, more long-time information
        hidden = []
        cell = []
        utility = []
        watch = []
        for i in range(self.num_frame):
            lstm_in = feature[i]
            if i == 0:
                h_x, c_x = self.lstm(lstm_in)  # h_x: Nxhidden, c_x: Nxhidden
            else:
                if self.c_trans:
                    previous_state = c_x
                else:
                    previous_state = h_x
                h_x, c_x = self.lstm(lstm_in, (h_x, c_x))  # h_x: Nxhidden, c_x: Nxhidden
                use = self.fc_use(h_x)
                use = F.gumbel_softmax(use, tau=1, hard=False)
                # matrix multiple, use gumbel softmax

                watch.append(torch.argmax(use))
                # TODO: 修改传递的h_x，因为use针对本帧，因此还是使用hidden作为use的计算量
                if self.c_trans:
                    h_x = torch.bmm(torch.stack([previous_state, c_x], dim=-1), use.unsqueeze(dim=-1)).squeeze(-1)
                else:
                    h_x = torch.bmm(torch.stack([previous_state, h_x], dim=-1), use.unsqueeze(dim=-1)).squeeze(-1)
                # h_x = [h_x_previous, h_x] * use
            cell.append(self.fc_pred_c(c_x))
            hidden.append(self.fc_pred(h_x))
            utility.append(self.fc_utility(h_x))
        # print(torch.tensor(watch).sum()/self.num_frame)
        # if self.output_c:
        #     return cell, utility
        # TODO: always output hidden, cell, utility
        return hidden, cell, utility


class Ada(nn.Module):
    def __init__(self, init_function, num_classes=2, num_frame=96):
        super(Ada, self).__init__()
        self.group1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.basemodel = resnet50(num_classes=2, init_function=init_function)

        self.num_classs = num_classes
        self.cls = ViT_cls(
            frames=num_frame,
            num_classes=self.num_classs,
            dim=2048,
            depth=6,
            heads=16,
            mlp_dim=4096,
            dropout=0.1,
            emb_dropout=0.1
        )

    def forward(self, input):
        img = input[0]

        feature = self.feature_extraction(self.feature, img)
        pred = self.cls(feature)

        return pred

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


class Ada_LSTM(nn.Module):
    def __init__(self, init_function, num_classes=2, num_frame=96):
        super(Ada_LSTM, self).__init__()
        self.group1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.basemodel = resnet50(num_classes=2, init_function=init_function)

        self.num_classs = num_classes
        self.encode = ViT_encode(
            frames=num_frame,
            num_classes=self.num_classs,
            dim=2048,
            depth=6,
            heads=16,
            mlp_dim=4096,
            dropout=0.1,
            emb_dropout=0.1
        )
        self.lstm = myLSTM(num_frame=num_frame)

    def forward(self, input):
        img = input[0]

        feature = self.feature_extraction(self.feature, img)
        feature = self.encode(feature)
        hidden, cell, utility = self.lstm(feature)
        return hidden, cell, utility


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

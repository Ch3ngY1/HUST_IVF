from models.p3d_model import P3D199
import torch.nn as nn
from models.resnet import resnet50
import torch
import torch.nn.functional as F
import math
from models import Action

'''
1. Input = Image + position = N * [128*224*224 + 512*128]
2. Net = ResNet + LSTM
3. concat(ResNet(Image), position) --> LSTM --> 4FC
4. Final frame --> prediction
5. Loss = Loss_pred + Loss_utility - Loss_Reward
'''


def weights_init_normal(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


# init_func = {'weights_init_xavier': weights_init_xavier,
#              'weights_init_he': weights_init_he,
#              'weights_init_normal': weights_init_normal}


class myLSTM(nn.Module):
    def __init__(self, num_frame=96, position_mode='cat', c_out=False):
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
        self.lstm = nn.LSTMCell(self.inshape, self.feature_len)
        self.fc_pred = nn.Linear(self.feature_len, self.num_class, bias=True)
        self.fc_pred_c = nn.Linear(self.feature_len, self.num_class, bias=True)
        self.fc_utility = nn.Linear(self.feature_len, 1, bias=True)
        self.fc_use = nn.Linear(self.feature_len, 2, bias=True)

        self.output_c = c_out

    def forward(self, feature):
        # feature : N * 128 * 4096
        feature = feature.transpose(1, 0)

        # --> 128 * N * 4096
        # TODO: use cell as output: slow change, more long-time information
        pred = []
        cell = []
        utility = []
        watch = []
        for i in range(self.num_frame):
            lstm_in = feature[i]
            if i == 0:
                h_x, c_x = self.lstm(lstm_in)  # h_x: Nxhidden, c_x: Nxhidden
            else:
                h_x_previous = h_x
                # c_x_previous = c_x
                h_x, c_x = self.lstm(lstm_in, (h_x, c_x))  # h_x: Nxhidden, c_x: Nxhidden
                # TODO: USE HIDDEN as next input
                use = self.fc_use(h_x)
                use = F.gumbel_softmax(use, tau=1, hard=False)
                # matrix multiple, use gumbel softmax
                watch.append(torch.argmax(use))
                h_x = torch.bmm(torch.stack([h_x_previous, h_x], dim=-1), use.unsqueeze(dim=-1)).squeeze(-1)
                # TODO: USE CELL as next input
                # use = self.fc_use(c_x)
                # use = F.gumbel_softmax(use, tau=1, hard=False)
                # # matrix multiple, use gumbel softmax
                # watch.append(torch.argmax(use))
                # h_x = torch.bmm(torch.stack([c_x_previous, c_x], dim=-1), use.unsqueeze(dim=-1)).squeeze(-1)


            cell.append(self.fc_pred_c(c_x))
            pred.append(self.fc_pred(h_x))
            utility.append(self.fc_utility(h_x))
        # print(torch.tensor(watch).sum()/self.num_frame)
        if self.output_c:
            return cell, utility
        return pred, utility


class Feature(nn.Module):
    def __init__(self, init_function, action):
        super(Feature, self).__init__()
        self.basemodel = resnet50(num_classes=2, init_function=init_function)
        self.basemodel.load_model('/data2/chengyi/.torch/models/resnet50-19c8e357.pth')
        self.group1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer1 = list(self.basemodel.children())[1]
        self.layer2 = list(self.basemodel.children())[2]
        self.layer3 = list(self.basemodel.children())[3]
        self.layer4 = list(self.basemodel.children())[4]
        self.avg_pool = list(self.basemodel.children())[5]
        # self.group2 = list(self.basemodel.children())[6]
        # self.layer5 = list(self.basemodel.children())[5]
        self.action = action
        if action:
            self.action1 = Action.Action(in_channels=256)
            self.action2 = Action.Action(in_channels=512)
            self.action3 = Action.Action(in_channels=1024)
            self.action4 = Action.Action(in_channels=2048)

    def forward(self, x):
        # input = b, f, c, h, w = img.shape
        b, f, c, h, w = x.shape
        x = x.view(b * f, c, h, w)
        x = self.group1(x)

        x = self.layer1(x)
        if self.action:
            _, c, h, w = x.shape
            x = x.view(b, f, c, h, w)
            x = self.action1(x)
        x = self.layer2(x)
        if self.action:
            _, c, h, w = x.shape
            x = x.view(b, f, c, h, w)
            x = self.action2(x)
        x = self.layer3(x)
        if self.action:
            _, c, h, w = x.shape
            x = x.view(b, f, c, h, w)
            x = self.action3(x)
        x = self.layer4(x)
        if self.action:
            _, c, h, w = x.shape
            x = x.view(b, f, c, h, w)
            x = self.action4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)

        # x = self.group2(x)
        return x


class Ada_test(nn.Module):
    def __init__(self, init_function, num_classes=2, num_frame=96, position_mode='add', c_out=False, action=False):
        super(Ada_test, self).__init__()
        self.group1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        if action:
            print('ACTION IS ACTIVATED')
        else:
            print('ACTION IS NOT ACTIVATED')
        self.feature = Feature(init_function=init_function, action=action)

        self.postition_mode = position_mode
        self.c_out = c_out
        self.lstm = myLSTM(num_frame=num_frame, position_mode=position_mode, c_out=self.c_out)
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
        pred, utility = self.lstm(lstmin)
        return pred, utility

    def load_model(self, model_path):
        pass

    def feature_extraction(self, model, img):
        # image = [N, 128, 1 ,224, 224]
        # net: 2D CNN
        b, f, c, h, w = img.shape
        # img = img.view(b * f, c, h, w)  # --> (128 * N) * 1 * 224 * 224

        feature = model(img)
        feature = feature.view(b, f, -1)
        return feature

    def parallel(self, gpus):
        self.feature = torch.nn.DataParallel(self.feature, device_ids=list(range(len(gpus))))


if __name__ == '__main__':
    blob = [torch.rand([1, 32, 1, 224, 224]), torch.rand([1, 32, 2048])]
    net = Ada_test(weights_init_normal, num_classes=2, num_frame=36, position_mode='add', c_out=True)
    out = net(blob)

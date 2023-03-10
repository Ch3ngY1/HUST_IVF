import torch
import numpy as np
from sklearn import metrics
import sys
sys.path.append("../")
from utils import my_dataloader, my_parse, utils
import os
from torchvision.models import densenet201
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# os.environ['CUDA_VISIBLE_DEVICES']='1'
args = my_parse.parse_args()
weight = np.linspace(start=0, stop=1, num=101)
class Temporal_Model(nn.Module):
    def __init__(self):
        super(Temporal_Model, self).__init__()
        self.embedding_indice = nn.Embedding(num_embeddings=900, embedding_dim=256)
        self.embedding_cell = nn.Embedding(num_embeddings=5, embedding_dim=256)
        self.lstm = nn.LSTM(input_size=512, hidden_size=256)
        self.drop = nn.Dropout(p=0.5)
        self.ac = nn.Sigmoid()
        self.fc = nn.Linear(256, 2)
        # self.base.classifier = nn.Linear(in_features=1920, out_features=1, bias=True)

    def forward(self, input):
        # [cells, imgs]
        cells, imgs = input
        cells = self.embedding_cell(cells)
        # imgs = imgs-imgs.min()
        imgs = self.embedding_indice(imgs)
        data = torch.cat([imgs, cells], dim=2)
        # x = self.embedding(input)
        x = self.lstm(data)
        x = x[0][:,-1,:]
        x = self.drop(x)
        x = self.fc(x)
        x = self.ac(x)
        return x


class Spatial_Model(nn.Module):
    def __init__(self):
        super(Spatial_Model, self).__init__()
        self.base = densenet201(pretrained=True)
        self.base.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.base.classifier = nn.Linear(in_features=1920, out_features=1, bias=True)
        self.fc1 = nn.Linear(in_features=1920*35, out_features=2, bias=True)
        self.fc2 = nn.Linear(in_features=1920, out_features=2, bias=True)

    def forward(self, input):
        # input.shape = batch * 7 * 5 * 224 * 224
        bs, f, p, h, w = input.shape
        input = input.view(bs*f*p,1,h,w)
        # input = input.view()
        feature = self.base.features(input)
        feature = F.relu(feature, inplace=True)
        feature = F.avg_pool2d(feature, kernel_size=7, stride=1)
        feature = feature.view(bs,-1)
        output = self.fc1(feature)

        # feature = feature.view(bs, 35, -1)
        # feature = feature.mean(dim=1)
        # output = self.fc2(feature)
        # input.shape = (batch * 7 * 5) * 224 * 224
        # 1. 直接cat在一起分类
        # 2. avg以后分类
        return output


temporal_file = '/data2/chengyi/myproject/Savings/save_models/Feb06_03-38-03_Ada_Ada_real-temporal-new/best_val_acc.pth'
spatial_file = '/data2/chengyi/myproject/Savings/save_models/temporal_direct_DenseNet_concat/best_val_acc.pth'

temporal_model = torch.load(temporal_file)['net'].cuda()
spatial_model = torch.load(spatial_file)['net'].cuda()

val_dataloader = my_dataloader.my_dataloader_res_comparison(
    root='/data2/chengyi/embryo_HUSH/first_travel_VideoImage/',
    filename=['valid_FOR_STEM_FINAL_RE.json','valid_PNF_TEMPORAL_NEW.json'], args=args, emsemble=True, train=False)

GPU=1
predict_list_spatial_prob, predict_temporal_prob, target_list = [], [], []
for batch_idx, (blobs, targets, imgs, blobs_spatial) in enumerate(val_dataloader):
    with torch.no_grad():
        cls_targets = utils.label_to_variable(targets, gpu=GPU, volatile=False)


        imgs = utils.label_to_variable(imgs, gpu=GPU, volatile=False)
        blobs = utils.img_data_to_variable(blobs, gpu=GPU)
        tempoal_cls_preds = temporal_model([blobs, imgs])


        blobs_spatial = utils.img_data_to_variable(blobs_spatial, gpu=GPU)
        spatial_cls_preds = spatial_model(blobs_spatial)

        # pred = (tempoal_cls_preds+spatial_cls_preds)/2


        predict_list_spatial_prob += list(spatial_cls_preds.data.cpu().numpy())
        predict_temporal_prob += list(tempoal_cls_preds.data.cpu().numpy())

        # pred_temporal = cls_preds.data.max(1)[1]  # get the index of the max log-probability
        # pred_temporal = cls_preds.data.max(1)[1]  # get the index of the max log-probability
        #
        #
        # train_predict_list += list(pred.cpu().numpy())
        target_list += list(targets)


auc, recall, acc, f1 = [],[],[],[]
best_acc =0.0

for each_weight in weight:
    pred_prob = [each_weight*spa+(1-each_weight)*tem for spa,tem in zip(predict_list_spatial_prob, predict_temporal_prob)]
    pred_prob = np.array(pred_prob)
    predict_list = np.argmax(pred_prob, axis=1)
    # pred = [each.data.max(1)[1] for each in pred_prob]
    # predict_list = list(pred.cpu().numpy())

    target_list_onehot = np.eye(2)[target_list]
    auc.append(metrics.roc_auc_score(target_list_onehot, pred_prob))


    recall.append([metrics.recall_score(target_list, predict_list, average='macro', labels=[cls])
                for cls in range(2)])
    acc.append(metrics.accuracy_score(target_list, predict_list))
    f1.append(metrics.f1_score(target_list, predict_list, average='weighted'))


    if metrics.accuracy_score(target_list, predict_list) > best_acc:
        best_acc = metrics.accuracy_score(target_list, predict_list)
        best_recall = [metrics.recall_score(target_list, predict_list, average='macro', labels=[cls])
                for cls in range(2)]
        best_f1 = metrics.f1_score(target_list, predict_list, average='weighted')
        best_auc = metrics.roc_auc_score(target_list_onehot, pred_prob)
        fpr, tpr, thersholds = metrics.roc_curve(target_list, predict_list)
    # if abs(each_weight-0.5) < 0.01:
    #     print(auc[-1])
    #     print(recall[-1])
    #     print(acc[-1])
    #     print(f1[-1])



print('acc={}; recall={}; f1={}; auc={}'.format(best_acc, best_recall, best_f1, best_auc))
print(fpr)
print(tpr)
print('?')


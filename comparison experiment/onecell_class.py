import os
import sys
sys.path.append("../")
import json
from datetime import datetime
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch
from utils import my_parse, my_dataloader, my_loadmodel, my_loss, utils, my_optimizer, Log, savemodel, random_fix
from utils.config import cfg
import time
from sklearn import metrics
from torchvision.models import densenet201
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import torch.utils.data as data
from numpy.random import randint
from torch.utils.data import DataLoader
from utils.transforms import *
import utils.transforms_visual as tv
class FocalLoss(nn.Module):

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

def main():
    args = my_parse.parse_args()
    if args.random_fix:
        random_fix.seed_torch(seed=623)

    this_train_tag = datetime.now().strftime('%b%d_%H-%M-%S_') + args.net + '_' + args.modality + '_' + args.tag
    log_dir = os.path.join(cfg.TensorboardSave_Path, this_train_tag)
    tensorboard_writer = Log.LogSummary(log_dir)
    GPU = 1

    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus).split('[')[-1].split(']')[0]
    # to store data and weight in cuda: args.gpus[0] instead of cuda:0


    net = densenet201(pretrained=True)
    net.classifier = nn.Linear(in_features=1920, out_features=2, bias=True)


    loss = FocalLoss(class_num=2)


    train_dataloader = my_dataloader.my_dataloader_stemcellcount(root='/data2/chengyi/embryo_HUSH/first_travel_VideoImage/stem',
                                                       filename='train_.json', args=args, train=True, onecell=True)
    val_dataloader = my_dataloader.my_dataloader_stemcellcount(root='/data2/chengyi/embryo_HUSH/first_travel_VideoImage/stem',
                                                 filename='valid_.json', args=args, train=False, onecell=True)
    # TODO: optimizer =
    optimizer = my_optimizer.init_optimizer(net=net, args=args)
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Model\'s total number of parameters: %.3f M' % (num_params / 1e6))


    if GPU:
        net = net.cuda()

    best_val_acc = 0.0
    for epoch in range(args.epochs):

        train_predict_list, train_target_list = [], []
        net.train()
        train_loss_total = 0
        # Data
        for batch_idx, (blobs, targets) in enumerate(train_dataloader):
            # print(targets)
            cls_targets = utils.label_to_variable(targets, gpu=GPU, volatile=False)
            cls_targets = cls_targets.long()
            blobs = utils.img_data_to_variable(blobs, gpu=GPU)

            # forward
            t0 = time.time()

            # print(targets)
            cls_preds = net(blobs)
            train_loss_strategy = loss(cls_preds.squeeze(), cls_targets.squeeze())
            # TODO: mixup in loss function

            # if isinstance(cls_preds, tuple) or isinstance(cls_preds, list):
            #     pred = cls_preds[0].data.max(1)[1]  # get the index of the max log-probability
            # else:
            pred = cls_preds.data.max(1)[1]  # get the index of the max log-probability

            train_predict_list += list(pred.cpu().numpy())
            train_target_list += list(targets)

            optimizer.zero_grad()

            train_loss = train_loss_strategy
            train_loss_total = train_loss_total + train_loss
            train_loss.backward()
            optimizer.step()

            t1 = time.time()


            utils.progress_bar(batch_idx, len(train_dataloader), 'Epoch: %d | Loss: %.5f | Acc: %.3f'
                               % (epoch, (train_loss_total / (batch_idx + 1)),
                                  100. * metrics.accuracy_score(train_target_list, train_predict_list)))

            # break
        '''
        直接在writer scalar中定义记录部分
        (self, loss, lr, data, n_iter, tag=None):
        '''
        tensorboard_writer.write_scalars(train_loss_total / len(train_dataloader),
                                         args.lr,
                                         [train_target_list, train_predict_list],
                                         n_iter=epoch,
                                         tag='train')

        savemodel.save_model(net, this_train_tag, 'last_checkpoint.pth', epoch, args.lr, False)
        # =========================================== validation ===========================================
        if not args.val_test:
            test_predict_list, test_target_list = [], []
            net.eval()
            val_loss_total = 0
            for batch_idx, (blobs, targets) in enumerate(val_dataloader):
                cls_targets = utils.label_to_variable(targets, gpu=GPU, volatile=False)


                blobs = utils.img_data_to_variable(blobs, gpu=GPU)

                with torch.no_grad():

                    cls_preds = net(blobs)
                    val_loss_strategy = loss(cls_preds, cls_targets)
                    # TODO: mixup in loss function

                    if isinstance(cls_preds, tuple) or isinstance(cls_preds, list):
                        pred = cls_preds[0].data.max(1)[1]  # get the index of the max log-probability
                    else:
                        pred = cls_preds.data.max(1)[1]  # get the index of the max log-probability

                test_predict_list += list(pred.cpu().numpy())
                test_target_list += list(targets)

                val_loss = val_loss_strategy
                val_loss_total = val_loss_total + val_loss

                utils.progress_bar(batch_idx, len(val_dataloader), 'Loss: %.5f | Acc: %.3f | BestAcc: %.3f'
                                   % (val_loss_total / (batch_idx + 1),
                                      100. * metrics.accuracy_score(test_target_list, test_predict_list), best_val_acc))

            tensorboard_writer.write_scalars(train_loss_total / len(val_dataloader),
                                             args.lr,
                                             [test_target_list, test_predict_list],
                                             n_iter=epoch,
                                             tag='validation')

            if metrics.accuracy_score(test_target_list, test_predict_list) > best_val_acc:
                best_val_acc = metrics.accuracy_score(test_target_list, test_predict_list)
                print('Saving best model, acc:{}'.format(best_val_acc))
                savemodel.save_model(net, this_train_tag, 'best_val_acc.pth',
                                     epoch, args.lr, False, val_acc=best_val_acc)

        #
        #     # save_model regularly
        #     if strategy_dict['regularly_save']:
        #         if step % (cfg.SOLVER.SAVE_INTERVAL * len(train_dataloader)) == 0 and step != start_step:
        #             print('Saving state regularly, iter:', step)
        #             save_model(net, resume_path, '{}_checkpoint.pth'.format(step), step, learning_rate, parallel_tag)
        # return valid_best_acc
        #
        #
        #
        #
    print('best validation acc = {}'.format(best_val_acc))
    # print('Ada\'s components are: cell_Trans={}, cell_Reward={}, cell_Pred={}'.format(args.cell_trans, args.cell_reward,
    #                                                                                   args.cell_pred))
    # default = ht cr cp
    print('Savings are saved at {}'.format(this_train_tag))



def my_dataloader_inference(root, filename, args, train, onecell=False):
    image_tmpl = "{:d}.jpg"

    transform = torchvision.transforms.Compose([GroupScale(int(224 * 256 // 224)),
                                                   GroupCenterCrop(224)])
    dataset = inference_dataset(root, filename, image_tmpl, transform, onecell)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=train, num_workers=0,
                             collate_fn=dataset.classify_collate, pin_memory=True)
    return data_loader


class inference_dataset(data.Dataset):
    def __init__(self, root_path, list_file, image_tmpl='img_{:05d}.jpg', transform=None,oncell=False):
        self.onecell = oncell
        self.img_list = list_file
        self.image_tmpl = image_tmpl
        self.transform = transform

    def __getitem__(self, index):
        img_name_ = self.img_list[index]
        img_t = []
        idx = 0
        for _ in range(400):
            # img_name = img_name_[:-2] if img_name_[-2] == '_' else img_name_[:-3]
            img_name = img_name_.split('_')[0] + '_' + img_name_.split('_')[1] + '_' + img_name_.split('_')[2]
            # TODO:
            directory = os.path.join('/data2/chengyi/embryo_HUSH/first_travel_VideoImage/Images', img_name)

            while not os.path.exists(os.path.join(directory, self.image_tmpl.format(idx))):
                idx += 1
            # try:
            img = [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
            # except FileNotFoundError:
            #     idx += 5
            #     img = [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
            process_data = self.transform(img)

            tmp = np.concatenate(process_data, axis=2)
            img = torch.from_numpy(tmp).permute(2, 0, 1).contiguous()/255.
            img_t.append(img)
            # slided_output = np.concatenate([tmp[:, :, x] for x in range(0, tmp.shape[2] - 1, 3)])
            idx += 1
        return torch.stack(img_t,dim=0), img_name_

    def __len__(self):
        return len(self.img_list)

    def classify_collate(self, batch):


        return batch[0]


def my_dataloader_inference_final(root, filename, args):


    image_tmpl = "{:d}.jpg"

    transform = torchvision.transforms.Compose([GroupScale(int(224 * 256 // 224)),
                                                   GroupCenterCrop(224)])
    # if args.modality == 'LSTM':
    #     dataset = LSTMDataSet(root, filename, args.num_segments, datalength, args.modality, image_tmpl, transform)

    dataset = inference_final_dataset(root_path=root, list_file=filename, image_tmpl=image_tmpl, transform=transform)

    # TODO: change shuffle

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                             collate_fn=dataset.classify_collate, pin_memory=True)

    return data_loader

class inference_final_dataset(data.Dataset):
    def __init__(self, root_path, list_file, image_tmpl='img_{:05d}.jpg', transform=None):
        self.root_path = root_path

        self.list_file = os.path.join(self.root_path, 'data_split', list_file)

        self.image_tmpl = image_tmpl
        self.transform = transform

        self._parse_list()

    def _load_image(self, directory, idx):
        # print(directory)
        return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]

    def _parse_list(self):

        self.info_dict = json.load(open(self.list_file, 'r'))
        self.video_list = [x.strip() for x in self.info_dict]

    def __getitem__(self, index):
        video_name = self.video_list[index]
        record = self.info_dict[video_name]

        segment_indices = list(range(400))

        return self.get(record, segment_indices, video_name)

    def get(self, record, indices, video_name):

        img_t = []
        for p in range(400):
            # img_name = img_name_[:-2] if img_name_[-2] == '_' else img_name_[:-3]
            sub_path = 'Images'
            img = self._load_image(os.path.join(self.root_path, sub_path, video_name),
                                        record['mapping_indexes'][p])
            # try:
            process_data = self.transform(img)

            tmp = np.concatenate(process_data, axis=2)
            img = torch.from_numpy(tmp).permute(2, 0, 1).contiguous()/255.
            img_t.append(img)
            # slided_output = np.concatenate([tmp[:, :, x] for x in range(0, tmp.shape[2] - 1, 3)])

        return torch.stack(img_t,dim=0), video_name




        #
        # images = list()
        # indexes = list()
        # for seg_ind in indices:
        #     p = int(seg_ind)
        #     try:
        #         sub_path = 'Images'
        #         seg_imgs = self._load_image(os.path.join(self.root_path, sub_path, video_name),
        #                                     record['mapping_indexes'][p])
        #     except IndexError:
        #         print(video_name, len(record['mapping_indexes']), p)
        #         print(record['mapping_indexes'][p])
        #         exit()
        #     # try:
        #     process_data = self.transform(seg_imgs)
        #
        #     tmp = np.concatenate(process_data, axis=2)
        #     img = torch.from_numpy(tmp).permute(2, 0, 1).contiguous()/255.
        #     images.append(img)
        #
        #     # images.extend(seg_imgs)
        #     # indexes.append(p)
        #     # except TypeError:
        #     #     print(self.root_path, sub_path, video_name, record['mapping_indexes'][p])
        #
        #     if p < record['num_frames']:
        #         p += 1
        #
        # # process_data = self.transform(images)
        #
        #
        #
        # return images, video_name

    def __len__(self):
        return len(self.video_list)

    def classify_collate(self, batch):

        # imgs = [x[0] for x in batch]
        # return torch.stack(imgs, 0).transpose(2, 1)
        return batch[0]

def inference():
    args = my_parse.parse_args()

    GPU = 1

    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus).split('[')[-1].split(']')[0]
    # to store data and weight in cuda: args.gpus[0] instead of cuda:0

    model_file = '/data2/chengyi/myproject/Savings/save_models/Jan21_04-56-19_Ada_Ada_onecell/best_val_acc.pth'
    ckpt = torch.load(model_file)
    net = ckpt['net']

    set = 'train'




    json_file_out = '/data2/chengyi/embryo_HUSH/first_travel_VideoImage/stem/{}_PNF_onecell.json'.format(set)

    df = pd.read_excel('/data2/chengyi/embryo_HUSH/first_travel_VideoImage/stem/stem-PNF-{}.xlsx'.format(set))
    data = df.values
    files = [file for file, _, _ in data]
    val_dataloader = my_dataloader_inference(root='/data2/chengyi/embryo_HUSH/first_travel_VideoImage/stem',
                                                 filename=files, args=args, train=False, onecell=True)

    if GPU:
        net = net.cuda()

    net.eval()
    one_cell_dir = {}
    for batch_idx, (blobs, img_name) in enumerate(val_dataloader):
        # blobs = blobs[0]
        blobs = utils.img_data_to_variable(blobs, gpu=GPU)

        with torch.no_grad():

            cls_preds = net(blobs)

            # TODO: mixup in loss function

            if isinstance(cls_preds, tuple) or isinstance(cls_preds, list):
                pred = cls_preds[0].data.max(1)[1]  # get the index of the max log-probability
            else:
                pred = cls_preds.data.max(1)[1]  # get the index of the max log-probability
            for i in range(399):
                present = pred[-10-i:-i]
                if present.cpu().sum() == 10:
                    one_cell_dir[img_name] = 400-i
                    break


        utils.progress_bar(batch_idx, len(val_dataloader))
    my_out = json.dumps(one_cell_dir)
    f2 = open(json_file_out, 'w')
    f2.write(my_out)
    f2.close()


def inference_final(set):
    args = my_parse.parse_args()


    GPU = 1

    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus).split('[')[-1].split(']')[0]
    # to store data and weight in cuda: args.gpus[0] instead of cuda:0

    model_file = '/data2/chengyi/myproject/Savings/save_models/Jan21_04-56-19_Ada_Ada_onecell/best_val_acc.pth'
    ckpt = torch.load(model_file)
    net = ckpt['net']

    # set = 'test'

    json_file_out = '/data2/chengyi/embryo_HUSH/first_travel_VideoImage/stem/final/{}_onecell.json'.format(set)
    json_file = '{}.json'.format(set)

    val_dataloader = my_dataloader_inference_final(root='/data2/chengyi/embryo_HUSH/first_travel_VideoImage/',
                                                 filename=json_file, args=args)

    if GPU:
        net = net.cuda()

    net.eval()
    one_cell_dir = {}
    for batch_idx, (blobs, img_name) in enumerate(val_dataloader):
        # blobs = blobs[0]
        blobs = utils.img_data_to_variable(blobs, gpu=GPU)

        with torch.no_grad():

            cls_preds = net(blobs)

            # TODO: mixup in loss function

            if isinstance(cls_preds, tuple) or isinstance(cls_preds, list):
                pred = cls_preds[0].data.max(1)[1]  # get the index of the max log-probability
            else:
                pred = cls_preds.data.max(1)[1]  # get the index of the max log-probability
            for i in range(399):
                present = pred[-10-i:-i]
                if present.cpu().sum() == 10:
                    one_cell_dir[img_name] = 400-i
                    break


        utils.progress_bar(batch_idx, len(val_dataloader))
    my_out = json.dumps(one_cell_dir)
    f2 = open(json_file_out, 'w')
    f2.write(my_out)
    f2.close()


if __name__ == '__main__':
    setname = 'valid'
    print(setname)
    inference_final(set=setname)

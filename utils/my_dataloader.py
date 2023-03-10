import torchvision

from PIL import Image
import os
import os.path
import json
import numpy as np
import torch
import pdb
import torch.utils.data as data
from numpy.random import randint
from torch.utils.data import DataLoader
from utils.transforms import *
import utils.transforms_visual as tv

__all__ = ['my_dataloader']


def my_dataloader(root, filename, args, train):
    if args.modality in ['RGB', 'SPECTRUM', 'CANCELLED', 'GRAY', 'GrayAnother',
                         'VideoGray', 'AdapPool', 'slowfast', 'CSP', 'LSTM', 'Ada']:
        image_tmpl = "{:d}.jpg"
    elif args.modality == 'FlowPlusGray':
        image_tmpl = [args.rgb_prefix + "{:05d}.jpg", args.flow_prefix + "{}_{:05d}.jpg"]
    else:
        image_tmpl = args.flow_prefix + "{}_{:05d}.jpg"

    if args.modality == 'RGB':
        datalength = 1
    elif args.modality == 'Flow':
        datalength = 5
    else:
        datalength = 1
    transform = my_transformer(train=train, modality=args.modality, num_segments=args.num_segments,
                               new_length=datalength, size=224)
    # if args.modality == 'LSTM':
    #     dataset = LSTMDataSet(root, filename, args.num_segments, datalength, args.modality, image_tmpl, transform)
    if args.modality == 'Ada':
        dataset = AdaDataSet_(root_path=root, list_file=filename, num_segments=args.num_segments, new_length=1,
                              image_tmpl=image_tmpl, transform=transform, random_shift=train, positionmode=args.positionmode)

    else:
        dataset = MyDataSet(root, filename, args.num_segments, datalength, args.modality, image_tmpl, transform)
    # TODO: change shuffle

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=train, num_workers=0,
                             collate_fn=dataset.classify_collate, pin_memory=True)

    return data_loader


def my_dataloader_DDP(root, filename, args, train):
    if args.modality in ['RGB', 'SPECTRUM', 'CANCELLED', 'GRAY', 'GrayAnother',
                         'VideoGray', 'AdapPool', 'slowfast', 'CSP', 'LSTM', 'Ada']:
        image_tmpl = "{:d}.jpg"
    elif args.modality == 'FlowPlusGray':
        image_tmpl = [args.rgb_prefix + "{:05d}.jpg", args.flow_prefix + "{}_{:05d}.jpg"]
    else:
        image_tmpl = args.flow_prefix + "{}_{:05d}.jpg"

    if args.modality == 'RGB':
        datalength = 1
    elif args.modality == 'Flow':
        datalength = 5
    else:
        datalength = 1
    transform = my_transformer(train=train, modality=args.modality, num_segments=args.num_segments,
                               new_length=datalength, size=224)
    # if args.modality == 'LSTM':
    #     dataset = LSTMDataSet(root, filename, args.num_segments, datalength, args.modality, image_tmpl, transform)
    if args.modality == 'Ada':
        dataset = AdaDataSet_(root_path=root, list_file=filename, num_segments=args.num_segments, new_length=1,
                              image_tmpl=image_tmpl, transform=transform, random_shift=train)

    else:
        dataset = MyDataSet(root, filename, args.num_segments, datalength, args.modality, image_tmpl, transform)
    # TODO: change shuffle

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=train, num_workers=0,
                             collate_fn=dataset.classify_collate, pin_memory=True, sampler=train_sampler)

    return data_loader


def my_dataloader_resample(root, filename, args, train):
    if args.modality in ['RGB', 'SPECTRUM', 'CANCELLED', 'GRAY', 'GrayAnother',
                         'VideoGray', 'AdapPool', 'slowfast', 'CSP', 'LSTM', 'Ada']:
        image_tmpl = "{:d}.jpg"
    elif args.modality == 'FlowPlusGray':
        image_tmpl = [args.rgb_prefix + "{:05d}.jpg", args.flow_prefix + "{}_{:05d}.jpg"]
    else:
        image_tmpl = args.flow_prefix + "{}_{:05d}.jpg"

    if args.modality == 'RGB':
        datalength = 1
    elif args.modality == 'Flow':
        datalength = 5
    else:
        datalength = 1
    transform = my_transformer(train=train, modality=args.modality, num_segments=args.num_segments,
                               new_length=datalength, size=224)
    # if args.modality == 'LSTM':
    #     dataset = LSTMDataSet(root, filename, args.num_segments, datalength, args.modality, image_tmpl, transform)
    if args.modality == 'Ada':
        dataset = AdaDataSet_resample(root, filename, args.num_segments, 1, image_tmpl, transform)
    else:
        dataset = MyDataSet(root, filename, args.num_segments, datalength, args.modality, image_tmpl, transform)
    # TODO: change shuffle
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                             collate_fn=dataset.classify_collate, pin_memory=True)

    return data_loader


def my_dataloader_visual(root, filename, args, train):
    if args.modality in ['RGB', 'SPECTRUM', 'CANCELLED', 'GRAY', 'GrayAnother',
                         'VideoGray', 'AdapPool', 'slowfast', 'CSP', 'LSTM', 'Ada']:
        image_tmpl = "{:d}.jpg"
    elif args.modality == 'FlowPlusGray':
        image_tmpl = [args.rgb_prefix + "{:05d}.jpg", args.flow_prefix + "{}_{:05d}.jpg"]
    else:
        image_tmpl = args.flow_prefix + "{}_{:05d}.jpg"

    if args.modality == 'RGB':
        datalength = 1
    elif args.modality == 'Flow':
        datalength = 5
    else:
        datalength = 1
    transform = tv.my_transformer(train=train, modality=args.modality,
                                  new_length=datalength, size=224)
    # if args.modality == 'LSTM':
    #     dataset = LSTMDataSet(root, filename, args.num_segments, datalength, args.modality, image_tmpl, transform)
    if args.modality == 'Ada':
        dataset = AdaDataSet_visual(root, filename, args.num_segments, 1, image_tmpl, transform)
    else:
        dataset = MyDataSet(root, filename, args.num_segments, datalength, args.modality, image_tmpl, transform)
    # TODO: change shuffle
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                             collate_fn=dataset.classify_collate, pin_memory=True)

    return data_loader


def my_dataloader_alex(root, filename, args, train):
    image_tmpl = "{:d}.jpg"

    datalength = 1

    transform = my_transformer(train=train, modality=args.modality, num_segments=args.num_segments,
                               new_length=datalength, size=227)
    # if args.modality == 'LSTM':
    #     dataset = LSTMDataSet(root, filename, args.num_segments, datalength, args.modality, image_tmpl, transform)

    dataset = MyDataSet_alex(root, filename, args.num_segments, datalength, args.modality, image_tmpl, transform)
    # TODO: change shuffle

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=train, num_workers=0,
                             collate_fn=dataset.classify_collate, pin_memory=True)

    return data_loader


def my_dataloader_res_comparison(root, filename, args, train, temporal=False, temporal_training=False, emsemble=False):
    image_tmpl = "{:d}.jpg"

    datalength = 1

    transform = my_transformer(train=train, modality=args.modality, num_segments=args.num_segments,
                               new_length=datalength, size=224)

    if emsemble:
        dataset = MyDataSet_res_seg_emsemble(root, filename, args.num_segments, datalength, args.modality,
                                                      image_tmpl,
                                                      transform,
                                                      num_frames_each=7)
    elif temporal_training:
        dataset = MyDataSet_res_seg_temporal_training(root, filename, args.num_segments, datalength, args.modality,
                                                      image_tmpl,
                                                      transform,
                                                      num_frames_each=7)
    elif temporal:
        dataset = MyDataSet_res_seg_temporal(root, filename, args.num_segments, datalength, args.modality, image_tmpl,
                                             transform,
                                             num_frames_each=7)
    else:
        dataset = MyDataSet_res_seg(root, filename, args.num_segments, datalength, args.modality, image_tmpl, transform,
                                    num_frames_each=7)

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=train, num_workers=0,
                             collate_fn=dataset.classify_collate, pin_memory=True)

    return data_loader


def my_dataloader_stem2(root, filename, args, train):
    image_tmpl = "{:d}.jpg"

    datalength = 1

    transform = my_transformer(train=train, modality=args.modality, num_segments=args.num_segments,
                               new_length=datalength, size=224)

    dataset = AdaDataSet_stem2(root, filename, args.num_segments, datalength, args.modality, image_tmpl, transform)

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=train, num_workers=0,
                             collate_fn=dataset.classify_collate, pin_memory=True)

    return data_loader


def my_dataloader_stemcellcount(root, filename, args, train, onecell=False):
    image_tmpl = "{:d}.jpg"

    datalength = 1
    if train:
        transform = torchvision.transforms.Compose([GroupMultiScaleCrop(224, [1, .875, .75, .66]),
                                                    GroupRandomHorizontalFlip(is_flow=False),
                                                    GroupRandomVerticalFlip(),
                                                    GroupRandomBrightness(),
                                                    GroupShiftScaleRotate()])
    else:
        transform = torchvision.transforms.Compose([GroupScale(int(224 * 256 // 224)),
                                                    GroupCenterCrop(224)])

    dataset = AdaDataSet_stemcellcount(root, filename, image_tmpl, transform, onecell)

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=train, num_workers=0,
                             collate_fn=dataset.classify_collate, pin_memory=True)

    return data_loader


class MyDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True,
                 test_mode=False, mixtureimg=None,
                 sample_inverval=1):
        self.mixtureimg = mixtureimg
        self.root_path = root_path
        if isinstance(list_file, list):
            self.list_file = [os.path.join(self.root_path, 'data_split', list_file[0]),
                              os.path.join(self.root_path, 'data_split', list_file[1])]
        else:
            self.list_file = os.path.join(self.root_path, 'data_split', list_file)
        self.sample_interval = sample_inverval
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        # print(directory)
        if self.modality in ['RGB', 'RGBDiff']:
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality in ['SPECTRUM', 'CANCELLED', 'GRAY', 'GrayAnother', 'VideoGray',
                               'AdapPool', 'slowfast', 'CSP']:
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('L')]
        elif self.modality == 'Flow':
            idx = idx + 1
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')

            return [x_img, y_img]
        elif self.modality == 'FlowPlusGray':
            idx = idx + 1
            # image_tmpl = [args.rgb_prefix + "{:d}.jpg", args.flow_prefix + "{}_{:05d}.jpg"]
            img = Image.open(os.path.join(directory, self.image_tmpl[0].format(idx))).convert('L')
            x_img = Image.open(os.path.join(directory, self.image_tmpl[1].format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl[1].format('y', idx))).convert('L')
            return [img, x_img, y_img]

    def _parse_list(self):
        if isinstance(self.list_file, list):
            self.info_dict = json.load(open(self.list_file[0], 'r'))
            self.info_dict.update(json.load(open(self.list_file[1], 'r')))
        else:
            self.info_dict = json.load(open(self.list_file, 'r'))
        self.video_list = [x.strip() for x in self.info_dict]

    def _sample_indices(self, record, num_segments):
        """
        :param record: VideoRecord
        :return: list
        """

        average_duration = (record['num_frames'] - self.new_length * self.sample_interval + 1) // num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(num_segments)), average_duration) + randint(0, average_duration,
                                                                                         size=num_segments)
        elif record['num_frames'] > num_segments:
            offsets = np.sort(
                randint(record['num_frames'] - self.new_length * self.sample_interval + 1, size=num_segments))
        else:
            offsets = np.zeros((num_segments,))
        return offsets

    def _get_val_indices(self, record, num_segments):
        if self.modality in ['VideoGray', 'AdapPool', 'GRAY', 'slowfast']:
            num_frame = record['num_frames'] - 100
        else:
            num_frame = record['num_frames']
        if num_frame > num_segments + self.new_length * self.sample_interval - 1:
            tick = (num_frame - self.new_length * self.sample_interval + 1) / float(num_segments)
            # 相当于只取了每个clip中间位置的1帧，这种方式进行inference是否合理还有待商榷
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)])
        else:
            offsets = np.zeros((num_segments,))
        if self.modality in ['VideoGray', 'AdapPool', 'GRAY', 'slowfast']:
            return offsets + 100
        else:
            return offsets + 1

    def _get_test_indices(self, record):
        if self.modality in ['VideoGray', 'AdapPool', 'GRAY', 'slowfast']:
            num_frame = record['num_frames'] - 100
        else:
            num_frame = record['num_frames']
        tick = (num_frame - self.new_length * self.sample_interval + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        if self.modality in ['VideoGray', 'AdapPool', 'GRAY', 'slowfast']:
            return offsets + 100
        else:
            return offsets + 1

    def __getitem__(self, index):
        video_name = self.video_list[index]
        record = self.info_dict[video_name]

        if not self.test_mode:
            segment_indices = self._sample_indices(record,
                                                   self.num_segments) if self.random_shift else self._get_val_indices(
                record, self.num_segments)
        else:
            segment_indices = self._get_test_indices(record)

        if self.mixtureimg is not None:
            # TODO
            if self.mixtureimg == 'couple':
                N = 4
                segment_indices_couple = [x - N for x in segment_indices]
                segment_indices_mix = [rv for r in zip([x - N // 2 - N for x in segment_indices],
                                                       [x - N // 2 for x in segment_indices],
                                                       [x + N // 2 for x in segment_indices]) for rv in r]
                # 2 4 6 12 14 16 ....
                segment_indices = [rv for r in zip(segment_indices_couple, segment_indices) for rv in r]
                # 3 5 13 15 23 25...
            elif self.mixtureimg == 'single':
                interval = (segment_indices[1] - segment_indices[0]) // 2
                segment_indices_mix = [x - interval for x in segment_indices]
                segment_indices_mix.append(segment_indices[-1] + interval // 2)

            return [self.get(record, segment_indices, video_name), self.get(record, segment_indices_mix, video_name)]
        if self.modality == 'slowfast':
            segment_indices_slow = self._sample_indices(record,
                                                        self.num_segments // 8) if self.random_shift else self._get_val_indices(
                record, self.num_segments // 8)
            return [self.get(record, segment_indices_slow, video_name), self.get(record, segment_indices, video_name)]
        # sequenct: slow(self.seg) --> fast(self.seg//8)

        return self.get(record, segment_indices, video_name)

    def get(self, record, indices, video_name):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                try:
                    if self.modality in ['Flow', 'FlowPlusGray']:
                        sub_path = 'OpticalFlow'
                    else:
                        sub_path = 'Images'
                    seg_imgs = self._load_image(os.path.join(self.root_path, sub_path, video_name),
                                                record['mapping_indexes'][p])
                except IndexError:
                    print(video_name, len(record['mapping_indexes']), p)
                    print(record['mapping_indexes'][p])
                    exit()
                # try:
                images.extend(seg_imgs)
                # except TypeError:
                #     print(self.root_path, sub_path, video_name, record['mapping_indexes'][p])

                if p < record['num_frames']:
                    p += self.sample_interval

        process_data = self.transform(images)
        return process_data, record['label']

    def __len__(self):
        return len(self.video_list)

    '''
    collate_fn of this data-set
    '''

    def classify_collate(self, batch):
        '''
        :param batch: [imgs, labels] dtype = np.ndarray
        imgs:
          shape = (C H W)
        :return:
        '''
        if self.mixtureimg:
            labels = [x[0][1] for x in batch]
            imgs_original = [x[0][0] for x in batch]
            imgs_expand = [x[1][0] for x in batch]
            imgs = []
            if self.mixtureimg == 'single':
                for i in range(len(labels)):
                    each_batch_img = [imgs_original[i][:, j, :, :] * 0.5 \
                                      + imgs_expand[i][:, j, :, :] * 0.25 \
                                      + imgs_expand[i][:, j + 1, :, :] * 0.25 for j in range(self.num_segments)]
                    imgs.append(torch.cat(each_batch_img, dim=0))
            else:
                for i in range(len(labels)):
                    each_batch_img = [imgs_original[i][:, j, :, :] * 0.5 \
                                      + imgs_expand[i][:, j // 2 + j % 2, :, :] * 0.25 \
                                      + imgs_expand[i][:, j // 2 + 1 + j % 2, :, :] * 0.25 for j in
                                      range(self.num_segments * 2)]
                    imgs.append(torch.cat(each_batch_img, dim=0))
            imgs = torch.stack(imgs, 0)
            imgs = imgs.unsqueeze(dim=1)
            return imgs, np.asarray(labels).astype(np.int64)

        if self.modality == 'slowfast':
            labels = [x[0][1] for x in batch]
            imgs_slow = [x[0][0] for x in batch]
            imgs_fast = [x[1][0] for x in batch]
            return [torch.stack(imgs_slow, 0), torch.stack(imgs_fast, 0)], np.asarray(labels).astype(np.int64)

        else:
            if self.modality == 'CSP':
                labels = [1 - x[1] for x in batch]
            else:
                labels = [x[1] for x in batch]

            imgs = [x[0] for x in batch]
            return torch.stack(imgs, 0), np.asarray(labels).astype(np.int64)


class AdaDataSet_(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=128, new_length=1,
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True,
                 test_mode=False,
                 sample_inverval=1, positionmode='850'):
        self.positionmode = positionmode
        self.root_path = root_path
        if isinstance(list_file, list):
            self.list_file = [os.path.join(self.root_path, 'data_split', list_file[0]),
                              os.path.join(self.root_path, 'data_split', list_file[1])]
        else:
            self.list_file = os.path.join(self.root_path, 'data_split', list_file)
        self.sample_interval = sample_inverval
        self.num_segments = num_segments
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.new_length = 1
        self._parse_list()

    def _load_image(self, directory, idx):
        # print(directory)
        return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('L')]

    def _parse_list(self):
        if isinstance(self.list_file, list):
            self.info_dict = json.load(open(self.list_file[0], 'r'))
            self.info_dict.update(json.load(open(self.list_file[1], 'r')))
        else:
            self.info_dict = json.load(open(self.list_file, 'r'))
        self.video_list = [x.strip() for x in self.info_dict]

    def _sample_indices(self, record, num_segments):
        """
        :param record: VideoRecord
        :return: list
        """

        average_duration = (record['num_frames'] - self.new_length * self.sample_interval + 1) // num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(num_segments)), average_duration) + randint(0, average_duration,
                                                                                         size=num_segments)
        elif record['num_frames'] > num_segments:
            offsets = np.sort(
                randint(record['num_frames'] - self.new_length * self.sample_interval + 1, size=num_segments))
        else:
            offsets = np.zeros((num_segments,))
        return offsets

    def _get_val_indices(self, record, num_segments):
        # 采样位置为： 50-750

        num_frame = record['num_frames']
        if num_frame > 750:
            num_frame = 750

        num_frame = num_frame - 50

        if num_frame > num_segments + self.new_length * self.sample_interval - 1:
            tick = (num_frame - self.new_length * self.sample_interval + 1) / float(num_segments)
            # 相当于只取了每个clip中间位置的1帧，这种方式进行inference是否合理还有待商榷
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)])
        else:
            offsets = np.zeros((num_segments,))

        return offsets + 50

    def __getitem__(self, index):
        video_name = self.video_list[index]
        record = self.info_dict[video_name]
        # if index > 200:
        #     pass
        #     print(index)
        segment_indices = self._sample_indices(record, self.num_segments) if \
            self.random_shift else self._get_val_indices(record, self.num_segments)
        # record['mapping_indexes']
        return self.get(record, segment_indices, video_name)

    def get(self, record, indices, video_name):

        images = list()
        indexes = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                try:
                    sub_path = 'Images'
                    seg_imgs = self._load_image(os.path.join(self.root_path, sub_path, video_name),
                                                record['mapping_indexes'][p])
                except IndexError:
                    print(video_name, len(record['mapping_indexes']), p)
                    print(record['mapping_indexes'][p])
                    exit()
                # try:
                images.extend(seg_imgs)
                indexes.append(p)
                # except TypeError:
                #     print(self.root_path, sub_path, video_name, record['mapping_indexes'][p])

                if p < record['num_frames']:
                    p += self.sample_interval

        process_data = self.transform(images)
        positional_encoding = self.positional_encoding(indexes)
        return process_data, positional_encoding, record['label']

    def __len__(self):
        return len(self.video_list)

    def positional_encoding(self, indexes, max_len=2048):
        '''
        :param indexes: dtype = int, max_len: int
        max_len: length of LSTM input
        dim = indexes length

        :return:
        max_len * dim
        '''
        # positionmode
        # embedding = True
        # index850 = True

        if self.positionmode == 'embedding':
            # return indexes*5
            return [i*5 for i in indexes]
        elif self.positionmode == '850':
            # 用于index为实际图像的绝对index：0-850
            dim = 850
            pe = torch.zeros(max_len, dim)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)))
            pe[:, 0::2] = torch.sin(position.float() * div_term)
            pe[:, 1::2] = torch.cos(position.float() * div_term)
            # pe.shape = 2048(LSTM length) * 128(data length)
            pe = pe.transpose(1, 0)
            pe = torch.stack([pe[x] for x in indexes], dim=0)
            return pe
            # print(pefinal.shape)
        elif self.positionmode == '32':
            # 用于index为实际图像的相对index：0-31
            dim = len(indexes)
            pe = torch.zeros(max_len, dim)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)))
            pe[:, 0::2] = torch.sin(position.float() * div_term)
            pe[:, 1::2] = torch.cos(position.float() * div_term)
            # pe.shape = 2048(LSTM length) * 128(data length)
            pe = pe.transpose(1, 0)
            return pe
        else:
            return None
            # print('WRONG POSITION MODE')
            # raise ValueError


    def classify_collate(self, batch):
        '''
        :param batch: [imgs, indexes, labels] dtype = np.ndarray
        imgs:
          shape = (C H W)
        :return:
        [img, position], label
        '''

        labels = [x[2] for x in batch]
        if isinstance(batch[0][1], torch.Tensor):
            positional_encoding = torch.stack([x[1] for x in batch])
        elif isinstance(batch[0][1], list):
            positional_encoding = [x[1] for x in batch]
        else:
            positional_encoding = None
        imgs = [x[0] for x in batch]
        # return [torch.stack(imgs, 0).transpose(2, 1), torch.stack(positional_encoding)], np.asarray(labels).astype(
        #     np.int64)
        return [torch.stack(imgs, 0).transpose(2, 1), positional_encoding], np.asarray(labels).astype(
            np.int64) # 2022-2-15


class AdaDataSet_visual(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=128, new_length=1,
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True,
                 test_mode=False,
                 sample_inverval=1):
        self.root_path = root_path
        if isinstance(list_file, list):
            self.list_file = [os.path.join(self.root_path, 'data_split', list_file[0]),
                              os.path.join(self.root_path, 'data_split', list_file[1])]
        else:
            self.list_file = os.path.join(self.root_path, 'data_split', list_file)
        self.sample_interval = sample_inverval
        self.num_segments = num_segments
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.new_length = 1
        self._parse_list()

    def _load_image(self, directory, idx):
        # print(directory)
        return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('L')]

    def _parse_list(self):
        if isinstance(self.list_file, list):
            self.info_dict = json.load(open(self.list_file[0], 'r'))
            self.info_dict.update(json.load(open(self.list_file[1], 'r')))
        else:
            self.info_dict = json.load(open(self.list_file, 'r'))
        self.video_list = [x.strip() for x in self.info_dict]

    def _sample_indices(self, record, num_segments):
        """
        :param record: VideoRecord
        :return: list
        """

        average_duration = (record['num_frames'] - self.new_length * self.sample_interval + 1) // num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(num_segments)), average_duration) + randint(0, average_duration,
                                                                                         size=num_segments)
        elif record['num_frames'] > num_segments:
            offsets = np.sort(
                randint(record['num_frames'] - self.new_length * self.sample_interval + 1, size=num_segments))
        else:
            offsets = np.zeros((num_segments,))
        return offsets

    def _get_val_indices(self, record, num_segments):
        # 采样位置为： 50-750

        num_frame = record['num_frames']

        if num_frame > num_segments + self.new_length * self.sample_interval - 1:
            tick = (num_frame - self.new_length * self.sample_interval + 1) / float(num_segments)
            # 相当于只取了每个clip中间位置的1帧，这种方式进行inference是否合理还有待商榷
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)])
        else:
            offsets = np.zeros((num_segments,))

        return offsets

    def __getitem__(self, index):
        video_name = self.video_list[index]
        record = self.info_dict[video_name]

        segment_indices = self._sample_indices(record, self.num_segments) if \
            self.random_shift else self._get_val_indices(record, self.num_segments)

        return self.get(record, segment_indices, video_name)

    def get(self, record, indices, video_name):

        images = list()
        indexes = list()
        # video_name_ = []
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                try:
                    sub_path = 'Images'
                    seg_imgs = self._load_image(os.path.join(self.root_path, sub_path, video_name),
                                                record['mapping_indexes'][p])
                except IndexError:
                    print(video_name, len(record['mapping_indexes']), p)
                    print(record['mapping_indexes'][p])
                    exit()
                # try:
                images.extend(seg_imgs)
                indexes.append(p)
                # except TypeError:
                #     print(self.root_path, sub_path, video_name, record['mapping_indexes'][p])

                if p < record['num_frames']:
                    p += self.sample_interval

        process_data = self.transform(images)
        positional_encoding = self.positional_encoding(indexes)
        return process_data, positional_encoding, record['label'], video_name, indices

    def __len__(self):
        return len(self.video_list)

    def positional_encoding(self, indexes, max_len=2048):
        '''
        :param indexes: dtype = int, max_len: int
        max_len: length of LSTM input
        dim = indexes length

        :return:
        max_len * dim
        '''
        dim = len(indexes)
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        # pe.shape = 2048(LSTM length) * 128(data length)
        pe = pe.transpose(1, 0)
        return pe

    def classify_collate(self, batch):
        '''
        :param batch: [imgs, indexes, labels] dtype = np.ndarray
        imgs:
          shape = (C H W)
        :return:
        [img, position], label
        '''

        labels = [x[2] for x in batch]
        positional_encoding = [x[1] for x in batch]
        imgs = [x[0] for x in batch]
        videoname = [x[3] for x in batch]
        indices = [x[4] for x in batch]
        return [torch.stack(imgs, 0).transpose(2, 1), torch.stack(positional_encoding)], np.asarray(labels).astype(
            np.int64), videoname, indices


class AdaDataSet_resample(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=128, new_length=1,
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True,
                 test_mode=False,
                 sample_inverval=1):
        self.root_path = root_path
        if isinstance(list_file, list):
            self.list_file = [os.path.join(self.root_path, 'data_split_resample', list_file[0]),
                              os.path.join(self.root_path, 'data_split_resample', list_file[1])]
        else:
            self.list_file = os.path.join(self.root_path, 'data_split_resample', list_file)
        self.sample_interval = sample_inverval
        self.num_segments = num_segments
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.new_length = 1
        self._parse_list()

    def _load_image(self, directory, idx):
        # print(directory)
        return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('L')]

    def _parse_list(self):
        if isinstance(self.list_file, list):
            self.info_dict = json.load(open(self.list_file[0], 'r'))
            self.info_dict.update(json.load(open(self.list_file[1], 'r')))
        else:
            self.info_dict = json.load(open(self.list_file, 'r'))
        self.video_list = [x.strip() for x in self.info_dict]

    def _sample_indices(self, record, num_segments):
        """
        :param record: VideoRecord
        :return: list
        """

        average_duration = (record['num_frames'] - self.new_length * self.sample_interval + 1) // num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(num_segments)), average_duration) + randint(0, average_duration,
                                                                                         size=num_segments)
        elif record['num_frames'] > num_segments:
            offsets = np.sort(
                randint(record['num_frames'] - self.new_length * self.sample_interval + 1, size=num_segments))
        else:
            offsets = np.zeros((num_segments,))
        return offsets

    def _get_val_indices(self, record, num_segments):
        # 采样位置为： 50-750

        num_frame = record['num_frames']
        if num_frame > 750:
            num_frame = 750

        num_frame = num_frame - 50

        if num_frame > num_segments + self.new_length * self.sample_interval - 1:
            tick = (num_frame - self.new_length * self.sample_interval + 1) / float(num_segments)
            # 相当于只取了每个clip中间位置的1帧，这种方式进行inference是否合理还有待商榷
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)])
        else:
            offsets = np.zeros((num_segments,))

        return offsets + 50

    def __getitem__(self, index):
        video_name = self.video_list[index]
        record = self.info_dict[video_name]

        segment_indices = self._sample_indices(record, self.num_segments) if \
            self.random_shift else self._get_val_indices(record, self.num_segments)

        return self.get(record, segment_indices, video_name)

    def get(self, record, indices, video_name):

        images = list()
        indexes = list()
        # for seg_ind in indices:
        for seg_ind in range(len(indices)):
            p = int(seg_ind)
            for i in range(self.new_length):
                try:
                    sub_path = 'Images'
                    seg_imgs = self._load_image(os.path.join(self.root_path, sub_path, video_name),
                                                record['mapping_indexes'][p])
                except IndexError:
                    print(video_name, len(record['mapping_indexes']), p)
                    print(record['mapping_indexes'][p])
                    exit()
                # try:
                images.extend(seg_imgs)
                indexes.append(p)
                # except TypeError:
                #     print(self.root_path, sub_path, video_name, record['mapping_indexes'][p])

                if p < record['num_frames']:
                    p += self.sample_interval

        process_data = self.transform(images)
        positional_encoding = self.positional_encoding(indexes)
        return process_data, positional_encoding, record['label']

    def __len__(self):
        return len(self.video_list)

    def positional_encoding(self, indexes, max_len=2048):
        '''
        :param indexes: dtype = int, max_len: int
        max_len: length of LSTM input
        dim = indexes length

        :return:
        max_len * dim
        '''
        dim = len(indexes)
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        # pe.shape = 2048(LSTM length) * 128(data length)
        pe = pe.transpose(1, 0)
        return pe

    def classify_collate(self, batch):
        '''
        :param batch: [imgs, indexes, labels] dtype = np.ndarray
        imgs:
          shape = (C H W)
        :return:
        [img, position], label
        '''

        labels = [x[2] for x in batch]
        positional_encoding = [x[1] for x in batch]
        imgs = [x[0] for x in batch]
        return [torch.stack(imgs, 0).transpose(2, 1), torch.stack(positional_encoding)], np.asarray(labels).astype(
            np.int64)


class MyDataSet_alex(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=1, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True,
                 test_mode=False, mixtureimg=None,
                 sample_inverval=1):
        self.mixtureimg = mixtureimg
        self.root_path = root_path
        if isinstance(list_file, list):
            self.list_file = [os.path.join(self.root_path, 'data_split', list_file[0]),
                              os.path.join(self.root_path, 'data_split', list_file[1])]
        else:
            self.list_file = os.path.join(self.root_path, 'data_split', list_file)
        self.sample_interval = sample_inverval
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        # print(directory)

        return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]

    def _parse_list(self):
        if isinstance(self.list_file, list):
            self.info_dict = json.load(open(self.list_file[0], 'r'))
            self.info_dict.update(json.load(open(self.list_file[1], 'r')))
        else:
            self.info_dict = json.load(open(self.list_file, 'r'))
        self.video_list = [x.strip() for x in self.info_dict]

    def _get_indices(self, record):

        offsets = 570
        return offsets

    def __getitem__(self, index):
        video_name = self.video_list[index]
        record = self.info_dict[video_name]

        segment_indices = self._get_indices(record)

        return self.get(record, segment_indices, video_name)

    def get(self, record, indices, video_name):

        images = list()
        p = indices
        try:
            sub_path = 'Images'
            seg_imgs = self._load_image(os.path.join(self.root_path, sub_path, video_name),
                                        record['mapping_indexes'][p])
        except IndexError:
            print(video_name, len(record['mapping_indexes']), p)
            print(record['mapping_indexes'][p])
            exit()
        images.extend(seg_imgs)

        process_data = self.transform(images)
        return process_data, record['label']

    def __len__(self):
        return len(self.video_list)

    '''
    collate_fn of this data-set
    '''

    def classify_collate(self, batch):
        '''
        :param batch: [imgs, labels] dtype = np.ndarray
        imgs:
          shape = (C H W)
        :return:
        '''

        labels = [x[1] for x in batch]

        imgs = [x[0].squeeze(dim=0) for x in batch]

        return torch.stack(imgs, 0), np.asarray(labels).astype(np.int64)


class MyDataSet_res_seg(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True,
                 mixtureimg=None, num_frames_each=7,
                 sample_inverval=1):
        self.mixtureimg = mixtureimg
        self.root_path = root_path
        if isinstance(list_file, list):
            self.list_file = [os.path.join(self.root_path, list_file[0]),
                              os.path.join(self.root_path, list_file[1])]
        else:
            self.list_file = os.path.join(self.root_path, 'stem/final/', list_file)
        self.sample_interval = sample_inverval
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift

        self.num_frames_each = num_frames_each

        self._parse_list()

    def _load_image(self, directory, idx):
        # print(directory)
        return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('L')]

    def _parse_list(self):
        if isinstance(self.list_file, list):
            self.info_dict = json.load(open(self.list_file[0], 'r'))
            self.info_dict.update(json.load(open(self.list_file[1], 'r')))
        else:
            self.info_dict = json.load(open(self.list_file, 'r'))
        self.video_list = [x.strip() for x in self.info_dict]

    #
    # def _sample_indices(self, record, num_segments, num_frames_each):
    #     """
    #     :param record: VideoRecord
    #     :return: list
    #     """
    #     num_segments = num_segments // num_frames_each
    #     num_frame = record['num_frames'] - 100
    #     average_duration = (num_frame - self.new_length * self.sample_interval + 1) // num_segments
    #     if average_duration > 0:
    #         offsets = np.multiply(list(range(num_segments)), average_duration) + randint(0, average_duration,
    #                                                                                      size=num_segments)
    #     elif num_frame > num_segments:
    #         offsets = np.sort(
    #             randint(num_frame - self.new_length * self.sample_interval + 1, size=num_segments))
    #     else:
    #         offsets = np.zeros((num_segments,))
    #     offsets = [np.arange(each-num_frames_each//2,each+num_frames_each-num_frames_each//2,1) for each in offsets]
    #     offsets = [x + 50 for item in offsets for x in item]
    #     return offsets
    #
    # def _get_val_indices(self, record, num_segments,num_frames_each):
    #     num_segments = num_segments // num_frames_each
    #     num_frame = record['num_frames'] - 100
    #     if num_frame > num_segments + self.new_length * self.sample_interval - 1:
    #         tick = (num_frame - self.new_length * self.sample_interval + 1) / float(num_segments)
    #         # 相当于只取了每个clip中间位置的1帧，这种方式进行inference是否合理还有待商榷
    #         offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)])
    #     else:
    #         offsets = np.zeros((num_segments,))
    #
    #     offsets = [np.arange(each-num_frames_each//2,each+num_frames_each-num_frames_each//2,1) for each in offsets]
    #     offsets = [x + 50 for item in offsets for x in item]
    #     # return offsets
    #
    #     return offsets
    #

    def __getitem__(self, index):
        video_name = self.video_list[index]
        record = self.info_dict[video_name]

        segment_indices = self.info_dict[video_name]['PNF_spatial']
        # segment_indices = self.info_dict[video_name]['PNF_temporal']
        # segment_indices = self._sample_indices(record, self.num_segments, num_frames_each=self.num_frames_each) if self.random_shift else \
        #     self._get_val_indices(record, self.num_segments,  num_frames_each=self.num_frames_each)

        return self.get(record, segment_indices, video_name)

    def get(self, record, indices, video_name):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                try:

                    sub_path = 'Images'
                    seg_imgs = self._load_image(os.path.join(self.root_path, sub_path, video_name),
                                                seg_ind)
                except IndexError:
                    print(video_name, len(record['mapping_indexes']), p)
                    print(record['mapping_indexes'][p])
                    exit()
                # try:
                images.extend(seg_imgs)
                # except TypeError:
                #     print(self.root_path, sub_path, video_name, record['mapping_indexes'][p])

                if p < record['num_frames']:
                    p += self.sample_interval

        process_data = self.transform(images)
        return process_data, record['label']

    def __len__(self):
        return len(self.video_list)

    '''
    collate_fn of this data-set
    '''

    def classify_collate(self, batch):
        '''
        :param batch: [imgs, labels] dtype = np.ndarray
        imgs:
          shape = (C H W)
        :return:
        '''

        labels = [x[1] for x in batch]

        # imgs = [x[0].view(-1, self.num_frames_each, 224, 224) for x in batch]
        imgs = [x[0].view(self.num_frames_each, -1, 224, 224) for x in batch]
        return torch.stack(imgs, 0), np.asarray(labels).astype(np.int64)


class MyDataSet_res_seg_temporal(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True,
                 mixtureimg=None, num_frames_each=7,
                 sample_inverval=1):
        self.mixtureimg = mixtureimg
        self.root_path = root_path
        if isinstance(list_file, list):
            self.list_file = [os.path.join(self.root_path, list_file[0]),
                              os.path.join(self.root_path, list_file[1])]
        else:
            self.list_file = os.path.join(self.root_path, 'stem/final/', list_file)
        self.sample_interval = sample_inverval
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift

        self.num_frames_each = num_frames_each

        self._parse_list()

    def _load_image(self, directory, idx):
        # print(directory)
        return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('L')]

    def _parse_list(self):
        if isinstance(self.list_file, list):
            self.info_dict = json.load(open(self.list_file[0], 'r'))
            self.info_dict.update(json.load(open(self.list_file[1], 'r')))
        else:
            self.info_dict = json.load(open(self.list_file, 'r'))


        self.video_list = [x.strip() for x in self.info_dict]
        self.video_list.remove('20170605_142_E1')
        self.video_list.remove('20170322_468_E10')

    def __getitem__(self, index):
        video_name = self.video_list[index]
        record = self.info_dict[video_name]

        segment_indices = self.info_dict[video_name]['PNF_temporal']

        return self.get(record, segment_indices, video_name)

    def get(self, record, indices, video_name):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                try:

                    sub_path = 'Images'
                    seg_imgs = self._load_image(os.path.join(self.root_path, sub_path, video_name),
                                                seg_ind)
                except IndexError:
                    print(video_name, len(record['mapping_indexes']), p)
                    print(record['mapping_indexes'][p])
                    exit()
                # try:
                images.extend(seg_imgs)
                # except TypeError:
                #     print(self.root_path, sub_path, video_name, record['mapping_indexes'][p])
        try:
            process_data = self.transform(images)
        except:
            print('===============================')
            print(video_name)
        return process_data, record['label'], video_name, indices

    def __len__(self):
        return len(self.video_list)

    '''
    collate_fn of this data-set
    '''

    def classify_collate(self, batch):
        '''
        :param batch: [imgs, labels] dtype = np.ndarray
        imgs:
          shape = (C H W)
        :return:
        '''

        labels = [x[1] for x in batch]

        # imgs = [x[0].view(-1, self.num_frames_each, 224, 224) for x in batch]
        # imgs = [x[0].view(self.num_frames_each, -1 , 224, 224) for x in batch]
        imgs = [x[0] for x in batch]
        video = [x[2] for x in batch]
        feature = torch.stack(imgs, 0)
        indices = [x[3] for x in batch]

        return feature, np.asarray(labels).astype(np.int64), video, indices


class MyDataSet_res_seg_temporal_training(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True,
                 mixtureimg=None, num_frames_each=7,
                 sample_inverval=1):
        self.root_path = root_path

        self.list_file_feature = os.path.join(self.root_path, 'stem/final',list_file)



        self._parse_list()

    def _parse_list(self):
        self.feature_dict = json.load(open(self.list_file_feature, 'r'))

        self.video_list = [x.strip() for x in self.feature_dict]

    def __getitem__(self, index):
        video_name = self.video_list[index]
        cell_count = self.feature_dict[video_name]['cell_count']
        label = self.feature_dict[video_name]['targets'][0]
        cell_count = torch.tensor(cell_count)

        indice = self.feature_dict[video_name]['imgs'][0]
        indice = torch.tensor(indice)

        return cell_count, label, indice

    def __len__(self):
        return len(self.video_list)

    '''
    collate_fn of this data-set
    '''

    def classify_collate(self, batch):
        '''
        :param batch: [imgs, labels] dtype = np.ndarray
        imgs:
          shape = (C H W)
        :return:
        '''

        labels = [x[1] for x in batch]
        indices = [x[2] for x in batch]
        cell_count = [x[0] for x in batch]

        return torch.stack(cell_count, dim=0), np.asarray(labels).astype(np.int64), torch.stack(indices, dim=0)


class MyDataSet_res_seg_emsemble(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True,
                 mixtureimg=None, num_frames_each=7,
                 sample_inverval=1):
        self.root_path = root_path
        self.new_length = new_length
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.num_frames_each = num_frames_each
        self.sample_interval = sample_inverval
        self.list_file_feature = os.path.join(self.root_path, 'stem/final',list_file[1])
        self.list_file_info = os.path.join(self.root_path, 'stem/final', list_file[0])


        self._parse_list()

    def _parse_list(self):
        self.feature_dict = json.load(open(self.list_file_feature, 'r'))
        self.info_dict = json.load(open(self.list_file_info, 'r'))
        self.video_list = [x.strip() for x in self.feature_dict]

    def __getitem__(self, index):
        video_name = self.video_list[index]
        cell_count = self.feature_dict[video_name]['cell_count']
        label = self.feature_dict[video_name]['targets'][0]
        cell_count = torch.tensor(cell_count)

        indice = self.feature_dict[video_name]['imgs'][0]
        indice = torch.tensor(indice)

        record = self.info_dict[video_name]

        segment_indices = self.info_dict[video_name]['PNF_spatial']
        # segment_indices = self.info_dict[video_name]['PNF_temporal']
        # segment_indices = self._sample_indices(record, self.num_segments, num_frames_each=self.num_frames_each) if self.random_shift else \
        #     self._get_val_indices(record, self.num_segments,  num_frames_each=self.num_frames_each)
        return cell_count, label, indice, self.get(record, segment_indices, video_name)

    def _load_image(self, directory, idx):
        # print(directory)
        return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('L')]


    def get(self, record, indices, video_name):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                try:

                    sub_path = 'Images'
                    seg_imgs = self._load_image(os.path.join(self.root_path, sub_path, video_name),
                                                seg_ind)
                except IndexError:
                    print(video_name, len(record['mapping_indexes']), p)
                    print(record['mapping_indexes'][p])
                    exit()
                # try:
                images.extend(seg_imgs)
                # except TypeError:
                #     print(self.root_path, sub_path, video_name, record['mapping_indexes'][p])

                if p < record['num_frames']:
                    p += self.sample_interval

        process_data = self.transform(images)
        return process_data

    def __len__(self):
        return len(self.video_list)

    '''
    collate_fn of this data-set
    '''

    def classify_collate(self, batch):
        '''
        :param batch: [imgs, labels] dtype = np.ndarray
        imgs:
          shape = (C H W)
        :return:
        '''

        labels = [x[1] for x in batch]
        indices = [x[2] for x in batch]
        cell_count = [x[0] for x in batch]
        imgs = [x[3] for x in batch]

        feature = torch.stack(imgs, 0)

        return torch.stack(cell_count, dim=0), np.asarray(labels).astype(np.int64), torch.stack(indices, dim=0), feature


class AdaDataSet_stem2(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=128, new_length=1,
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True,
                 test_mode=False,
                 sample_inverval=1):
        self.root_path = root_path
        if isinstance(list_file, list):
            self.list_file = [os.path.join(self.root_path, 'data_split_resample', list_file[0]),
                              os.path.join(self.root_path, 'data_split_resample', list_file[1])]
        else:
            self.list_file = os.path.join(self.root_path, 'data_split_resample', list_file)
        self.sample_interval = sample_inverval
        self.num_segments = num_segments
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.new_length = 1
        self._parse_list()

    def _load_image(self, directory, idx):
        # print(directory)
        return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('L')]

    def _parse_list(self):
        if isinstance(self.list_file, list):
            self.info_dict = json.load(open(self.list_file[0], 'r'))
            self.info_dict.update(json.load(open(self.list_file[1], 'r')))
        else:
            self.info_dict = json.load(open(self.list_file, 'r'))
        self.video_list = [x.strip() for x in self.info_dict]

    def _sample_indices(self, record, num_segments):
        """
        :param record: VideoRecord
        :return: list
        """
        num_frame = 600

        average_duration = (num_frame - self.new_length * self.sample_interval + 1) // num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(num_segments)), average_duration) + randint(0, average_duration,
                                                                                         size=num_segments)
        elif num_frame > num_segments:
            offsets = np.sort(
                randint(num_frame - self.new_length * self.sample_interval + 1, size=num_segments))
        else:
            offsets = np.zeros((num_segments,))
        return offsets + 50

    def _get_val_indices(self, record, num_segments):
        # 采样位置为： 50-750

        # num_frame = record['num_frames']
        # if num_frame > 750:
        num_frame = 600

        # num_frame = num_frame - 50

        if num_frame > num_segments + self.new_length * self.sample_interval - 1:
            tick = (num_frame - self.new_length * self.sample_interval + 1) / float(num_segments)
            # 相当于只取了每个clip中间位置的1帧，这种方式进行inference是否合理还有待商榷
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)])
        else:
            offsets = np.zeros((num_segments,))

        return offsets + 50

    def __getitem__(self, index):
        video_name = self.video_list[index]
        record = self.info_dict[video_name]

        segment_indices = self._sample_indices(record, self.num_segments) if \
            self.random_shift else self._get_val_indices(record, self.num_segments)

        return self.get(record, segment_indices, video_name)

    def get(self, record, indices, video_name):

        images = list()
        indexes = list()
        # for seg_ind in indices:
        for seg_ind in range(len(indices)):
            p = int(seg_ind)
            for i in range(self.new_length):
                try:
                    sub_path = 'Images'
                    seg_imgs = self._load_image(os.path.join(self.root_path, sub_path, video_name),
                                                record['mapping_indexes'][p])
                except IndexError:
                    print(video_name, len(record['mapping_indexes']), p)
                    print(record['mapping_indexes'][p])
                    exit()
                # try:
                images.extend(seg_imgs)
                indexes.append(p)
                # except TypeError:
                #     print(self.root_path, sub_path, video_name, record['mapping_indexes'][p])

                if p < record['num_frames']:
                    p += self.sample_interval

        process_data = self.transform(images)
        positional_encoding = self.positional_encoding(indexes)
        return process_data, positional_encoding, record['label']

    def __len__(self):
        return len(self.video_list)

    def positional_encoding(self, indexes, max_len=2048):
        '''
        :param indexes: dtype = int, max_len: int
        max_len: length of LSTM input
        dim = indexes length

        :return:
        max_len * dim
        '''
        dim = len(indexes)
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        # pe.shape = 2048(LSTM length) * 128(data length)
        pe = pe.transpose(1, 0)
        return pe

    def classify_collate(self, batch):
        '''
        :param batch: [imgs, indexes, labels] dtype = np.ndarray
        imgs:
          shape = (C H W)
        :return:
        [img, position], label
        '''

        labels = [x[2] for x in batch]
        positional_encoding = [x[1] for x in batch]
        imgs = [x[0] for x in batch]
        return [torch.stack(imgs, 0).transpose(2, 1), torch.stack(positional_encoding)], np.asarray(labels).astype(
            np.int64)


class AdaDataSet_stemcellcount(data.Dataset):
    def __init__(self, root_path, list_file, image_tmpl='img_{:05d}.jpg', transform=None, oncell=False):
        self.root_path = root_path
        self.onecell = oncell
        self.list_file = os.path.join(self.root_path, list_file)
        self.image_tmpl = image_tmpl
        self.transform = transform
        self._parse_list()

    def _load_image(self, directory, idx):
        # print(directory)
        return

    def _parse_list(self):
        self.info_dict = json.load(open(self.list_file, 'r'))
        self.img_list = [x.strip() for x in self.info_dict]

    def __getitem__(self, index):
        img_name_ = self.img_list[index]
        # if self.onecell:
        #     label = int(self.info_dict[img_name_][1]=='1')  # 't2'-->0
        # else:
        label = int(self.info_dict[img_name_][1]) - 1  # 't2'-->0
        # label = int(self.info_dict[img_name_][0])
        idx = int(img_name_.split('_')[-1])
        # img_name = img_name_[:-2] if img_name_[-2] == '_' else img_name_[:-3]
        img_name = img_name_.split('_')[0] + '_' + img_name_.split('_')[1] + '_' + img_name_.split('_')[2]
        # TODO:
        directory = os.path.join('/data2/chengyi/embryo_HUSH/first_travel_VideoImage/Images', img_name)
        img = [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]

        process_data = self.transform(img)

        tmp = np.concatenate(process_data, axis=2)
        img = torch.from_numpy(tmp).permute(2, 0, 1).contiguous() / 255.
        # slided_output = np.concatenate([tmp[:, :, x] for x in range(0, tmp.shape[2] - 1, 3)])

        return img, label

    def __len__(self):
        return len(self.img_list)

    def classify_collate(self, batch):
        '''
        :param batch: [imgs, indexes, labels] dtype = np.ndarray
        imgs:
          shape = (C H W)
        :return:
        [img, position], label
        '''

        labels = [x[1] for x in batch]

        imgs = [x[0] for x in batch]
        return [torch.stack(imgs, 0), np.asarray(labels).astype(np.int64)]

import torchvision
import random
import cv2
import numbers
import math
import torch
import PIL
import numpy as np
from PIL import Image, ImageOps
from utils.Diff import Diff
from torch.autograd import Variable

def get_preprocess(modality):
    if modality == 'SPECTRUM':
        return torchvision.transforms.Compose([ToSpecturm(),
                                               Stack(roll=False)])
    elif modality == 'CANCELLED':
        return torchvision.transforms.Compose([ToCancelled(D0=5),
                                               Stack(roll=False)])

    elif modality in ['GRAY', 'VideoGray', 'AdapPool', 'CSP','LSTM']:
        return torchvision.transforms.Compose([ToGray(),
                                               Stack(roll=False)])

    else:
        return Stack(roll=False)


def get_augmentation(modality, input_size):
    if modality in ['RGB', 'SPECTRUM', 'CANCELLED', 'GRAY', 'GrayAnother', 'VideoGray', 'AdapPool',
                    'slowfast', 'CSP','LSTM','Ada']:
        return torchvision.transforms.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
                                               GroupRandomHorizontalFlip(is_flow=False),
                                               GroupRandomVerticalFlip(),
                                               GroupRandomBrightness(),
                                               GroupShiftScaleRotate()])

    elif modality == 'FlowPlusGray':
        return torchvision.transforms.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
                                               GroupRandomHorizontalFlip(is_flow=True, contain_plus=True),
                                               GroupRandomVerticalFlip(contain_plus=True),
                                               GroupRandomBrightness(contain_plus=True),
                                               GroupShiftScaleRotate(contain_plus=True)])

    elif modality == 'Flow':
        return torchvision.transforms.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75]),
                                               GroupRandomHorizontalFlip(is_flow=True)])
    elif modality == 'RGBDiff':
        return torchvision.transforms.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75]),
                                               GroupRandomHorizontalFlip(is_flow=False)])


def my_transformer(train, modality, num_segments, new_length, size=224):
    train_augmentation = get_augmentation(modality, size)
    pre_process = get_preprocess(modality)

    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]

    if modality in ['RGB', 'SPECTRUM', 'CANCELLED', 'GRAY', 'FlowPlusGray',
                         'GrayAnother', 'VideoGray', 'AdapPool', 'slowfast', 'CSP','LSTM', 'Ada']:
        pass
    elif modality == 'Flow':
        input_mean = [0.5]
        input_std = [np.mean(input_std)]
    elif modality == 'RGBDiff':
        input_mean = [0.485, 0.456, 0.406] + [0] * 3 * new_length
        input_std = input_std + [np.mean(input_std) * 2] * 3 * new_length
    else:
        raise ValueError

    if modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if train:
        augmentation = train_augmentation
    else:
        augmentation = torchvision.transforms.Compose([GroupScale(int(size * 256 // 224)),
                                   GroupCenterCrop(size)])

    transformer = torchvision.transforms.Compose([
        # NormalTransform(),
        augmentation,
        pre_process,
        ToTorchFormatTensor(num_segments, size, div=True,
                            modality=modality, new_length=new_length),
        normalize,
    ])
    return transformer


class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):

        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert (img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class RandomBrightnessShift(object):
    def __init__(self, limit=(-0.05, 0.05)):
        super(RandomBrightnessShift, self).__init__()
        self.low = limit[0]
        self.high = limit[1]

    def __call__(self, img_group):
        shift = np.random.uniform(self.low, self.high)
        img_group = [np.array(img).astype(np.float32) for img in img_group]
        return [Image.fromarray(np.clip(img + shift, 0, 255).astype(np.uint8)) for img in img_group]


class RandomBrightnessMultiply(object):
    def __init__(self, limit=(-0.05, 0.05)):
        super(RandomBrightnessMultiply, self).__init__()
        self.low = limit[0]
        self.high = limit[1]

    def __call__(self, img_group):
        rate = np.random.uniform(1 + self.low, 1 + self.high)
        img_group = [np.array(img).astype(np.float32) for img in img_group]
        return [Image.fromarray(np.clip(img * rate, 0, 255).astype(np.uint8)) for img in img_group]


class RandomBrightnessGamma(object):
    def __init__(self, limit=(-0.05, 0.05)):
        super(RandomBrightnessGamma, self).__init__()
        self.low = limit[0]
        self.high = limit[1]

    def __call__(self, img_group):
        gamma = np.random.uniform(1 + self.low, 1 + self.high)
        img_group = [np.array(img).astype(np.float32) for img in img_group]
        return [Image.fromarray(np.clip(np.power(img, gamma), 0, 255).astype(np.uint8)) for img in img_group]


class GroupRandomBrightness(object):
    """Only for RGB modality
    """

    def __init__(self, limit=(-0.05, 0.05), contain_plus=False):
        super(GroupRandomBrightness, self).__init__()
        self.limit = limit
        self.contain = contain_plus

    def __call__(self, img_group):
        v = random.random()
        # v = 0.4
        if v < 0.5:
            if self.contain:
                choice = np.random.choice([0, 1, 2])
                should_be_trans_group = [img_group[i] for i in range(0, len(img_group), 3)]
                if choice == 0:
                    should_be_trans_group = RandomBrightnessGamma(self.limit)(should_be_trans_group)
                elif choice == 1:
                    should_be_trans_group = RandomBrightnessMultiply(self.limit)(should_be_trans_group)
                else:
                    should_be_trans_group = RandomBrightnessShift(self.limit)(should_be_trans_group)
                for i in range(0, len(img_group), 3):
                    img_group[i] = should_be_trans_group[int(i / 3)]
                return img_group

            else:
                choice = np.random.choice([0, 1, 2])
                if choice == 0:
                    return RandomBrightnessGamma(self.limit)(img_group)
                elif choice == 1:
                    return RandomBrightnessMultiply(self.limit)(img_group)
                else:
                    return RandomBrightnessShift(self.limit)(img_group)
        else:
            return img_group


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __init__(self, is_flow=False, contain_plus=False):
        self.is_flow = is_flow
        self.contain = contain_plus

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.contain:
                for i in range(1, len(ret), 3):
                    ret[i] = ImageOps.invert(ret[i])  # invert flow pixel values when flipping
            elif self.is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = ImageOps.invert(ret[i])  # invert flow pixel values when flipping
                    # because Flip is at x-axis, x_optical_flow shows the velocity of object, after
                    # Flip, it should aim to the other direction.
            return ret
        else:
            return img_group


class GroupRandomVerticalFlip(object):
    """Only for RGB modality
    """

    def __init__(self, contain_plus=False):
        self.contain = contain_plus

    def __call__(self, img_group):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_TOP_BOTTOM) for img in img_group]
            if self.contain:
                for i in range(2, len(ret), 3):
                    ret[i] = ImageOps.invert(ret[i])
            return ret
        else:
            return img_group


class GroupShiftScaleRotate(object):
    def __init__(self, scale=1, angle_max=15, dx=0, dy=0, contain_plus=False):
        super(GroupShiftScaleRotate, self).__init__()
        self.scale = scale
        self.angle_max = angle_max
        self.dx = dx
        self.dy = dy
        self.contain = contain_plus

    def __call__(self, img_group):
        v = random.random()
        if v < 0.5:
            angle = np.random.uniform(-self.angle_max, self.angle_max)

            borderMode = cv2.BORDER_REFLECT_101
            # cv2.BORDER_REFLECT_101  cv2.BORDER_CONSTANT

            img_group = [np.array(img).astype(np.float32) for img in img_group]

            height, width = img_group[0].shape[:2]
            sx = self.scale
            sy = self.scale

            cc = math.cos(angle / 180 * math.pi) * (sx)
            ss = math.sin(angle / 180 * math.pi) * (sy)
            rotate_matrix = np.array([[cc, -ss], [ss, cc]])

            box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ], np.float32)
            box1 = box0 - np.array([width / 2, height / 2])
            box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + self.dx, height / 2 + self.dy])

            box0 = box0.astype(np.float32)
            box1 = box1.astype(np.float32)
            mat = cv2.getPerspectiveTransform(box0, box1)

            if self.contain:
                for i in range(0, len(img_group), 3):
                    img_group[i] = Image.fromarray(
                        cv2.warpPerspective(img_group[i], mat, (width, height), flags=cv2.INTER_NEAREST,
                                            borderMode=borderMode, borderValue=(0, 0, 0,)).astype(np.uint8))
                    img_group[i + 1] = Image.fromarray(img_group[i + 1]).convert('L')
                    img_group[i + 2] = Image.fromarray(img_group[i + 2]).convert('L')
                return img_group
            else:
                return [Image.fromarray(cv2.warpPerspective(img, mat, (width, height), flags=cv2.INTER_NEAREST,
                                                            borderMode=borderMode, borderValue=(0, 0, 0,)).astype(
                    np.uint8)
                    # cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101)]
                ) for img in img_group]
        else:
            return img_group


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        if isinstance(tensor, list) or len(tensor.size()) == 3 or tensor.size()[0] == 1:
            rep_mean = np.mean(self.mean)
            rep_std = np.mean(self.std)
            if isinstance(tensor, list):
                return [tensor[0].sub_(rep_mean).div_(rep_std), tensor[1].sub_(rep_mean).div_(rep_std)]
            return tensor.sub_(rep_mean).div_(rep_std)
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor


class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupOverSample(object):
    def __init__(self, crop_size, scale_size=None):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None

    def __call__(self, img_group):

        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        offsets = GroupMultiScaleCrop.fill_fix_offset(False, image_w, image_h, crop_w, crop_h)
        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                if img.mode == 'L' and i % 2 == 0:
                    flip_group.append(ImageOps.invert(flip_crop))
                else:
                    flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            oversample_group.extend(flip_group)
        return oversample_group


class GroupMultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img_group):

        im_size = img_group[0].size

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation)
                         for img in crop_img_group]
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret


class GroupRandomSizedCrop(object):
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        for attempt in range(10):
            area = img_group[0].size[0] * img_group[0].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img_group[0].size[0] and h <= img_group[0].size[1]:
                x1 = random.randint(0, img_group[0].size[0] - w)
                y1 = random.randint(0, img_group[0].size[1] - h)
                found = True
                break
        else:
            found = False
            x1 = 0
            y1 = 0

        if found:
            out_group = list()
            for img in img_group:
                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))
                out_group.append(img.resize((self.size, self.size), self.interpolation))
            return out_group
        else:
            # Fallback
            scale = GroupScale(self.size, interpolation=self.interpolation)
            crop = GroupRandomCrop(self.size)
            return crop(scale(img_group))


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll
        # self.dim = dim

    def __call__(self, img_group):
        # if self.dim == 3:
        #     return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)


class Stack_test_for_gray(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                tmp = np.concatenate(img_group, axis=2)
                slided_output = np.concatenate([tmp[:, :, x] for x in range(0, tmp.shape[2] - 1, 3)])
                return slided_output


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, num_segments, size, div=True, modality='RGB', new_length=1):
        self.div = div
        self.seg = num_segments
        self.size = size
        self.modality = modality
        self.new_length = new_length

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
            if self.modality in ['RGB', 'FlowPlusGray']:
                img = img.view(3 * self.new_length, self.seg, self.size, self.size)
            elif self.modality == 'CSP':
                img = img.view(self.seg, self.new_length, self.size, self.size)
                img = img.permute(1, 0, 2, 3)
            elif self.modality in ['SPECTRUM', 'CANCELLED', 'GRAY']:
                img = img.view(self.new_length, self.seg, self.size, self.size)
            elif self.modality == 'AdapPool':
                img = img.view(self.seg, self.new_length, self.size, self.size)
            elif self.modality == 'VideoGray':
                img = img.view(self.seg, self.size, self.size)
            elif self.modality == 'slowfast':
                seg, _, _ = img.shape
                # slow, fast
                img = img.view(self.new_length, seg, self.size, self.size)
            elif self.modality == 'LSTM':
                seg, _, _ = img.shape
                # TODO: 要同时给all和3frame用
                img = img.view(1, 64, self.size, self.size)
            elif self.modality == 'Ada':
                seg, _, _ = img.shape
                # TODO: 要同时给all和3frame用
                img = img.view(1, seg, self.size, self.size)

            else:
                img = img.view(2 * self.new_length, self.seg, self.size, self.size)
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()

        return img.float().div(255) if self.div else img.float()


class IdentityTransform(object):

    def __call__(self, data):
        return data


class NormalTransform(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        return (data - self.mean) / self.std


class ToGray(object):
    def __init__(self, contain_plus=False):
        self.contain = contain_plus

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            print('convert to Gary should be a picture not ndarray')
            raise ValueError
        else:
            if self.contain:
                for i in range(0, len(pic), 3):
                    pic[i] = pic[i].convert('L')
                return pic
            else:
                # handle PIL Image
                return [each_pic.convert('L') for each_pic in pic]


class ToSpecturm(object):
    def __call__(self, pic_group):
        out_pic = []
        if isinstance(pic_group, np.ndarray):
            raise ValueError
            # # handle nparray after Stack
            # picnum = np.shape(pic_group)
            # for num in picnum:
            #     img = pic_group[num,:,:]
            #     f = np.fft.fft2(img)  # 快速傅里叶变换算法得到频率分布
            #     fshift = np.fft.fftshift(f)  # 默认结果中心点位置是在左上角，转移到中间位置
            #     fimg = np.log(np.abs(fshift))  # fft 结果是复数，求绝对值结果才是振幅
            #     out_pic.append(Image.fromarray(fimg))
        else:
            # handle PIL Image
            for img in pic_group:
                img_arr = np.array(img)
                f = np.fft.fft2(img_arr)  # 快速傅里叶变换算法得到频率分布
                fshift = np.fft.fftshift(f)  # 默认结果中心点位置是在左上角，转移到中间位置
                fimg = np.log(np.abs(fshift))  # fft 结果是复数，求绝对值结果才是振幅
                out_pic.append(Image.fromarray(fimg).convert('L'))
        if not len(out_pic) == len(pic_group):
            print('sizes of IO are different')
            raise ValueError
        return out_pic


class ToCancelled(object):
    def __init__(self, D0=5):
        self.D0 = D0

    def __call__(self, pic_group):
        out_pic = []
        if isinstance(pic_group, np.ndarray):
            print('Input should be PIL')
            raise ValueError
            # # handle nparray after Stack
            # picnum = np.shape(pic_group)[0]
            # kernel = filter1(np.shape(pic_group)[1], self.D0)
            # for num in picnum:
            #     img = pic_group[num,:,:]
            #     f = np.fft.fft2(img)  # 快速傅里叶变换算法得到频率分布
            #     fshift = np.fft.fftshift(f)  # 默认结果中心点位置是在左上角，转移到中间位置
            #     fimg = np.log(np.abs(fshift))  # fft 结果是复数，求绝对值结果才是振幅
            #
            #     out_pic.append(Image.fromarray(fimg))
        else:
            # handle PIL Image
            for img in pic_group:
                kernel = filter1(img.size, self.D0)
                img_arr = np.array(img)
                f = np.fft.fft2(img_arr)  # 快速傅里叶变换算法得到频率分布
                fshift = np.fft.fftshift(f)  # 默认结果中心点位置是在左上角，转移到中间位置
                hpf_fft = kernel * fshift
                hpf_img = np.fft.ifft2(np.fft.ifftshift(hpf_fft))

                hpfImg = np.abs(hpf_img)
                cancelled_img = abs(hpfImg - img)
                out_pic.append(Image.fromarray(cancelled_img).convert('L'))
        if not len(out_pic) == len(pic_group):
            print('sizes of IO are different')
            raise ValueError
        return out_pic


class ToImgTsn(object):
    def __init__(self, D0):
        self.D0 = D0

    def __call__(self, pic_group):
        out_pic = []
        if isinstance(pic_group, np.ndarray):
            print('Input should be PIL')
            raise ValueError
        else:
            # handle PIL Image
            for img in pic_group:
                kernel = filter1(img.size, self.D0)
                img_arr = np.array(img)
                f = np.fft.fft2(img_arr)  # 快速傅里叶变换算法得到频率分布
                fshift = np.fft.fftshift(f)  # 默认结果中心点位置是在左上角，转移到中间位置
                hpf_fft = kernel * fshift
                hpf_img = np.fft.ifft2(np.fft.ifftshift(hpf_fft))

                hpfImg = np.abs(hpf_img)
                cancelled_img = abs(hpfImg - img)
                out_pic.append(Image.fromarray(cancelled_img))
        if len(out_pic) == len(pic_group):
            print('sizes of IO are different')
            raise ValueError
        return out_pic

def filter1(size ,D0):
    epsilon = 1.4E-45
    kernel = np.zeros(size)
    height, width = size
    for u in range(height):
        for v in range(width):
            D = np.sqrt(math.pow((u - height / 2), 2) + math.pow((v - width / 2), 2))
            kernel[u, v] = 1 - 1 / (1 + math.pow((D / D0), 2)) + epsilon
    return kernel

def filter2(size, D0):
    epsilon = 1.4E-45
    kernel = np.zeros(size)
    height, width = size
    for u in range(height):
        for v in range(width):
            D = np.sqrt(math.pow((u - height / 2), 2) + math.pow((v - width / 2), 2))
            if D > D0:
                kernel[u, v] = 1.0
            else:
                kernel[u, v] = epsilon
    return kernel

if __name__ == "__main__":
    trans = torchvision.transforms.Compose([
        GroupScale(256),
        GroupRandomCrop(224),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(
            mean=[.485, .456, .406],
            std=[.229, .224, .225]
        )]
    )

    im = Image.open('../tensorflow-model-zoo.torch/lena_299.png')

    color_group = [im] * 3
    rst = trans(color_group)

    gray_group = [im.convert('L')] * 9
    gray_rst = trans(gray_group)

    trans2 = torchvision.transforms.Compose([
        GroupRandomSizedCrop(256),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(
            mean=[.485, .456, .406],
            std=[.229, .224, .225])
    ])
    print(trans2(color_group))



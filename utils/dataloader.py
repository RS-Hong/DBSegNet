import os
# import mmcv
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.utils import preprocess_input, cvtColor
from typing import Dict, List, Optional, Sequence, Tuple, Union
from numpy import random


# class PhotoMetricDistortion(object):
#     '''
#      mmseg 中的颜色增强
#      if random.randint(2):
#      return self.convert(
#              img,
#              alpha=random.uniform(self.contrast_lower, self.contrast_upper))
#      '''
#
#     def __init__(self,
#                  brightness_delta: int = 32,
#                  contrast_range: Sequence[float] = (0.5, 1.5),
#                  saturation_range: Sequence[float] = (0.5, 1.5),
#                  hue_delta: int = 18):
#         self.brightness_delta = brightness_delta
#         self.contrast_lower, self.contrast_upper = contrast_range
#         self.saturation_lower, self.saturation_upper = saturation_range
#         self.hue_delta = hue_delta
#
#     def convert(self,
#                 img: np.ndarray,
#                 alpha: int = 1,
#                 beta: int = 0) -> np.ndarray:
#         """Multiple with alpha and add beat with clip.
#
#         Args:
#             img (np.ndarray): The input image.
#             alpha (int): Image weights, change the contrast/saturation
#                 of the image. Default: 1
#             beta (int): Image bias, change the brightness of the
#                 image. Default: 0
#
#         Returns:
#             np.ndarray: The transformed image.
#         """
#
#         img = img.astype(np.float32) * alpha + beta
#         img = np.clip(img, 0, 255)
#         return img.astype(np.uint8)
#
#     def brightness(self, img: np.ndarray) -> np.ndarray:
#         """Brightness distortion.
#
#         Args:
#             img (np.ndarray): The input image.
#         Returns:
#             np.ndarray: Image after brightness change.
#         """
#
#         if random.randint(2):
#             return self.convert(
#                 img,
#                 beta=random.uniform(-self.brightness_delta,
#                                     self.brightness_delta))
#         return img
#
#     def contrast(self, img: np.ndarray) -> np.ndarray:
#         """Contrast distortion.
#
#         Args:
#             img (np.ndarray): The input image.
#         Returns:
#             np.ndarray: Image after contrast change.
#         """
#
#         if random.randint(2):
#             return self.convert(
#                 img,
#                 alpha=random.uniform(self.contrast_lower, self.contrast_upper))
#         return img
#
#     def saturation(self, img: np.ndarray) -> np.ndarray:
#         """Saturation distortion.
#
#         Args:
#             img (np.ndarray): The input image.
#         Returns:
#             np.ndarray: Image after saturation change.
#         """
#
#         if random.randint(2):
#             img = mmcv.bgr2hsv(img)
#             img[:, :, 1] = self.convert(
#                 img[:, :, 1],
#                 alpha=random.uniform(self.saturation_lower,
#                                      self.saturation_upper))
#             img = mmcv.hsv2bgr(img)
#         return img
#
#     def hue(self, img: np.ndarray) -> np.ndarray:
#         """Hue distortion.
#
#         Args:
#             img (np.ndarray): The input image.
#         Returns:
#             np.ndarray: Image after hue change.
#         """
#
#         if random.randint(2):
#             img = mmcv.bgr2hsv(img)
#             img[:, :,
#             0] = (img[:, :, 0].astype(int) +
#                   random.randint(-self.hue_delta, self.hue_delta)) % 180
#             img = mmcv.hsv2bgr(img)
#         return img


def resize_img(self, img, scale, keep_ratio, interpolation='bilinear', backend='cv2'):
    """Resize images with ``results['scale']``."""

    if img is not None:
        if keep_ratio:
            img, scale_factor = mmcv.imrescale(
                img,
                scale,
                interpolation=interpolation,
                return_scale=True,
                backend=backend)
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_h, new_w = img.shape[:2]
            h, w = img.shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = mmcv.imresize(
                img,
                scale,
                interpolation=interpolation,
                return_scale=True,
                backend=backend)
        return img

#
# def resize_bboxes(self, img):
#     """Resize bounding boxes with ``results['scale_factor']``."""
#     if results.get('gt_bboxes', None) is not None:
#         bboxes = results['gt_bboxes'] * np.tile(
#             np.array(results['scale_factor']), 2)
#         if self.clip_object_border:
#             bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0,
#                                       results['img_shape'][1])
#             bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0,
#                                       results['img_shape'][0])
#         results['gt_bboxes'] = bboxes


class SegmentationDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(SegmentationDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path

    def __len__(self):
        return self.length


    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        # name            = annotation_line.split()[0]
        name = annotation_line.strip()

        #-------------------------------#
        #   从文件中读取图像
        #-------------------------------#
        jpg         = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/JPEGImages"), name + ".tif"))
        png         = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/SegmentationClass"), name + ".tif"))
        #-------------------------------#
        #   数据增强
        #-------------------------------#
        # jpg, png    = self.get_random_data(jpg, png, self.input_shape, random = self.train)

        jpg_aug, png_aug    = self.get_random_data(jpg, png, self.input_shape, random = self.train)

        # 计算原始和增强后前景像素数量
        original_fg_count = np.sum(np.array(png) == 1)
        aug_fg_count = np.sum(np.array(png_aug) == 1)

        # 如果增强后前景像素比例低于50%，禁用增强
        if self.train and original_fg_count > 0 and aug_fg_count < 0.7 * original_fg_count:
            jpg_aug, png_aug = self.get_random_data(jpg, png, self.input_shape, random=False)

        jpg = np.transpose(preprocess_input(np.array(jpg_aug, np.float64)), [2,0,1])
        png = np.array(png_aug)
        png[png >= self.num_classes] = self.num_classes
        #-------------------------------------------------------#
        #   转化成one_hot的形式
        #   在这里需要+1是因为voc数据集有些标签具有白边部分
        #   我们需要将白边部分进行忽略，+1的目的是方便忽略。
        #-------------------------------------------------------#
        seg_labels  = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels  = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return jpg, png, seg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def transform(self, img, label, scale, ratio_range = (0.5, 2.0), keep_ratio = True, crop_ratio = 0.75, flip_ratio = 0.5):

        # PhotoMetricDistortion

        img = self.brightness(img)
        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            img = self.contrast(img)
        # random saturation
        img = self.saturation(img)
        # random hue
        img = self.hue(img)
        # random contrast
        if mode == 0:
            img = self.contrast(img)

        return img, label

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        image   = cvtColor(image)
        label   = Image.fromarray(np.array(label))
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape

        if not random:
            iw, ih  = image.size
            scale   = min(w/iw, h/ih)
            nw      = int(iw*scale)
            nh      = int(ih*scale)

            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', [w, h], (128,128,128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))

            label       = label.resize((nw,nh), Image.NEAREST)
            new_label   = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w-nw)//2, (h-nh)//2))
            return new_image, new_label

        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.5, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        label = label.resize((nw,nh), Image.NEAREST)
        
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_label = Image.new('L', (w,h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        image_data      = np.array(image, np.uint8)
        #------------------------------------------#
        #   高斯模糊
        #------------------------------------------#
        blur = self.rand() < 0.25
        if blur: 
            image_data = cv2.GaussianBlur(image_data, (5, 5), 0)

        #------------------------------------------#
        #   旋转
        #------------------------------------------#
        rotate = self.rand() < 0.25
        if rotate: 
            center      = (w // 2, h // 2)
            rotation    = np.random.randint(-10, 11)
            M           = cv2.getRotationMatrix2D(center, -rotation, scale=1)
            image_data  = cv2.warpAffine(image_data, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128,128,128))
            label       = cv2.warpAffine(np.array(label, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0))

        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        
        return image_data, label


def seg_dataset_collate(batch):
    images      = []
    pngs        = []
    seg_labels  = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs        = torch.from_numpy(np.array(pngs)).long()
    seg_labels  = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels

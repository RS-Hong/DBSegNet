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
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path, aug_mode="default", dataset_type="voc", hrgldd_channels=(0, 1, 2), small_target_strategy=False, small_target_max_pixels=512, small_target_crop_probability=0.70, small_target_scale_range=(1.25, 2.0)):
        super(SegmentationDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path
        self.aug_mode           = aug_mode.lower()
        self.dataset_type       = dataset_type.lower()
        self.hrgldd_channels    = tuple(hrgldd_channels)
        self.small_target_strategy = bool(small_target_strategy)
        self.small_target_max_pixels = int(small_target_max_pixels)
        self.small_target_crop_probability = float(small_target_crop_probability)
        self.small_target_scale_range = tuple(float(value) for value in small_target_scale_range)
        self._hrgldd_cache      = {}
        self._cas_mask_cache    = {}
        if self.aug_mode not in ("default", "weak", "planet_mild", "resize_only", "direct_resize"):
            raise ValueError("aug_mode must be 'default', 'weak', 'planet_mild', 'resize_only' or 'direct_resize'.")
        if self.dataset_type not in ("voc", "bijie", "cas", "hrgldd"):
            raise ValueError("dataset_type must be 'voc', 'bijie', 'cas' or 'hrgldd'.")

    def __len__(self):
        return self.length


    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        # name            = annotation_line.split()[0]
        name = annotation_line.strip()
        if self.dataset_type == "hrgldd":
            return self.get_hrgldd_data(name)

        #-------------------------------#
        #   从文件中读取图像
        #-------------------------------#
        jpg, png = self.load_image_mask(name)
        if self.num_classes == 2:
            png = Image.fromarray((np.asarray(png) > 0).astype(np.uint8))
        #-------------------------------#
        #   数据增强
        #-------------------------------#
        # jpg, png    = self.get_random_data(jpg, png, self.input_shape, random = self.train)

        if self.train and self.aug_mode == "default":
            jpg_aug, png_aug = self.get_random_data(jpg, png, self.input_shape, random=True)
            original_fg_count = np.sum(np.array(png) == 1)
            aug_fg_count = np.sum(np.array(png_aug) == 1)
            if original_fg_count > 0 and aug_fg_count < 0.7 * original_fg_count:
                jpg_aug, png_aug = self.get_random_data(jpg, png, self.input_shape, random=False)
        elif self.train and self.aug_mode == "weak":
            jpg_aug, png_aug = self.get_weak_data(jpg, png, self.input_shape)
        elif self.train and self.aug_mode == "planet_mild":
            jpg_aug, png_aug = self.get_planet_mild_data(jpg, png, self.input_shape)
        elif self.aug_mode == "direct_resize":
            jpg_aug, png_aug = self.get_direct_resize_data(jpg, png, self.input_shape)
        else:
            jpg_aug, png_aug = self.get_random_data(jpg, png, self.input_shape, random=False)

        jpg = np.transpose(preprocess_input(np.array(jpg_aug, np.float64)), [2,0,1])
        png = np.array(png_aug, dtype=np.int64)
        if self.num_classes == 2 or self.dataset_type in ("bijie", "cas"):
            png = (png > 0).astype(np.int64)
        else:
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

    def get_hrgldd_arrays(self, split):
        if split not in self._hrgldd_cache:
            x_path = os.path.join(self.dataset_path, f"{split}X.npy")
            y_path = os.path.join(self.dataset_path, f"{split}Y.npy")
            self._hrgldd_cache[split] = (
                np.load(x_path, mmap_mode="r"),
                np.load(y_path, mmap_mode="r"),
            )
        return self._hrgldd_cache[split]

    def get_hrgldd_data(self, name):
        split, index = name.replace("\\", "/").split("/", 1)
        index = int(index)
        xs, ys = self.get_hrgldd_arrays(split)
        image = np.array(xs[index][..., self.hrgldd_channels], dtype=np.float32)
        mask = np.array(ys[index], dtype=np.float32).squeeze()

        h, w = self.input_shape
        if image.shape[0] != h or image.shape[1] != w:
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        if self.train and self.aug_mode == "weak":
            if self.rand() < 0.5:
                image = np.flip(image, axis=1)
                mask = np.flip(mask, axis=1)
            if self.rand() < 0.5:
                image = np.flip(image, axis=0)
                mask = np.flip(mask, axis=0)
            rotate_k = np.random.randint(0, 4)
            if rotate_k:
                image = np.rot90(image, rotate_k, axes=(0, 1))
                mask = np.rot90(mask, rotate_k, axes=(0, 1))

        image = self.preprocess_hrgldd_image(image)
        mask = (np.ascontiguousarray(mask) > 0.5).astype(np.int64)

        seg_labels = np.eye(self.num_classes + 1)[mask.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))
        return image, mask, seg_labels

    @staticmethod
    def preprocess_hrgldd_image(image):
        image = np.ascontiguousarray(image * 255.0, dtype=np.float32)
        if image.shape[-1] == 3:
            image = preprocess_input(image)
        else:
            processed = np.empty_like(image, dtype=np.float32)
            processed[..., :3] = preprocess_input(image[..., :3].copy())
            processed[..., 3:] = (image[..., 3:] - 127.5) / 58.0
            image = processed
        return np.transpose(image, [2, 0, 1])

    def load_image_mask(self, name):
        if self.dataset_type == "cas":
            subdataset, filename = name.replace("\\", "/").split("/", 1)
            image_path = os.path.join(self.dataset_path, subdataset, "img", filename)
            mask_dir = os.path.join(self.dataset_path, subdataset, "mask")
            mask_path = os.path.join(mask_dir, filename)
            if not os.path.exists(mask_path):
                if subdataset not in self._cas_mask_cache:
                    self._cas_mask_cache[subdataset] = {
                        f.lower(): f for f in os.listdir(mask_dir)
                        if os.path.isfile(os.path.join(mask_dir, f))
                    }
                mask_name = self._cas_mask_cache[subdataset].get(filename.lower())
                if mask_name is None:
                    raise FileNotFoundError(mask_path)
                mask_path = os.path.join(mask_dir, mask_name)
            return Image.open(image_path), Image.open(mask_path)

        if self.dataset_type == "bijie":
            subset, filename = name.replace("\\", "/").split("/", 1)
            if subset == "landslide":
                image_path = os.path.join(self.dataset_path, "landslide", "image", filename)
                mask_path = os.path.join(self.dataset_path, "landslide", "mask", filename)
                image = Image.open(image_path)
                mask = Image.open(mask_path)
            elif subset == "non-landslide":
                image_path = os.path.join(self.dataset_path, "non-landslide", "image", filename)
                image = Image.open(image_path)
                mask = Image.new("L", image.size, 0)
            else:
                raise ValueError(f"Unsupported Bijie subset in line: {name}")
            return image, mask

        image = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/JPEGImages"), name + ".tif"))
        mask = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/SegmentationClass"), name + ".tif"))
        return image, mask

    def get_direct_resize_data(self, image, label, input_shape):
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))
        h, w = input_shape

        image = image.resize((w, h), Image.BICUBIC)
        label = label.resize((w, h), Image.NEAREST)
        return image, label

    def get_weak_data(self, image, label, input_shape):
        image, label = self.get_random_data(image, label, input_shape, random=False)

        if self.rand() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        if self.rand() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            label = label.transpose(Image.FLIP_TOP_BOTTOM)

        rotate_k = np.random.randint(0, 4)
        if rotate_k == 1:
            image = image.transpose(Image.ROTATE_90)
            label = label.transpose(Image.ROTATE_90)
        elif rotate_k == 2:
            image = image.transpose(Image.ROTATE_180)
            label = label.transpose(Image.ROTATE_180)
        elif rotate_k == 3:
            image = image.transpose(Image.ROTATE_270)
            label = label.transpose(Image.ROTATE_270)
        return image, label

    def get_planet_mild_data(self, image, label, input_shape):
        image, label = self.get_weak_data(image, label, input_shape)
        foreground_pixels = int(np.count_nonzero(np.asarray(label)))
        is_small_target = 0 < foreground_pixels <= self.small_target_max_pixels
        crop_probability = (
            self.small_target_crop_probability
            if self.small_target_strategy and is_small_target
            else (0.25 if is_small_target else 0.50)
        )
        scale_range = (
            self.small_target_scale_range
            if self.small_target_strategy and is_small_target
            else (1.0, 1.5)
        )
        if self.rand() < crop_probability:
            image, label = self.get_target_preserving_crop(
                image, label, input_shape, scale_range=scale_range
            )
        if self.rand() >= 0.8:
            return image, label

        rgb = np.asarray(image, dtype=np.float32) / 255.0
        contrast = self.rand(0.95, 1.05)
        gamma = self.rand(0.95, 1.05)
        channel_gain = np.array([self.rand(0.97, 1.03) for _ in range(3)], dtype=np.float32)
        rgb = np.clip((rgb - 0.5) * contrast + 0.5, 0.0, 1.0)
        rgb = np.power(rgb, gamma) * channel_gain
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = np.mod(hsv[..., 0] + self.rand(-1.0, 1.0), 180.0)
        hsv[..., 1] = np.clip(hsv[..., 1] * self.rand(0.92, 1.08), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * self.rand(0.95, 1.05), 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return Image.fromarray(image), label

    def get_target_preserving_crop(self, image, label, input_shape, max_attempts=10, scale_range=(1.0, 1.5)):
        """Randomly zoom and crop while retaining useful foreground content."""
        h, w = input_shape
        scale = self.rand(*scale_range)
        scaled_w = max(w, int(round(w * scale)))
        scaled_h = max(h, int(round(h * scale)))
        image = image.resize((scaled_w, scaled_h), Image.BICUBIC)
        label = label.resize((scaled_w, scaled_h), Image.NEAREST)
        scaled_mask = np.asarray(label)
        scaled_foreground = int(np.count_nonzero(scaled_mask))

        best_image = None
        best_label = None
        best_foreground = -1
        required_foreground = min(
            scaled_foreground,
            max(128, int(np.ceil(0.70 * scaled_foreground))),
        )

        for _ in range(max_attempts):
            left = np.random.randint(0, scaled_w - w + 1)
            top = np.random.randint(0, scaled_h - h + 1)
            crop_box = (left, top, left + w, top + h)
            image_crop = image.crop(crop_box)
            label_crop = label.crop(crop_box)
            crop_foreground = int(np.count_nonzero(np.asarray(label_crop)))
            if crop_foreground > best_foreground:
                best_image = image_crop
                best_label = label_crop
                best_foreground = crop_foreground
            if scaled_foreground == 0 or crop_foreground >= required_foreground:
                return image_crop, label_crop

        if scaled_foreground > 0 and best_foreground == 0:
            foreground_y, foreground_x = np.where(scaled_mask > 0)
            center_x = int(np.median(foreground_x))
            center_y = int(np.median(foreground_y))
            left = int(np.clip(center_x - w // 2, 0, scaled_w - w))
            top = int(np.clip(center_y - h // 2, 0, scaled_h - h))
            crop_box = (left, top, left + w, top + h)
            best_image = image.crop(crop_box)
            best_label = label.crop(crop_box)

        return best_image, best_label

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

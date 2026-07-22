import os
import json

import matplotlib
import torch
import torch.nn.functional as F

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal

import cv2
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .utils import cvtColor, preprocess_input, resize_image
from .utils_metrics import compute_mIoU


class LossHistory():
    def __init__(self, log_dir, model, input_shape, input_channels=3):
        self.log_dir = log_dir
        self.losses = []
        self.val_loss = []

        os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        try:
            dummy_input = torch.randn(2, input_channels, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--', linewidth=2,
                     label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")


class EvalCallback():
    def __init__(self, net, input_shape, num_classes, image_ids, dataset_path, log_dir, cuda, \
                 miou_out_path=".temp_miou_out", eval_flag=True, period=1, dataset_type="voc", hrgldd_channels=(0, 1, 2), metric_prefix=""):
        super(EvalCallback, self).__init__()

        self.net = net
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.image_ids = image_ids
        self.dataset_path = dataset_path
        self.log_dir = log_dir
        self.cuda = cuda
        self.miou_out_path = miou_out_path
        self.eval_flag = eval_flag
        self.period = period
        self.dataset_type = dataset_type.lower()
        self.hrgldd_channels = tuple(hrgldd_channels)
        self.metric_prefix = metric_prefix.strip("_")
        self._hrgldd_cache = {}
        self._cas_mask_cache = {}
        if self.dataset_type not in ("voc", "bijie", "cas", "hrgldd"):
            raise ValueError("dataset_type must be 'voc', 'bijie', 'cas' or 'hrgldd'.")

        # self.image_ids          = [image_id.split()[0] for image_id in image_ids]
        self.image_ids = [image_id.strip() for image_id in image_ids]
        self.mious = [0]
        self.best_miou = 0
        self.epoches = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, self.metric_filename("epoch_miou.txt")), 'a') as f:
                f.write(str(0))
                f.write("\n")

    def metric_filename(self, default_name):
        if not self.metric_prefix:
            return default_name
        names = {
            "epoch_miou.txt": f"epoch_{self.metric_prefix}_miou.txt",
            "epoch_metrics.jsonl": f"epoch_{self.metric_prefix}_metrics.jsonl",
            "best_miou_weights.pth": f"best_{self.metric_prefix}_miou_weights.pth",
            "best_metrics.json": f"best_{self.metric_prefix}_metrics.json",
            "epoch_miou.png": f"epoch_{self.metric_prefix}_miou.png",
        }
        return names.get(default_name, f"{self.metric_prefix}_{default_name}")

    @staticmethod
    def safe_id(image_id):
        return image_id.strip().replace("\\", "/").replace("/", "__").replace(".", "_")

    def get_image_path(self, image_id):
        if self.dataset_type == "hrgldd":
            return image_id
        if self.dataset_type == "cas":
            subdataset, filename = image_id.replace("\\", "/").split("/", 1)
            return os.path.join(self.dataset_path, subdataset, "img", filename)
        if self.dataset_type == "bijie":
            subset, filename = image_id.replace("\\", "/").split("/", 1)
            if subset == "landslide":
                return os.path.join(self.dataset_path, "landslide", "image", filename)
            if subset == "non-landslide":
                return os.path.join(self.dataset_path, "non-landslide", "image", filename)
            raise ValueError(f"Unsupported Bijie subset in line: {image_id}")
        return os.path.join(self.dataset_path, "VOC2007/JPEGImages/" + image_id + ".tif")

    def get_gt_mask(self, image_id):
        if self.dataset_type == "hrgldd":
            split, index = image_id.replace("\\", "/").split("/", 1)
            _, ys = self.get_hrgldd_arrays(split)
            mask = np.array(ys[int(index)], dtype=np.float32).squeeze()
            return Image.fromarray((mask > 0.5).astype(np.uint8))
        if self.dataset_type == "cas":
            subdataset, filename = image_id.replace("\\", "/").split("/", 1)
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
            mask = np.array(Image.open(mask_path), dtype=np.uint8)
            return Image.fromarray((mask > 0).astype(np.uint8))
        if self.dataset_type == "bijie":
            subset, filename = image_id.replace("\\", "/").split("/", 1)
            if subset == "landslide":
                mask_path = os.path.join(self.dataset_path, "landslide", "mask", filename)
                mask = np.array(Image.open(mask_path), dtype=np.uint8)
                return Image.fromarray((mask > 0).astype(np.uint8))
            if subset == "non-landslide":
                image = Image.open(self.get_image_path(image_id))
                return Image.new("L", image.size, 0)
            raise ValueError(f"Unsupported Bijie subset in line: {image_id}")
        return Image.open(os.path.join(self.dataset_path, "VOC2007/SegmentationClass/" + image_id + ".tif"))

    def get_hrgldd_arrays(self, split):
        if split not in self._hrgldd_cache:
            x_path = os.path.join(self.dataset_path, f"{split}X.npy")
            y_path = os.path.join(self.dataset_path, f"{split}Y.npy")
            self._hrgldd_cache[split] = (
                np.load(x_path, mmap_mode="r"),
                np.load(y_path, mmap_mode="r"),
            )
        return self._hrgldd_cache[split]

    def get_main_output(self, outputs):
        if isinstance(outputs, dict):
            return outputs['out']
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
            if isinstance(outputs, dict):
                return outputs['out']
            return outputs
        return outputs

    def get_miou_png(self, image):
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        # ---------------------------------------------------------#
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))

        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)),
            0
        )

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            # 先前向一次
            outputs = self.net(images)

            # 兼容 tensor / dict / tuple
            outputs = self.net(images)
            outputs = self.get_main_output(outputs)

            # outputs 现在应该是 [B, C, H, W]
            if len(outputs.shape) == 4:
                pr = outputs[0]
            elif len(outputs.shape) == 3:
                pr = outputs
            else:
                raise ValueError(f"Unexpected output shape: {outputs.shape}")

            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()

            # --------------------------------------#
            #   将灰条部分截取掉
            # --------------------------------------#
            pr = pr[
                 int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)
                 ]

            # ---------------------------------------------------#
            #   进行图片的resize
            # ---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)

            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            pr = pr.argmax(axis=-1)

        image = Image.fromarray(np.uint8(pr))
        return image

    def get_hrgldd_miou_png(self, image_id):
        split, index = image_id.replace("\\", "/").split("/", 1)
        xs, _ = self.get_hrgldd_arrays(split)
        image = np.array(xs[int(index)][..., self.hrgldd_channels], dtype=np.float32)
        orininal_h, orininal_w = image.shape[:2]

        if image.shape[0] != self.input_shape[0] or image.shape[1] != self.input_shape[1]:
            image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]), interpolation=cv2.INTER_LINEAR)

        image = np.ascontiguousarray(image * 255.0, dtype=np.float32)
        if image.shape[-1] == 3:
            image = preprocess_input(image)
        else:
            processed = np.empty_like(image, dtype=np.float32)
            processed[..., :3] = preprocess_input(image[..., :3].copy())
            processed[..., 3:] = (image[..., 3:] - 127.5) / 58.0
            image = processed
        image_data = np.expand_dims(
            np.transpose(image, (2, 0, 1)),
            0
        )

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.get_main_output(outputs)
            pr = outputs[0] if len(outputs.shape) == 4 else outputs
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            if pr.shape[0] != orininal_h or pr.shape[1] != orininal_w:
                pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)
        return Image.fromarray(np.uint8(pr))

    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            if self.dataset_type in ("bijie", "cas", "hrgldd"):
                gt_dir = os.path.join(self.miou_out_path, "ground-truth")
            else:
                gt_dir = os.path.join(self.dataset_path, "VOC2007/SegmentationClass/")
            pred_dir = os.path.join(self.miou_out_path, 'detection-results')
            if not os.path.exists(self.miou_out_path):
                os.makedirs(self.miou_out_path)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            if self.dataset_type in ("bijie", "cas", "hrgldd") and not os.path.exists(gt_dir):
                os.makedirs(gt_dir)
            label = self.metric_prefix or "primary_val"
            print(f"Get miou: {label}.")
            eval_ids = []
            for image_id in tqdm(self.image_ids):
                # -------------------------------#
                #   从文件中读取图像
                # -------------------------------#
                if self.dataset_type == "hrgldd":
                    image = self.get_hrgldd_miou_png(image_id)
                else:
                    image_path = self.get_image_path(image_id)
                    image = Image.open(image_path)
                    image = self.get_miou_png(image)
                eval_id = self.safe_id(image_id) if self.dataset_type in ("bijie", "cas", "hrgldd") else image_id
                if self.dataset_type in ("bijie", "cas", "hrgldd"):
                    self.get_gt_mask(image_id).save(os.path.join(gt_dir, eval_id + ".tif"))
                image.save(os.path.join(pred_dir, eval_id + ".png"))
                eval_ids.append(eval_id)

            print("Calculate miou.")
            _, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, eval_ids, self.num_classes, None)
            print(IoUs)
            print(IoUs)
            temp_miou = np.nanmean(IoUs) * 100

            self.mious.append(temp_miou)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, self.metric_filename("epoch_miou.txt")), 'a') as f:
                f.write(str(temp_miou))
                f.write("\n")

            metrics = {
                "epoch": int(epoch),
                "evaluation": label,
                "mIoU": float(temp_miou),
                "class_iou": (IoUs * 100).tolist(),
                "recall": (PA_Recall * 100).tolist(),
                "precision": (Precision * 100).tolist(),
            }
            with open(os.path.join(self.log_dir, self.metric_filename("epoch_metrics.jsonl")), "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            if temp_miou > self.best_miou:
                self.best_miou = temp_miou
                best_metrics = dict(metrics)
                best_metrics["best_miou_epoch"] = int(epoch)
                best_metrics["best_miou"] = float(temp_miou)
                save_model = model_eval.module if hasattr(model_eval, "module") else model_eval
                best_weights_name = self.metric_filename("best_miou_weights.pth")
                best_metrics_name = self.metric_filename("best_metrics.json")
                torch.save(save_model.state_dict(), os.path.join(self.log_dir, best_weights_name))
                run_dir = os.path.dirname(self.log_dir)
                if run_dir and run_dir != self.log_dir:
                    torch.save(save_model.state_dict(), os.path.join(run_dir, best_weights_name))
                with open(os.path.join(self.log_dir, best_metrics_name), "w", encoding="utf-8") as f:
                    json.dump(best_metrics, f, indent=2, ensure_ascii=False)
                if run_dir and run_dir != self.log_dir:
                    with open(os.path.join(run_dir, best_metrics_name), "w", encoding="utf-8") as f:
                        json.dump(best_metrics, f, indent=2, ensure_ascii=False)

            plt.figure()
            plt.plot(self.epoches, self.mious, 'red', linewidth=2, label=label + ' miou')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Miou')
            plt.title('A Miou Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, self.metric_filename("epoch_miou.png")))
            plt.cla()
            plt.close("all")

            print("Get miou done.")
            shutil.rmtree(self.miou_out_path)

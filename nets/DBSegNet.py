"""Final DBSegNet architecture used in the paper.

The encoder combines CNN and Transformer streams with cross-branch residual
exchange and dense complementary gating. Bidirectional cross-attention is used
at Stage 3. The decoder aggregates all four scales in top-down and bottom-up
directions.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.encoder import ConvBNAct, DualEncoder, LayerNorm2d, PHI


class BidirectionalMultiScaleDecoder(nn.Module):
    def __init__(self, in_channels: List[int], decoder_dim: int, num_classes: int):
        super().__init__()
        self.proj = nn.ModuleList(
            [ConvBNAct(channels, decoder_dim, 1, 1, 0) for channels in in_channels]
        )
        self.down = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fuse = nn.Sequential(
            ConvBNAct(decoder_dim * 8, decoder_dim * 4, 3, 1, 1),
            ConvBNAct(decoder_dim * 4, decoder_dim * 2, 3, 1, 1),
            ConvBNAct(decoder_dim * 2, decoder_dim, 3, 1, 1),
            nn.Dropout(0.1),
            nn.Conv2d(decoder_dim, num_classes, 1),
        )

    @staticmethod
    def _resize(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == reference.shape[-2:]:
            return x
        return F.interpolate(
            x, size=reference.shape[-2:], mode="bilinear", align_corners=False
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        f1, f2, f3, f4 = [
            projection(feature)
            for projection, feature in zip(self.proj, features)
        ]

        td4 = f4
        td3 = f3 + self._resize(td4, f3)
        td2 = f2 + self._resize(td3, f2)
        td1 = f1 + self._resize(td2, f1)

        bu1 = f1
        bu2 = f2 + self._resize(self.down(bu1), f2)
        bu3 = f3 + self._resize(self.down(bu2), f3)
        bu4 = f4 + self._resize(self.down(bu3), f4)

        target = f1
        features = [
            td1,
            self._resize(td2, target),
            self._resize(td3, target),
            self._resize(td4, target),
            bu1,
            self._resize(bu2, target),
            self._resize(bu3, target),
            self._resize(bu4, target),
        ]
        return self.fuse(torch.cat(features, dim=1))


class DBSegNet(nn.Module):
    """Final dual-branch DBSegNet.

    ``branch``, ``fusion_mode`` and ``use_cross_attention`` are retained only
    for the paper's controlled ablations. The final model uses ``dual``,
    ``gated`` and ``True`` respectively.
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 3,
        branch: str = "dual",
        fusion_mode: str = "gated",
        use_cross_attention: bool = True,
    ):
        super().__init__()
        if branch not in {"dual", "cnn", "tr"}:
            raise ValueError("branch must be 'dual', 'cnn' or 'tr'")
        if fusion_mode not in {"gated", "sum"}:
            raise ValueError("fusion_mode must be 'gated' or 'sum'")

        cfg = PHI["cvmv_tiny_fast"]
        fusion_type = "gate" if fusion_mode == "gated" else "sum"
        self.backbone = DualEncoder(
            cfg["dims"],
            cfg["depths_cnn"],
            cfg["depths_tr"],
            cfg["heads"],
            cfg["sr"],
            cfg["drop_path"],
            cfg["use_ca"],
            cfg["sr_kv"],
            fusion_type,
            branch,
            not use_cross_attention,
            in_channels=in_channels,
        )
        self.decode = BidirectionalMultiScaleDecoder(
            list(cfg["dims"]), max(cfg["decoder_dim"] // 2, 64), num_classes
        )
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, LayerNorm2d, nn.BatchNorm2d)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        _, f1, f2, f3, f4 = self.backbone(x)
        logits = self.decode([f1, f2, f3, f4])
        return F.interpolate(
            logits, size=(height, width), mode="bilinear", align_corners=False
        )


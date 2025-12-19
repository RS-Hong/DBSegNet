import math
from typing import List, Tuple, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
DBSetNet (CvT + MobileNetV2 双分支, 结合轻量 Cross-Attn 融合)
---------------------------------------------------------------------
构建：
- 构建 CNN(局部) + Transformer(全局) 的双分支；
- 参考卷积化注意力的有效策略；
- 先提取特征，后在低分辨率 stage 进行分支交互，以提升效率；
- 解码沿用 SegFormer-MLP：不使用辅助损失
  DBSegNet(num_classes, phi, pretrained=False, return_aux=False)

配置：
- 'cvmv_tiny_fast'   → 速度≈ SegFormer-B0：仅 s3 做一次 Cross-Attn，sr=(8,4,2,1)
- 'cvmv_tiny_strong' → s3/s4 都做 Cross-Attn（K/V下采样），精度↑，速度略降
- 'cvmv_small_strong'→ 更宽更深版本（显存允许时）
"""

# --------------------- Utils ---------------------

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rnd = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        rnd.floor_()
        return x.div(keep) * rnd

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act=True, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# --------------------- CNN branch ---------------------

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, drop_path=0.0, se=False):
        super().__init__()
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res = stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(inp, hidden_dim, k=1, s=1, p=0))
        layers.extend([
            ConvBNAct(hidden_dim, hidden_dim, k=3, s=stride, p=1, groups=hidden_dim),
            ConvBNAct(hidden_dim, oup, k=1, s=1, p=0, act=False),
        ])
        self.block = nn.Sequential(*layers)
        self.dp = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    def forward(self, x):
        out = self.block(x)
        return x + self.dp(out) if hasattr(self, 'use_res') and self.use_res else out

class CNNStage(nn.Module):
    def __init__(self, in_ch, out_ch, depth, stride, dprs):
        super().__init__()
        blocks = []
        # 第一个块负责下采样（stride），其余 stride=1
        blocks.append(InvertedResidual(in_ch, out_ch, stride=stride, expand_ratio=4, drop_path=dprs[0]))
        for i in range(1, depth):
            blocks.append(InvertedResidual(out_ch, out_ch, stride=1, expand_ratio=4, drop_path=dprs[i]))
        self.blocks = nn.Sequential(*blocks)
        self.out_norm = LayerNorm2d(out_ch)
    def forward(self, x):
        x = self.blocks(x)
        return self.out_norm(x)

# --------------------- Transformer branch ---------------------

class ConvEmbed(nn.Module):
    """Overlap Patch Embedding via conv (stride controls downsample)."""
    def __init__(self, in_ch, embed_dim, stride):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, 3, stride, 1)
        self.norm = LayerNorm2d(embed_dim)
    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        H, W = x.shape[-2:]
        return x, H, W

class CvTAttention(nn.Module):
    """卷积化自注意力：q,k,v 均由深度可分卷积产生；k/v 支持空间下采样 (sr_ratio)。"""
    def __init__(self, dim, heads=8, sr_ratio=1):
        super().__init__()
        self.h = heads
        self.d = dim // heads
        self.scale = self.d ** -0.5
        # q,k,v 用 1x1 pointwise 之前加一个 depthwise conv 做局部感受野
        self.q_dw = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.kv_dw= nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.q_pw = nn.Conv2d(dim, dim, 1)
        self.kv_pw= nn.Conv2d(dim, dim*2, 1)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.ln = nn.LayerNorm(dim)
    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q_pw(self.q_dw(x))                                     # [B,C,H,W]
        if self.sr_ratio > 1:
            kv_in = self.sr(self.kv_dw(x))                              # 下采样
        else:
            kv_in = self.kv_dw(x)
        kv = self.kv_pw(kv_in)
        # flatten
        q = q.flatten(2).transpose(1, 2).reshape(B, H*W, self.h, self.d).transpose(1, 2)
        kv = kv.flatten(2).transpose(1, 2).reshape(B, -1, 2, self.h, self.d).permute(2,0,3,1,4)
        k, v = kv[0], kv[1]                                             # [B,h,Nk,d]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(-1)
        out = (attn @ v).transpose(1, 2).reshape(B, H*W, C)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return self.proj(out)

class MixFFN(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hid = int(dim*mlp_ratio)
        self.pw1 = nn.Conv2d(dim, hid, 1)
        self.dw  = nn.Conv2d(hid, hid, 3, 1, 1, groups=hid)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hid, dim, 1)
        self.norm= LayerNorm2d(dim)
    def forward(self, x):
        s = x
        x = self.pw1(x)
        x = self.dw(x)
        x = self.act(x)
        x = self.pw2(x)
        return self.norm(x + s)

class CvTBlock(nn.Module):
    def __init__(self, dim, heads, sr_ratio, drop_path=0.0):
        super().__init__()
        self.attn = CvTAttention(dim, heads=heads, sr_ratio=sr_ratio)
        self.ffn  = MixFFN(dim)
        self.gamma1 = nn.Parameter(torch.ones(1, dim, 1, 1) * 1e-6)
        self.gamma2 = nn.Parameter(torch.ones(1, dim, 1, 1) * 1e-6)
        self.dp = DropPath(drop_path) if drop_path>0 else nn.Identity()
        self.norm = LayerNorm2d(dim)
    def forward(self, x):
        x = x + self.dp(self.gamma1 * self.attn(self.norm(x)))
        x = x + self.dp(self.gamma2 * self.ffn(self.norm(x)))
        return x

class TRStage(nn.Module):
    def __init__(self, in_ch, dim, depth, heads, sr_ratio, stride, dprs):
        super().__init__()
        self.patch = ConvEmbed(in_ch, dim, stride)
        self.blocks = nn.Sequential(*[CvTBlock(dim, heads, sr_ratio, dprs[i]) for i in range(depth)])
        self.out_norm = LayerNorm2d(dim)
    def forward(self, x):
        x, H, W = self.patch(x)
        x = self.blocks(x)
        return self.out_norm(x)

# --------------------- Cross-Attention + Light EX + Gate ---------------------

class LightEX(nn.Module):
    """轻量化特征 互投影 + SE 通道门控，几乎无额外算力。"""
    def __init__(self, c_ch, t_ch, out_ch):
        super().__init__()
        self.c2t = nn.Conv2d(c_ch, out_ch, 1, bias=False)
        self.t2c = nn.Conv2d(t_ch, out_ch, 1, bias=False)
        self.sec1 = nn.Conv2d(out_ch, out_ch//4, 1)
        self.sec2 = nn.Conv2d(out_ch//4, out_ch, 1)
        self.set1 = nn.Conv2d(out_ch, out_ch//4, 1)
        self.set2 = nn.Conv2d(out_ch//4, out_ch, 1)
        self.sig = nn.Sigmoid()
    def forward(self, c, t):
        c2t = self.c2t(c)
        t2c = self.t2c(t)
        wc = self.sig(self.sec2(F.silu(self.sec1(t2c))))
        wt = self.sig(self.set2(F.silu(self.set1(c2t))))
        c = c + t2c * wc
        t = t + c2t * wt
        return c, t

class CrossAttnLite(nn.Module):
    def __init__(self, ch, heads=4, sr_kv=2):
        super().__init__()
        self.h = max(1, heads)
        self.d = ch // self.h
        self.scale = self.d ** -0.5
        self.q = nn.Linear(ch, ch)
        self.kv= nn.Linear(ch, ch*2)
        self.out = nn.Linear(ch, ch)
        self.sr_kv = sr_kv
        if sr_kv > 1:
            self.sr = nn.Conv2d(ch, ch, sr_kv, sr_kv)
            self.norm = nn.LayerNorm(ch)
    def forward(self, q_in, kv_in):
        B, C, Hq, Wq = q_in.shape
        Nq = Hq * Wq
        q = q_in.flatten(2).transpose(1,2)
        if self.sr_kv > 1:
            kvd = self.sr(kv_in)
            kv = kvd.flatten(2).transpose(1,2)
            kv = self.norm(kv)
        else:
            kv = kv_in.flatten(2).transpose(1,2)
        q = self.q(q).reshape(B, Nq, self.h, self.d).transpose(1,2)
        kv = self.kv(kv).reshape(B, -1, 2, self.h, self.d).permute(2,0,3,1,4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(-1)
        out = (attn @ v).transpose(1,2).reshape(B, Nq, self.h*self.d)
        out = self.out(out).transpose(1,2).reshape(B, -1, Hq, Wq)
        return out

class GatedFusion(nn.Module):
    def __init__(self, c_ch, t_ch, out_ch):
        super().__init__()
        self.c_proj = ConvBNAct(c_ch, out_ch, 1,1,0, act=False)
        self.t_proj = ConvBNAct(t_ch, out_ch, 1,1,0, act=False)
        self.g = nn.Conv2d(out_ch*2, out_ch, 1)
        self.bias = nn.Parameter(torch.ones(1, out_ch, 1, 1) * (-2.0))
        self.tau  = nn.Parameter(torch.tensor(2.0))
        self.bn   = nn.BatchNorm2d(out_ch)
    def forward(self, c, t):
        c_ = self.c_proj(c); t_ = self.t_proj(t)
        g = self.g(torch.cat([c_, t_], dim=1)) + self.bias
        g = torch.sigmoid(g / torch.clamp(self.tau, min=0.1))
        out = g * t_ + (1 - g) * c_
        return self.bn(out)

# --------------------- Decoder(SegFormer-MLP) ---------------------

class SegFormerDecoderHead(nn.Module):
    def __init__(self, in_channels: List[int], decoder_dim: int, num_classes: int):
        super().__init__()
        self.proj = nn.ModuleList([ConvBNAct(c, decoder_dim, 1,1,0) for c in in_channels])
        self.fuse = ConvBNAct(decoder_dim*4, decoder_dim, 3,1,1)
        self.drop = nn.Dropout(0.1)
        self.cls  = nn.Conv2d(decoder_dim, num_classes, 1)
    def forward(self, feats: List[torch.Tensor]):
        B,_,H4,W4 = feats[0].shape
        outs = []
        for i,x in enumerate(feats):
            x = self.proj[i](x)
            if x.shape[-2:] != (H4,W4):
                x = F.interpolate(x, size=(H4,W4), mode='bilinear', align_corners=False)
            outs.append(x)
        x = torch.cat(outs, dim=1)
        x = self.fuse(x)
        x = self.drop(x)
        return self.cls(x)

class AuxHead(nn.Module):
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.conv = ConvBNAct(in_ch, in_ch, 3,1,1)
        self.cls  = nn.Conv2d(in_ch, num_classes, 1)
    def forward(self, x, size_hw: Tuple[int,int]):
        x = self.conv(x)
        x = self.cls(x)
        if x.shape[-2:] != size_hw:
            x = F.interpolate(x, size=size_hw, mode='bilinear', align_corners=False)
        return x

# --------------------- Backbone (stem -> dual -> fusion) ---------------------

PHI = {
    'cvmv_tiny_fast': {
        'dims': (64, 128, 256, 512),
        'depths_cnn': (1, 2, 2, 2),
        'depths_tr' : (1, 2, 2, 2),
        'heads'     : (1, 2, 4, 8),
        'sr'        : (8, 4, 2, 1),
        'decoder_dim': 256,
        'drop_path' : 0.1,
        'use_ca'    : (False, False, True,  False),
        'sr_kv'     : (0, 0, 2, 0),
    },
    'cvmv_tiny_strong': {
        'dims': (64, 128, 256, 512),
        'depths_cnn': (1, 2, 2, 2),
        'depths_tr' : (1, 2, 2, 2),
        'heads'     : (1, 2, 4, 8),
        'sr'        : (8, 4, 2, 1),
        'decoder_dim': 320,
        'drop_path' : 0.15,
        'use_ca'    : (False, False, True,  True),
        'sr_kv'     : (0, 0, 2, 2),
    },
    'cvmv_small_strong': {
        'dims': (64, 128, 320, 512),
        'depths_cnn': (2, 2, 3, 2),
        'depths_tr' : (2, 2, 3, 2),
        'heads'     : (1, 2, 5, 8),
        'sr'        : (8, 4, 2, 1),
        'decoder_dim': 384,
        'drop_path' : 0.2,
        'use_ca'    : (False, False, True,  True),
        'sr_kv'     : (0, 0, 2, 2),
    },
}

class DualEncoder(nn.Module):
    def __init__(self, dims, depths_cnn, depths_tr, heads, sr, drop_path, use_ca, sr_kv):
        super().__init__()
        # Stem：1/2 -> 1/4（两层 3×3,s2）并预留 Aux 点
        self.stem = nn.Sequential(ConvBNAct(3, 32, 3,2,1), ConvBNAct(32, 64, 3,2,1))

        total = sum(depths_cnn) + sum(depths_tr)
        dprs = [x.item() for x in torch.linspace(0, drop_path, total)]
        it = iter(dprs)
        def take(n):
            return [next(it, 0.0) for _ in range(n)]

        # s1 1/4
        self.c1 = CNNStage(64, dims[0], depths_cnn[0], stride=1, dprs=take(depths_cnn[0]))
        self.t1 = TRStage(64, dims[0], depths_tr[0], heads[0], sr[0], stride=1, dprs=take(depths_tr[0]))
        self.lex1= LightEX(dims[0], dims[0], dims[0])
        self.ca1 = None
        self.fu1 = GatedFusion(dims[0], dims[0], dims[0])

        # s2 1/8
        self.c2 = CNNStage(dims[0], dims[1], depths_cnn[1], stride=2, dprs=take(depths_cnn[1]))
        self.t2 = TRStage(dims[0], dims[1], depths_tr[1], heads[1], sr[1], stride=2, dprs=take(depths_tr[1]))
        self.lex2= LightEX(dims[1], dims[1], dims[1])
        self.ca2 = None
        self.fu2 = GatedFusion(dims[1], dims[1], dims[1])

        # s3 1/16
        self.c3 = CNNStage(dims[1], dims[2], depths_cnn[2], stride=2, dprs=take(depths_cnn[2]))
        self.t3 = TRStage(dims[1], dims[2], depths_tr[2], heads[2], sr[2], stride=2, dprs=take(depths_tr[2]))
        self.lex3= LightEX(dims[2], dims[2], dims[2])
        self.ca3 = CrossAttnLite(dims[2], heads=max(1, heads[2]), sr_kv=max(1, sr_kv[2])) if use_ca[2] else None
        self.fu3 = GatedFusion(dims[2], dims[2], dims[2])

        # s4 1/32
        self.c4 = CNNStage(dims[2], dims[3], depths_cnn[3], stride=2, dprs=take(depths_cnn[3]))
        self.t4 = TRStage(dims[2], dims[3], depths_tr[3], heads[3], sr[3], stride=2, dprs=take(depths_tr[3]))
        self.lex4= LightEX(dims[3], dims[3], dims[3])
        self.ca4 = CrossAttnLite(dims[3], heads=max(1, heads[3]), sr_kv=max(1, sr_kv[3])) if use_ca[3] else None
        self.fu4 = GatedFusion(dims[3], dims[3], dims[3])

    def forward(self, x):
        x = self.stem(x)
        f0 = x
        c1 = self.c1(x); t1 = self.t1(x); c1, t1 = self.lex1(c1, t1); f1 = self.fu1(c1, t1)
        c2 = self.c2(f1); t2 = self.t2(t1); c2, t2 = self.lex2(c2, t2); f2 = self.fu2(c2, t2)
        c3 = self.c3(f2); t3 = self.t3(t2); c3, t3 = self.lex3(c3, t3);
        if self.ca3 is not None:
            t3 = t3 + self.ca3(t3, c3)
        f3 = self.fu3(c3, t3)
        c4 = self.c4(f3); t4 = self.t4(t3); c4, t4 = self.lex4(c4, t4)
        if self.ca4 is not None:
            t4 = t4 + self.ca4(t4, c4)
        f4 = self.fu4(c4, t4)
        return f0, f1, f2, f3, f4

# --------------------- Net (SegFormer-compatible) ---------------------

class DBSegNet(nn.Module):
    def __init__(self, num_classes: int = 21, phi: str = 'cvmv_tiny_fast', pretrained: bool = False, return_aux: bool = False):
        super().__init__()
        assert phi in PHI, f"Unsupported phi: {phi}. Options: {list(PHI.keys())}"
        cfg = PHI[phi]
        self.return_aux = return_aux

        self.backbone = DualEncoder(cfg['dims'], cfg['depths_cnn'], cfg['depths_tr'], cfg['heads'], cfg['sr'], cfg['drop_path'], cfg['use_ca'], cfg['sr_kv'])
        self.decode = SegFormerDecoderHead(list(cfg['dims']), cfg['decoder_dim'], num_classes)
        if self.return_aux:
            self.aux0 = AuxHead(64, num_classes)
            self.aux3 = AuxHead(cfg['dims'][2], num_classes)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, LayerNorm2d, nn.BatchNorm2d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        H, W = x.shape[-2:]
        f0, f1, f2, f3, f4 = self.backbone(x)
        out_1_4 = self.decode([f1, f2, f3, f4])
        out = F.interpolate(out_1_4, size=(H, W), mode='bilinear', align_corners=False)
        if not self.return_aux:
            return out
        return {"out": out, "aux0": self.aux0(f0, (H, W)), "aux3": self.aux3(f3, (H, W))}


if __name__ == '__main__':
    m = DBSegNet(num_classes=2, phi='cvmv_tiny_fast', return_aux=False)
    x = torch.randn(1,3,512,512)
    y = m(x)
    print(y.shape)

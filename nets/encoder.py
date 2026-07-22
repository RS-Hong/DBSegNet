"""CNN-Transformer dual-branch encoder used by DBSegNet."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
DBSetNet 改进版v2 (CvT + MobileNetV2 双分支, 增强双向Cross-Attn + 聚合融合 + Transformer warmup)
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
        return x + self.dp(out) if self.use_res else out

class CNNStage(nn.Module):
    def __init__(self, in_ch, out_ch, depth, stride, dprs):
        super().__init__()
        blocks = []
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
    def __init__(self, dim, heads=8, sr_ratio=1):
        super().__init__()
        self.h = heads
        self.d = dim // heads
        self.scale = self.d ** -0.5
        self.q_dw = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.kv_dw = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.q_pw = nn.Conv2d(dim, dim, 1)
        self.kv_pw = nn.Conv2d(dim, dim*2, 1)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.ln = nn.LayerNorm(dim)
    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q_pw(self.q_dw(x))
        if self.sr_ratio > 1:
            kv_in = self.sr(self.kv_dw(x))
        else:
            kv_in = self.kv_dw(x)
        kv = self.kv_pw(kv_in)
        q = q.flatten(2).transpose(1, 2).reshape(B, H*W, self.h, self.d).transpose(1, 2)
        kv = kv.flatten(2).transpose(1, 2).reshape(B, -1, 2, self.h, self.d).permute(2,0,3,1,4)
        k, v = kv[0], kv[1]
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
        self.norm = LayerNorm2d(dim)
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
        self.dp = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm = LayerNorm2d(dim)
    def forward(self, x):
        x = x + self.dp(self.gamma1 * self.attn(self.norm(x)))
        x = x + self.dp(self.gamma2 * self.ffn(self.norm(x)))
        return x

class TRStage(nn.Module):
    def __init__(self, in_ch, dim, depth, heads, sr_ratio, stride, dprs, warmup=False):
        super().__init__()
        self.patch = ConvEmbed(in_ch, dim, stride)
        self.blocks = nn.Sequential(*[CvTBlock(dim, heads, sr_ratio, dprs[i]) for i in range(depth)])
        self.out_norm = LayerNorm2d(dim)
        self.warmup = warmup
        self.skip_proj = nn.Conv2d(in_ch, dim, 1, bias=False)  # 新增：投影 cnn_skip 通道

    def forward(self, x, cnn_skip=None):
        x, H, W = self.patch(x)
        if cnn_skip is not None:
            cnn_skip = F.interpolate(cnn_skip, size=(H, W), mode='bilinear', align_corners=False)
            cnn_skip = self.skip_proj(cnn_skip)  # 投影通道匹配
            x = x + 0.1 * cnn_skip
        x = self.blocks(x)
        if self.warmup:
            x = x * 0.5
        return self.out_norm(x)

# --------------------- Fusion Modules ---------------------

class SELightEX(nn.Module):
    def __init__(self, c_ch, t_ch, out_ch):
        super().__init__()
        self.c2t = nn.Conv2d(c_ch, out_ch, 1, bias=False)
        self.t2c = nn.Conv2d(t_ch, out_ch, 1, bias=False)
        self.se_c = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch//4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch//4, out_ch, 1, bias=False),
            nn.Sigmoid()
        )
        self.se_t = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch//4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch//4, out_ch, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, c, t):
        c2t = self.c2t(c)
        t2c = self.t2c(t)
        wc = self.se_t(t2c)
        wt = self.se_c(c2t)
        c = c + t2c * wc
        t = t + c2t * wt
        return c, t

class GatedFusion(nn.Module):
    def __init__(self, c_ch, t_ch, out_ch, fusion_type='gate'):
        super().__init__()
        self.fusion_type = fusion_type
        self.c_proj = ConvBNAct(c_ch, out_ch, 1, 1, 0, act=False)
        self.t_proj = ConvBNAct(t_ch, out_ch, 1, 1, 0, act=False)
        if fusion_type in ['gate', 'gated']:
            self.g = nn.Sequential(
                nn.Conv2d(out_ch*2, out_ch//4, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch//4, out_ch, 1, bias=False),
                nn.Sigmoid()
            )
            self.bias = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
            self.tau = nn.Parameter(torch.tensor(2.0))
        elif fusion_type == 'agg':
            self.ch_attn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_ch*2, out_ch//4, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch//4, out_ch*2, 1, bias=False),
                nn.Sigmoid()
            )
            self.sp_attn = nn.Conv2d(out_ch*2, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, c, t):
        c_ = self.c_proj(c)
        t_ = self.t_proj(t)
        if self.fusion_type == 'sum':
            out = c_ + t_
        elif self.fusion_type in ['gate', 'gated']:
            g = self.g(torch.cat([c_, t_], dim=1)) + self.bias
            g = torch.sigmoid(g / torch.clamp(self.tau, min=0.1))
            out = g * t_ + (1 - g) * c_
        elif self.fusion_type == 'agg':
            cat = torch.cat([c_, t_], dim=1)
            ch_w = self.ch_attn(cat).chunk(2, dim=1)
            sp_mask = torch.sigmoid(self.sp_attn(cat))
            out = (c_ * ch_w[0] * sp_mask) + (t_ * ch_w[1] * (1 - sp_mask))
        else:
            raise ValueError(f"Unsupported fusion_type: {self.fusion_type}")
        return self.bn(out)

    def forward_c_only(self, c):
        return self.bn(self.c_proj(c))

class CrossAttnLite(nn.Module):
    def __init__(self, ch, heads=4, sr_kv=2):
        super().__init__()
        self.h = max(1, heads)
        self.d = ch // self.h
        self.scale = self.d ** -0.5
        self.q = nn.Linear(ch, ch)
        self.kv = nn.Linear(ch, ch*2)
        self.out = nn.Linear(ch, ch)
        self.sr_kv = sr_kv
        if sr_kv > 1:
            self.sr = nn.Conv2d(ch, ch, sr_kv, sr_kv)
            self.norm = nn.LayerNorm(ch)
    def forward(self, q_in, kv_in):
        B, C, Hq, Wq = q_in.shape
        Nq = Hq * Wq
        q = q_in.flatten(2).transpose(1, 2)
        if self.sr_kv > 1:
            kvd = self.sr(kv_in)
            kv = kvd.flatten(2).transpose(1, 2)
            kv = self.norm(kv)
        else:
            kv = kv_in.flatten(2).transpose(1, 2)
        q = self.q(q).reshape(B, Nq, self.h, self.d).transpose(1, 2)
        kv = self.kv(kv).reshape(B, -1, 2, self.h, self.d).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(-1)
        out = (attn @ v).transpose(1, 2).reshape(B, Nq, self.h*self.d)
        out = self.out(out).transpose(1, 2).reshape(B, -1, Hq, Wq)
        return out

# --------------------- Backbone ---------------------

PHI = {
    'cvmv_tiny_fast': {
        'dims': (64, 128, 256, 512),
        'depths_cnn': (1, 2, 2, 2),
        'depths_tr': (1, 2, 2, 2),
        'heads': (1, 2, 4, 8),
        'sr': (8, 4, 2, 1),
        'decoder_dim': 256,
        'drop_path': 0.1,
        'use_ca': (False, False, True, False),
        'sr_kv': (0, 0, 2, 0),
    },
}

class DualEncoder(nn.Module):
    def __init__(self, dims, depths_cnn, depths_tr, heads, sr, drop_path, use_ca, sr_kv, fusion_mode='gate', branch='dual', force_no_ca=False, in_channels=3):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.branch = branch
        self.force_no_ca = force_no_ca
        self.zero_transformer = False
        self.stem = nn.Sequential(ConvBNAct(in_channels, 32, 3, 2, 1), ConvBNAct(32, 64, 3, 2, 1))

        total = sum(depths_cnn) + sum(depths_tr)
        dprs = [x.item() for x in torch.linspace(0, drop_path, total)]
        it = iter(dprs)
        def take(n): return [next(it, 0.0) for _ in range(n)]

        self.c1 = CNNStage(64, dims[0], depths_cnn[0], 1, take(depths_cnn[0]))
        self.t1 = TRStage(64, dims[0], depths_tr[0], heads[0], sr[0], 1, take(depths_tr[0]), warmup=True)
        self.lex1 = SELightEX(dims[0], dims[0], dims[0])
        self.ca1 = None
        self.fu1 = GatedFusion(dims[0], dims[0], dims[0], fusion_mode)

        self.c2 = CNNStage(dims[0], dims[1], depths_cnn[1], 2, take(depths_cnn[1]))
        self.t2 = TRStage(dims[0], dims[1], depths_tr[1], heads[1], sr[1], 2, take(depths_tr[1]), warmup=True)
        self.lex2 = SELightEX(dims[1], dims[1], dims[1])
        self.ca2 = CrossAttnLite(dims[1], heads[1], sr_kv[1]) if use_ca[1] and branch != 'cnn' else None
        self.ca2_cnn = CrossAttnLite(dims[1], heads[1], sr_kv[1]) if use_ca[1] and branch == 'dual' else None
        self.fu2 = GatedFusion(dims[1], dims[1], dims[1], fusion_mode)

        self.c3 = CNNStage(dims[1], dims[2], depths_cnn[2], 2, take(depths_cnn[2]))
        self.t3 = TRStage(dims[1], dims[2], depths_tr[2], heads[2], sr[2], 2, take(depths_tr[2]), warmup=True)
        self.lex3 = SELightEX(dims[2], dims[2], dims[2])
        self.ca3 = CrossAttnLite(dims[2], heads[2], sr_kv[2]) if use_ca[2] and branch != 'cnn' else None
        self.ca3_cnn = CrossAttnLite(dims[2], heads[2], sr_kv[2]) if use_ca[2] and branch == 'dual' else None
        self.fu3 = GatedFusion(dims[2], dims[2], dims[2], fusion_mode)

        self.c4 = CNNStage(dims[2], dims[3], depths_cnn[3], 2, take(depths_cnn[3]))
        self.t4 = TRStage(dims[2], dims[3], depths_tr[3], heads[3], sr[3], 2, take(depths_tr[3]), warmup=True)
        self.lex4 = SELightEX(dims[3], dims[3], dims[3])
        self.ca4 = CrossAttnLite(dims[3], heads[3], sr_kv[3]) if use_ca[3] and branch != 'cnn' else None
        self.ca4_cnn = CrossAttnLite(dims[3], heads[3], sr_kv[3]) if use_ca[3] and branch == 'dual' else None
        self.fu4 = GatedFusion(dims[3], dims[3], dims[3], fusion_mode)

    def set_zero_transformer(self, enabled: bool):
        self.zero_transformer = bool(enabled)

    def forward(self, x):
        x = self.stem(x)
        f0 = x

        if self.branch == 'cnn':
            f1 = self.c1(x)
            f2 = self.c2(f1)
            f3 = self.c3(f2)
            f4 = self.c4(f3)
            return f0, f1, f2, f3, f4

        # A true Transformer-only ablation must bypass cross-branch exchange
        # and gated fusion, just as the CNN-only path does above.  Keeping the
        # zero-filled CNN tensor in the dual path would still reconstruct a
        # non-zero "CNN" feature through LightEX and would therefore not be a
        # clean single-branch control.
        if self.branch == 'tr':
            f1 = self.t1(x)
            f2 = self.t2(f1)
            f3 = self.t3(f2)
            f4 = self.t4(f3)
            return f0, f1, f2, f3, f4

        if self.branch == 'dual' and self.zero_transformer and not self.training:
            c1 = self.c1(x)
            f1 = self.fu1.forward_c_only(c1)
            c2 = self.c2(f1)
            f2 = self.fu2.forward_c_only(c2)
            c3 = self.c3(f2)
            f3 = self.fu3.forward_c_only(c3)
            c4 = self.c4(f3)
            f4 = self.fu4.forward_c_only(c4)
            return f0, f1, f2, f3, f4

        t1 = self.t1(x)
        c1 = self.c1(x) if self.branch == 'dual' else torch.zeros_like(t1)
        c1, t1 = self.lex1(c1, t1)
        f1 = self.fu1(c1, t1)

        t2 = self.t2(f1, cnn_skip=c1 if self.branch == 'dual' else None)
        c2 = self.c2(f1) if self.branch == 'dual' else torch.zeros_like(t2)
        if self.ca2 is not None and not self.force_no_ca:
            t2 = t2 + self.ca2(t2, c2)
        if self.ca2_cnn is not None and not self.force_no_ca:
            c2 = c2 + self.ca2_cnn(c2, t2)
        c2, t2 = self.lex2(c2, t2)
        f2 = self.fu2(c2, t2)

        t3 = self.t3(f2, cnn_skip=c2 if self.branch == 'dual' else None)
        c3 = self.c3(f2) if self.branch == 'dual' else torch.zeros_like(t3)
        if self.ca3 is not None and not self.force_no_ca:
            t3 = t3 + self.ca3(t3, c3)
        if self.ca3_cnn is not None and not self.force_no_ca:
            c3 = c3 + self.ca3_cnn(c3, t3)
        c3, t3 = self.lex3(c3, t3)
        f3 = self.fu3(c3, t3)

        t4 = self.t4(f3, cnn_skip=c3 if self.branch == 'dual' else None)
        c4 = self.c4(f3) if self.branch == 'dual' else torch.zeros_like(t4)
        if self.ca4 is not None and not self.force_no_ca:
            t4 = t4 + self.ca4(t4, c4)
        if self.ca4_cnn is not None and not self.force_no_ca:
            c4 = c4 + self.ca4_cnn(c4, t4)
        c4, t4 = self.lex4(c4, t4)
        f4 = self.fu4(c4, t4)

        return f0, f1, f2, f3, f4

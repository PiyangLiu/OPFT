import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_2d_sincos_pos_embed(embed_dim, grid_size, grid_size2=None, cls_token=False):
    if grid_size2 is None:
        grid_size2 = grid_size

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size2, dtype=np.float32)

    grid = np.meshgrid(grid_h, grid_w, indexing="ij")
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size2])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(self, dim, pt_seq_len, num_cls_token=0):
        super().__init__()
        self.dim = dim
        self.pt_seq_len = pt_seq_len
        self.num_cls_token = num_cls_token

        self.register_buffer("cos_cached", torch.empty(0))
        self.register_buffer("sin_cached", torch.empty(0))

    def _compute_cos_sin(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=dtype)
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, self.dim, 2, device=device, dtype=dtype) / self.dim)
        )
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        freqs = freqs.repeat_interleave(2, dim=-1)
        return torch.cos(freqs), torch.sin(freqs)

    def forward(self, x):
        B, num_heads, seq_len, head_dim = x.shape

        if (
            self.cos_cached.shape[0] == 0
            or self.cos_cached.shape[2] < seq_len
            or self.cos_cached.device != x.device
            or self.cos_cached.dtype != x.dtype
        ):
            cos, sin = self._compute_cos_sin(
                max(seq_len, self.pt_seq_len), x.device, x.dtype
            )

            cos = cos[None, None, :, :]
            sin = sin[None, None, :, :]

            self.cos_cached = cos
            self.sin_cached = sin

        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]

        return self.apply_rotary_pos_emb(x, cos, sin)

    @staticmethod
    def apply_rotary_pos_emb(t, cos, sin):
        t_half = t.shape[-1] // 2
        t1, t2 = t[..., :t_half], t[..., t_half:]
        cos1, sin1 = cos[..., :t_half], sin[..., :t_half]

        rotated = torch.cat([t1 * cos1 - t2 * sin1, t1 * sin1 + t2 * cos1], dim=-1)

        return rotated


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        x = x / norm.to(x.dtype)
        return x * self.weight


class OverlapPatchEmbed(nn.Module):

    def __init__(
        self,
        img_size=(60, 60),
        patch_size=(12, 12),
        stride=(6, 6),
        in_chans=1,
        embed_dim=512,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        H, W = img_size
        ph, pw = patch_size
        sh, sw = stride

        self.grid_h = (H - ph) // sh + 1
        self.grid_w = (W - pw) // sw + 1
        self.num_patches = self.grid_h * self.grid_w

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride
        )

        print(f"重叠Patch嵌入配置:")
        print(f"  输入尺寸: {img_size}")
        print(f"  Patch大小: {patch_size}")
        print(f"  步长: {stride} ({(1 - stride[0] / patch_size[0]) * 100:.1f}% 重叠)")
        print(f"  Patch网格: {self.grid_h}×{self.grid_w} = {self.num_patches}个patch")

    def forward(self, x):

        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"输入尺寸({H},{W})与预期({self.img_size[0]},{self.img_size[1]})不匹配"

        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ObservationEmbedder(nn.Module):
    def __init__(self, obs_dim=500, hidden_size=512):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(obs_dim, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, obs):
        return self.embedding(obs.float())


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = rope(q)
        k = rope(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.w12 = nn.Linear(dim, hidden_dim * 2)
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class JiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads, attn_drop, proj_drop)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, mlp_ratio)

        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c, rope):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(
            c
        ).chunk(6, dim=-1)

        x = x + gate_msa.unsqueeze(1) * self.attn(
            self.norm1(x * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)), rope
        )

        x = x + gate_mlp.unsqueeze(1) * self.ffn(
            self.norm2(x * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1))
        )

        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, patch_size=(15, 15), out_chans=3):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.proj = nn.Linear(dim, patch_size[0] * patch_size[1] * out_chans)
        self.out_chans = out_chans
        self.patch_size = patch_size

    def forward(self, x, c):
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        x = self.norm(x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))
        x = self.proj(x)
        return x


class ShiftedConvFusion(nn.Module):

    def __init__(
        self,
        img_size=(60, 60),
        patch_size=(15, 15),
        stride=(5, 5),
        in_chans=3,
        fusion_channels=64,
        use_learnable_weights=True,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.in_chans = in_chans
        self.fusion_channels = fusion_channels
        self.use_learnable_weights = use_learnable_weights

        H, W = img_size
        ph, pw = patch_size
        sh, sw = stride

        self.grid_h = (H - ph) // sh + 1
        self.grid_w = (W - pw) // sw + 1
        self.num_patches = self.grid_h * self.grid_w

        if use_learnable_weights:
            self.weight_param = nn.Parameter(torch.ones(1, 1, ph, pw))
            nn.init.normal_(self.weight_param, mean=1.0, std=0.1)

        self.fusion_net = self._build_fusion_net()

        self.edge_enhance = nn.Sequential(
            nn.Conv2d(in_chans, fusion_channels // 2, 3, padding=1),
            nn.GroupNorm(8, fusion_channels // 2),
            nn.SiLU(),
            nn.Conv2d(fusion_channels // 2, in_chans, 3, padding=1),
        )

        self._init_weights()

        print(f"移位卷积融合配置:")
        print(f"  输入尺寸: {img_size}")
        print(f"  Patch大小: {patch_size}")
        print(f"  步长: {stride}")
        print(f"  Patch网格: {self.grid_h}×{self.grid_w} = {self.num_patches}个patch")
        print(f"  可学习权重: {use_learnable_weights}")

    def _build_fusion_net(self):

        return nn.Sequential(
            nn.Conv2d(self.in_chans, self.fusion_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, self.fusion_channels), self.fusion_channels),
            nn.SiLU(),
            nn.Conv2d(
                self.fusion_channels, self.fusion_channels, kernel_size=3, padding=1
            ),
            nn.GroupNorm(min(8, self.fusion_channels), self.fusion_channels),
            nn.SiLU(),
            nn.Conv2d(
                self.fusion_channels,
                self.fusion_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
            ),
            nn.GroupNorm(min(8, self.fusion_channels), self.fusion_channels),
            nn.SiLU(),
            nn.Conv2d(self.fusion_channels, self.fusion_channels // 2, kernel_size=1),
            nn.GroupNorm(min(8, self.fusion_channels // 2), self.fusion_channels // 2),
            nn.SiLU(),
            nn.Conv2d(
                self.fusion_channels // 2, self.in_chans, kernel_size=3, padding=1
            ),
        )

    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _create_gaussian_weight(self, ph, pw, device="cpu", sigma=0.5):

        center_h, center_w = ph / 2 - 0.5, pw / 2 - 0.5

        h = torch.arange(ph, device=device).float()
        w = torch.arange(pw, device=device).float()
        h, w = torch.meshgrid(h, w, indexing="ij")

        weight = torch.exp(
            -((h - center_h) ** 2 + (w - center_w) ** 2) / (2 * sigma**2)
        )
        weight = weight / weight.max()

        return weight.unsqueeze(0).unsqueeze(0)

    def _reconstruct_with_weights(self, patches, weights):

        B, N, C, ph, pw = patches.shape
        H, W = self.img_size
        sh, sw = self.stride

        output = torch.zeros(B, C, H, W, device=patches.device)
        weight_accum = torch.zeros(1, 1, H, W, device=patches.device)

        idx = 0
        for i in range(self.grid_h):
            for j in range(self.grid_w):
                h_start = i * sh
                w_start = j * sw
                h_end = h_start + ph
                w_end = w_start + pw

                patch_weight = weights.expand(B, C, -1, -1)

                output[:, :, h_start:h_end, w_start:w_end] += (
                    patches[:, idx] * patch_weight
                )
                weight_accum[:, :, h_start:h_end, w_start:w_end] += patch_weight[
                    0:1, 0:1
                ]

                idx += 1

        weight_accum = torch.clamp(weight_accum, min=1e-8)
        output = output / weight_accum

        return output

    def forward(self, patches, apply_fusion=True):

        B, N, C, ph, pw = patches.shape

        if self.use_learnable_weights:

            weights = torch.sigmoid(self.weight_param)
        else:

            weights = self._create_gaussian_weight(ph, pw, patches.device)

        base_recon = self._reconstruct_with_weights(patches, weights)

        if not apply_fusion:

            return base_recon

        features = self.fusion_net(base_recon)

        refined = base_recon + 0.3 * features
        output = refined

        return output


class JiTWithShiftedConv(nn.Module):
    def __init__(
        self,
        img_size=(60, 60),
        patch_size=(20, 20),
        stride=(10, 10),
        in_chans=1,
        embed_dim=512,
        depth=4,
        num_heads=4,
        mlp_ratio=3.0,
        obs_dim=500,
        attn_drop=0.3,
        proj_drop=0.3,
        fusion_channels=32,
        use_learnable_weights=True,
    ):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = in_chans
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.obs_dim = obs_dim
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.original_img_size = img_size

        self.patch_embed = OverlapPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            stride=stride,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.t_embedder = TimestepEmbedder(embed_dim)
        self.obs_embedder = ObservationEmbedder(obs_dim, embed_dim)

        self.grid_h = self.patch_embed.grid_h
        self.grid_w = self.patch_embed.grid_w
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim), requires_grad=False
        )
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim, self.grid_h, self.grid_w, cls_token=False
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        head_dim = embed_dim // num_heads
        half_head_dim = head_dim // 2
        self.rope = VisionRotaryEmbeddingFast(half_head_dim, num_patches)

        self.blocks = nn.ModuleList(
            [
                JiTBlock(embed_dim, num_heads, mlp_ratio, attn_drop, proj_drop)
                for _ in range(depth)
            ]
        )

        self.final_layer = FinalLayer(embed_dim, patch_size, in_chans)

        self.fusion_layer = ShiftedConvFusion(
            img_size=img_size,
            patch_size=patch_size,
            stride=stride,
            in_chans=in_chans,
            fusion_channels=fusion_channels,
            use_learnable_weights=use_learnable_weights,
        )

        self.apply(self._init_weights)

        print(f"\nJiT with ShiftedConv Fusion 模型配置:")
        print(f"  输入尺寸: {img_size}")
        print(f"  Patch大小: {patch_size}")
        print(f"  重叠步长: {stride}")
        print(f"  嵌入维度: {embed_dim}")
        print(f"  Transformer深度: {depth}")
        print(f"  注意力头数: {num_heads}")
        print(f"  Patch数量: {num_patches}")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, t, obs, apply_fusion=True):

        B, C, H, W = x.shape
        assert (
            H == self.original_img_size[0] and W == self.original_img_size[1]
        ), f"输入图像尺寸应为{self.original_img_size}，但得到({H}, {W})"

        x_patches = self.patch_embed(x)
        x_patches = x_patches + self.pos_embed

        t_emb = self.t_embedder(t)
        obs_emb = self.obs_embedder(obs)
        c_emb = t_emb + obs_emb

        for block in self.blocks:
            x_patches = block(x_patches, c_emb, self.rope)

        x_out = self.final_layer(x_patches, c_emb)

        B, N, D = x_out.shape
        patch_h, patch_w = self.patch_size
        out_chans = self.out_chans

        patch_pixels = patch_h * patch_w
        expected_dim = out_chans * patch_pixels
        assert D == expected_dim, f"输出维度{D}与期望的{expected_dim}不匹配"
        assert (
            N == self.patch_embed.num_patches
        ), f"Patch数量{N}与预期{self.patch_embed.num_patches}不匹配"

        patches = x_out.view(B, N, out_chans, patch_h, patch_w)

        recon_img = self.fusion_layer(patches, apply_fusion=apply_fusion)

        return recon_img


JiT_MODELS = {
    "jit_small": lambda **kwargs: JiTWithShiftedConv(
        depth=4, num_heads=4, embed_dim=512, fusion_channels=32, **kwargs
    ),
    "jit_shifted_conv_medium": lambda **kwargs: JiTWithShiftedConv(
        depth=12, num_heads=8, embed_dim=768, fusion_channels=96, **kwargs
    ),
    "jit_shifted_conv_large": lambda **kwargs: JiTWithShiftedConv(
        depth=24, num_heads=16, embed_dim=1024, fusion_channels=128, **kwargs
    ),
}

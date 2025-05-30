import math
import argparse
import numpy as np

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from .attention import MultiHeadAttention
from .utils import shift_dim
from einops import rearrange

class VQVAE(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding_dim = args.embedding_dim
        self.n_codes = args.n_codes
        
        # Store input shape for later use
        self.input_shape = (args.sequence_length, args.resolution, args.resolution)
        
        self.encoder = TransformerEncoder(args.n_hiddens, args.n_res_layers, args.downsample)
        self.decoder = TransformerDecoder(args.n_hiddens, args.n_res_layers, args.downsample)

        self.pre_vq_conv = SamePadConv3d(args.n_hiddens, args.embedding_dim, 1)
        self.post_vq_conv = SamePadConv3d(args.embedding_dim, args.n_hiddens, 1)

        self.codebook = Codebook(args.n_codes, args.embedding_dim)
        self.save_hyperparameters()

    @property
    def latent_shape(self):
        input_shape = (self.args.sequence_length, self.args.resolution,
                       self.args.resolution)
        return tuple([s // d for s, d in zip(input_shape,
                                             self.args.downsample)])

    def encode(self, x, include_embeddings=False):
        h = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(h)
        if include_embeddings:
            return vq_output['encodings'], vq_output['embeddings']
        else:
            return vq_output['encodings']

    def decode(self, encodings):
        h = F.embedding(encodings, self.codebook.embeddings)
        h = self.post_vq_conv(shift_dim(h, -1, 1))
        return self.decoder(h)

    def forward(self, x):
        # Save original shape
        orig_shape = x.shape
        
        # Encode
        z = self.pre_vq_conv(self.encoder(x))
        
        # VQ
        vq_output = self.codebook(z)
        
        # Decode
        x_recon = self.decoder(self.post_vq_conv(vq_output['embeddings']))
        
        # Ensure output matches input shape
        if x_recon.shape != orig_shape:
            x_recon = F.interpolate(x_recon, size=(orig_shape[2], orig_shape[3], orig_shape[4]), 
                                  mode='trilinear', align_corners=False)
        
        recon_loss = F.mse_loss(x_recon, x) / 0.06
        return recon_loss, x_recon, vq_output

    def training_step(self, batch, batch_idx):
        x = batch['video']
        recon_loss, _, vq_output = self.forward(x)
        commitment_loss = vq_output['commitment_loss']
        loss = recon_loss + commitment_loss
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['video']
        recon_loss, _, vq_output = self.forward(x)
        self.log('val/recon_loss', recon_loss, prog_bar=True)
        self.log('val/perplexity', vq_output['perplexity'], prog_bar=True)
        self.log('val/commitment_loss', vq_output['commitment_loss'], prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4, betas=(0.9, 0.999))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--embedding_dim', type=int, default=256)
        parser.add_argument('--n_codes', type=int, default=2048)
        parser.add_argument('--n_hiddens', type=int, default=240)
        parser.add_argument('--n_res_layers', type=int, default=4)
        parser.add_argument('--downsample', nargs='+', type=int, default=(4, 4, 4))
        return parser


class AxialBlock(nn.Module):
    def __init__(self, n_hiddens, n_head):
        super().__init__()
        kwargs = dict(shape=(0,) * 3, dim_q=n_hiddens,
                      dim_kv=n_hiddens, n_head=n_head,
                      n_layer=1, causal=False, attn_type='axial')
        self.attn_w = MultiHeadAttention(attn_kwargs=dict(axial_dim=-2),
                                         **kwargs)
        self.attn_h = MultiHeadAttention(attn_kwargs=dict(axial_dim=-3),
                                         **kwargs)
        self.attn_t = MultiHeadAttention(attn_kwargs=dict(axial_dim=-4),
                                         **kwargs)

    def forward(self, x):
        x = shift_dim(x, 1, -1)
        x = self.attn_w(x, x, x) + self.attn_h(x, x, x) + self.attn_t(x, x, x)
        x = shift_dim(x, -1, 1)
        return x


class AttentionResidualBlock(nn.Module):
    def __init__(self, n_hiddens):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU(),
            SamePadConv3d(n_hiddens, n_hiddens // 2, 3, bias=False),
            nn.BatchNorm3d(n_hiddens // 2),
            nn.ReLU(),
            SamePadConv3d(n_hiddens // 2, n_hiddens, 1, bias=False),
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU(),
            AxialBlock(n_hiddens, 2)
        )

    def forward(self, x):
        return x + self.block(x)

class Codebook(nn.Module):
    def __init__(self, n_codes, embedding_dim):
        super().__init__()
        self.register_buffer('embeddings', torch.randn(n_codes, embedding_dim))
        self.register_buffer('N', torch.zeros(n_codes))
        self.register_buffer('z_avg', self.embeddings.data.clone())

        self.n_codes = n_codes
        self.embedding_dim = embedding_dim
        self._need_init = True

    def _tile(self, x):
        d, ew = x.shape
        if d < self.n_codes:
            n_repeats = (self.n_codes + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _init_embeddings(self, z):
        # z: [b, c, t, h, w]
        self._need_init = False
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)
        y = self._tile(flat_inputs)

        d = y.shape[0]
        _k_rand = y[torch.randperm(y.shape[0])][:self.n_codes]
        if dist.is_initialized():
            dist.broadcast(_k_rand, 0)
        self.embeddings.data.copy_(_k_rand)
        self.z_avg.data.copy_(_k_rand)
        self.N.data.copy_(torch.ones(self.n_codes))

    def forward(self, z):
        # z: [b, c, t, h, w]
        if self._need_init and self.training:
            self._init_embeddings(z)
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)
        distances = (flat_inputs ** 2).sum(dim=1, keepdim=True) \
                    - 2 * flat_inputs @ self.embeddings.t() \
                    + (self.embeddings.t() ** 2).sum(dim=0, keepdim=True)

        encoding_indices = torch.argmin(distances, dim=1)
        encode_onehot = F.one_hot(encoding_indices, self.n_codes).type_as(flat_inputs)
        encoding_indices = encoding_indices.view(z.shape[0], *z.shape[2:])

        embeddings = F.embedding(encoding_indices, self.embeddings)
        embeddings = shift_dim(embeddings, -1, 1)

        commitment_loss = 0.25 * F.mse_loss(z, embeddings.detach())

        # EMA codebook update
        if self.training:
            n_total = encode_onehot.sum(dim=0)
            encode_sum = flat_inputs.t() @ encode_onehot
            if dist.is_initialized():
                dist.all_reduce(n_total)
                dist.all_reduce(encode_sum)

            self.N.data.mul_(0.99).add_(n_total, alpha=0.01)
            self.z_avg.data.mul_(0.99).add_(encode_sum.t(), alpha=0.01)

            n = self.N.sum()
            weights = (self.N + 1e-7) / (n + self.n_codes * 1e-7) * n
            encode_normalized = self.z_avg / weights.unsqueeze(1)
            self.embeddings.data.copy_(encode_normalized)

            y = self._tile(flat_inputs)
            _k_rand = y[torch.randperm(y.shape[0])][:self.n_codes]
            if dist.is_initialized():
                dist.broadcast(_k_rand, 0)

            usage = (self.N.view(self.n_codes, 1) >= 1).float()
            self.embeddings.data.mul_(usage).add_(_k_rand * (1 - usage))

        embeddings_st = (embeddings - z).detach() + z

        avg_probs = torch.mean(encode_onehot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return dict(embeddings=embeddings_st, encodings=encoding_indices,
                    commitment_loss=commitment_loss, perplexity=perplexity)

    def dictionary_lookup(self, encodings):
        embeddings = F.embedding(encodings, self.embeddings)
        return embeddings

class Encoder(nn.Module):
    def __init__(self, n_hiddens, n_res_layers, downsample):
        super().__init__()
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        self.convs = nn.ModuleList()
        max_ds = n_times_downsample.max()
        for i in range(max_ds):
            in_channels = 3 if i == 0 else n_hiddens
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            conv = SamePadConv3d(in_channels, n_hiddens, 4, stride=stride)
            self.convs.append(conv)
            n_times_downsample -= 1
        self.conv_last = SamePadConv3d(in_channels, n_hiddens, kernel_size=3)

        self.res_stack = nn.Sequential(
            *[AttentionResidualBlock(n_hiddens)
              for _ in range(n_res_layers)],
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU()
        )

    def forward(self, x):
        h = x
        for conv in self.convs:
            h = F.relu(conv(h))
        h = self.conv_last(h)
        h = self.res_stack(h)
        return h


class Decoder(nn.Module):
    def __init__(self, n_hiddens, n_res_layers, upsample):
        super().__init__()
        self.res_stack = nn.Sequential(
            *[AttentionResidualBlock(n_hiddens)
              for _ in range(n_res_layers)],
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU()
        )

        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()
        self.convts = nn.ModuleList()
        for i in range(max_us):
            out_channels = 3 if i == max_us - 1 else n_hiddens
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            convt = SamePadConvTranspose3d(n_hiddens, out_channels, 4,
                                           stride=us)
            self.convts.append(convt)
            n_times_upsample -= 1

    def forward(self, x):
        h = self.res_stack(x)
        for i, convt in enumerate(self.convts):
            h = convt(h)
            if i < len(self.convts) - 1:
                h = F.relu(h)
        return h


# Does not support dilation
class SamePadConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]: # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input))


class SamePadConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]: # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input

        self.convt = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                        stride=stride, bias=bias,
                                        padding=tuple([k - 1 for k in kernel_size]))

    def forward(self, x):
        return self.convt(F.pad(x, self.pad_input))


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        # x shape: [b, t, h, w, c]
        b, t, h, w, c = x.shape
        # Reshape to [b*t*h*w, c] for LayerNorm
        x = x.reshape(-1, c)
        x = self.norm(x)
        # Restore shape and apply function
        x = x.reshape(b, t, h, w, c)
        return self.fn(x)

class SpatioTemporalTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head=64, mlp_dim=None, dropout=0.):
        super().__init__()
        mlp_dim = mlp_dim or dim * 4
        self.layers = nn.ModuleList([])
        
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SpatialAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, TemporalAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        # x: [b, c, t, h, w]
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 3, 4, 1)  # [b, t, h, w, c]
        
        for spatial_attn, temporal_attn, ff in self.layers:
            # Spatial attention
            x = spatial_attn(x) + x
            # Temporal attention
            x = temporal_attn(x) + x
            # Feedforward
            x = ff(x) + x
            
        x = x.permute(0, 4, 1, 2, 3)  # [b, c, t, h, w]
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class SpatialAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x shape: [b, t, h, w, c]
        b, t, h, w, c = x.shape
        
        # Process each time step independently
        x = x.reshape(b * t, h * w, c)  # [bt, hw, c]
        
        # Get qkv
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 3 x [bt, hw, inner_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # Attention
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)
        
        # Reshape back
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        # Restore original shape
        out = out.reshape(b, t, h, w, c)
        return out

class TemporalAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x shape: [b, t, h, w, c]
        b, t, h, w, c = x.shape
        
        # Reshape for temporal attention
        x = x.reshape(b, t, h * w, c)  # [b, t, hw, c]
        
        # Get qkv
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 3 x [b, t, hw, inner_dim]
        q, k, v = map(lambda t: rearrange(t, 'b t n (h d) -> b h t n d', h=self.heads), qkv)

        # Attention
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)
        
        # Reshape back
        out = rearrange(out, 'b h t n d -> b t n (h d)')
        out = self.to_out(out)
        
        # Restore original shape
        out = out.reshape(b, t, h, w, c)
        return out

class TransformerEncoder(nn.Module):
    def __init__(self, n_hiddens, n_layers, downsample):
        super().__init__()
        
        # Initial projection from 3 channels to n_hiddens
        self.input_proj = nn.Conv3d(3, n_hiddens, 1)
        
        # Downsample layers
        self.downsample_layers = nn.ModuleList()
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        max_ds = n_times_downsample.max()
        
        for i in range(max_ds):
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            self.downsample_layers.append(
                nn.Conv3d(n_hiddens, n_hiddens, kernel_size=3, stride=stride, padding=1)
            )
            n_times_downsample -= 1
        
        # Transformer layers
        self.transformer = SpatioTemporalTransformer(
            dim=n_hiddens,
            depth=n_layers,
            heads=8,
            dim_head=n_hiddens // 8,
            mlp_dim=n_hiddens * 4,
            dropout=0.1
        )

    def forward(self, x):
        # x shape: [b, c, t, h, w]
        
        # Initial projection
        x = self.input_proj(x)
        
        # Downsample
        for layer in self.downsample_layers:
            x = F.gelu(layer(x))
        
        # Transform
        x = self.transformer(x)  # Transformer handles permutations internally
        
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, n_hiddens, n_layers, upsample):
        super().__init__()
        
        # Transformer layers
        self.transformer = SpatioTemporalTransformer(
            dim=n_hiddens,
            depth=n_layers,
            heads=8,
            dim_head=n_hiddens // 8,
            mlp_dim=n_hiddens * 4,
            dropout=0.1
        )
        
        # Upsample layers
        self.upsample_layers = nn.ModuleList()
        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()
        
        for i in range(max_us):
            stride = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            out_channels = 3 if i == max_us-1 else n_hiddens
            self.upsample_layers.append(
                nn.Sequential(
                    SamePadConvTranspose3d(n_hiddens, out_channels, 4, stride=stride),
                    nn.BatchNorm3d(out_channels) if i < max_us-1 else nn.Identity()
                )
            )
            n_times_upsample -= 1

    def forward(self, x):
        # Transform
        x = self.transformer(x)
        
        # Upsample
        for i, layer in enumerate(self.upsample_layers):
            x = layer(x)
            if i < len(self.upsample_layers) - 1:
                x = F.gelu(x)
        
        return x


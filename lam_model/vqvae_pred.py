import math
import argparse
import numpy as np
import matplotlib.pyplot as plt

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import torchvision.models as models

from .attention import MultiHeadAttention
from .utils import shift_dim

class VQVAE(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding_dim = args.embedding_dim
        self.n_codes = args.n_codes

        # Video encoder/decoder
        self.encoder = Encoder(args.n_hiddens, args.n_res_layers, args.downsample)
        self.decoder = Decoder(args.n_hiddens, args.n_res_layers, args.downsample)
        
        # Image encoder for initial frame
        self.image_encoder = self._build_image_encoder(args.n_hiddens)
        
        self.pre_vq_conv = SamePadConv3d(args.n_hiddens, args.embedding_dim, 1)
        self.post_vq_conv = SamePadConv3d(args.embedding_dim, args.n_hiddens, 1)

        # Fusion layer for combining action and frame features
        self.fusion_conv = SamePadConv3d(args.n_hiddens * 2, args.n_hiddens, 1)

        self.codebook = Codebook(args.n_codes, args.embedding_dim)
        self.save_hyperparameters()
        
        # For visualization
        self.viz_every_n_epochs = 5

    def _build_image_encoder(self, n_hiddens):
        """Build ResNet-based image encoder"""
        # Load pretrained ResNet
        resnet = models.resnet18(pretrained=True)
        
        # Remove the final FC layer
        layers = list(resnet.children())[:-1]
        
        # Add adaptation layers to match expected dimensions
        layers.extend([
            nn.Flatten(),
            nn.Linear(512, n_hiddens),  # ResNet18 has 512 features
            nn.ReLU(),
            # Reshape to spatial features
            Lambda(lambda x: x.view(x.size(0), n_hiddens, 1, 1, 1))
        ])
        
        return nn.Sequential(*layers)

    def encode_frame(self, x):
        """
        Encode a single frame using the image encoder
        Args:
            x: Input tensor of shape [B, C, H, W] or [B, C, 1, H, W]
        Returns:
            Encoded frame features [B, n_hiddens, 1, H//downsample, W//downsample]
        """
        if x.dim() == 4:
            x = x.unsqueeze(2)  # Add time dimension
        
        # Use the same encoder as action sequence to ensure matching dimensions
        features = self.encoder(x)  # This will downsample spatially
        features = self.pre_vq_conv(features)
        
        return features

    def encode_action(self, x):
        """
        Encode action sequence and get VQ embeddings
        Args:
            x: Input tensor of shape [B, C, T, H, W]
        Returns:
            Dictionary containing VQ output
        """
        z = self.pre_vq_conv(self.encoder(x))
        return self.codebook(z)

    def decode_with_action(self, action_embedding, initial_frame_features):
        """
        Decode using action embedding and initial frame features
        Args:
            action_embedding: VQ embedding representing action [B, C, T, H, W]
            initial_frame_features: Encoded features of first frame [B, C, 1, H, W]
        """
        # Print shapes for debugging
        # print(f"Action embedding shape: {action_embedding.shape}")
        # print(f"Initial frame features shape: {initial_frame_features.shape}")
        
        # Expand initial frame features to match action sequence length
        initial_frame_features = initial_frame_features.expand(
            -1, -1, action_embedding.size(2), -1, -1)
        
        # Process action embedding through post_vq_conv
        action_features = self.post_vq_conv(action_embedding)
        
        # Ensure spatial dimensions match
        if action_features.shape[-2:] != initial_frame_features.shape[-2:]:
            print("Warning: Spatial dimensions don't match, resizing initial frame features")
            initial_frame_features = F.interpolate(
                initial_frame_features,
                size=action_features.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Concatenate along channel dimension
        combined = torch.cat([action_features, initial_frame_features], dim=1)
        
        # Fuse features
        fused = self.fusion_conv(combined)
        
        # Decode
        return self.decoder(fused)

    def forward(self, batch):
        """
        Modified forward pass that handles sequence length T properly:
        - Input sequence: F1:T (T frames)
        - Actions: a1:T-1 (T-1 actions)
        - Predictions: F̂2:T (T-1 frames)
        - Target: F2:T (T-1 frames)
        """
        if isinstance(batch, dict):
            # Action-conditioned generation mode
            action_seq = batch['action']
            initial_frame = batch['initial_frame']
            
            # Get action embedding
            vq_output = self.encode_action(action_seq)
            
            # Encode initial frame
            initial_features = self.encode_frame(initial_frame)
            
            # Generate using action and initial frame
            x_gen = self.decode_with_action(vq_output['embeddings'], initial_features)
            
            return x_gen, vq_output
        else:
            # Standard reconstruction mode
            x = batch  # Shape: [B, C, T, H, W]
            
            # Split into initial frame and remaining frames
            initial_frame = x[:, :, 0:1]  # Shape: [B, C, 1, H, W]
            target_frames = x[:, :, 1:]   # Shape: [B, C, T-1, H, W]
            
            # Encode action sequence (using frames 1:T-1 to predict actions)
            vq_output = self.encode_action(target_frames)
            
            # Encode initial frame
            initial_features = self.encode_frame(initial_frame)
            
            # Generate reconstruction (frames 2:T)
            x_recon = self.decode_with_action(vq_output['embeddings'], initial_features)
            
            # Compare reconstruction with target frames (2:T)
            recon_loss = F.mse_loss(x_recon, target_frames) / 0.06
            
            return recon_loss, x_recon, vq_output

    def training_step(self, batch, batch_idx):
        """
        Training step with proper sequence handling:
        - Input sequence: F1:T (T frames)
        - Actions: a1:T-1 (T-1 actions)
        - Predictions: F̂2:T (T-1 frames)
        - Target: F2:T (T-1 frames)
        """
        if isinstance(batch, dict):
            # Action-conditioned generation mode
            x = batch['video'][:, :, 1:]  # Only use frames 2:T as targets
            x_gen, vq_output = self.forward(batch)
            recon_loss = F.mse_loss(x_gen, x) / 0.06
            commitment_loss = vq_output['commitment_loss']
        else:
            # Standard reconstruction mode
            recon_loss, _, vq_output = self.forward(batch)
            commitment_loss = vq_output['commitment_loss']
        
        loss = recon_loss + commitment_loss
        
        # Log training metrics
        self.log('train/recon_loss', recon_loss, prog_bar=True)
        self.log('train/commitment_loss', commitment_loss, prog_bar=True)
        self.log('train/total_loss', loss, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Handle both training and inference validation with proper sequence handling"""
        if isinstance(batch, dict):
            # Inference mode
            x = batch['video'][:, :, 1:]  # Only use frames 2:T as targets
            x_gen, vq_output = self.forward(batch)
            recon_loss = F.mse_loss(x_gen, x) / 0.06
        else:
            # Training mode
            recon_loss, x_gen, vq_output = self.forward(batch)
        
        # Log metrics
        self.log('val/recon_loss', recon_loss, prog_bar=True)
        self.log('val/perplexity', vq_output['perplexity'], prog_bar=True)
        self.log('val/commitment_loss', vq_output['commitment_loss'], prog_bar=True)
        
        # For visualization
        if batch_idx == 0:
            if isinstance(batch, dict):
                original = batch['video']
            else:
                original = batch
            self.val_step_outputs = {
                'original': original,
                'reconstruction': torch.cat([original[:, :, 0:1], x_gen], dim=2)  # Add initial frame back
            }
        
        return {
            'recon_loss': recon_loss,
            'ssim': 1.0 - recon_loss * 0.06  # Approximate SSIM from MSE
        }

    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % self.viz_every_n_epochs == 0:
            self._visualize_reconstructions()
    
    def _visualize_reconstructions(self):
        # Get original and reconstructed videos
        original = self.val_step_outputs['original']
        recon = self.val_step_outputs['reconstruction']
        
        # Select first video in batch
        original = original[0]  # [C, T, H, W]
        recon = recon[0]
        
        # Select frames to visualize (e.g., first, middle, last)
        t = original.shape[1]
        frame_indices = [0, t//2, -1]
        
        # Create figure
        fig, axes = plt.subplots(2, len(frame_indices), figsize=(15, 6))
        
        for i, idx in enumerate(frame_indices):
            # Original frame
            orig_frame = original[:, idx].cpu()
            axes[0, i].imshow(orig_frame.permute(1, 2, 0).clip(0, 1))
            axes[0, i].set_title(f'Original (t={idx})')
            axes[0, i].axis('off')
            
            # Reconstructed frame
            recon_frame = recon[:, idx].cpu()
            axes[1, i].imshow(recon_frame.permute(1, 2, 0).clip(0, 1))
            axes[1, i].set_title(f'Reconstructed (t={idx})')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        # Log figure to tensorboard
        self.logger.experiment.add_figure(
            'reconstructions', 
            fig, 
            global_step=self.current_epoch
        )
        plt.close(fig)

    def test_step(self, batch, batch_idx):
        x = batch['video']
        recon_loss, x_recon, vq_output = self.forward(x)
        
        # Log metrics
        self.log('test/recon_loss', recon_loss)
        
        return {
            'recon_loss': recon_loss,
        }
        
    def test_epoch_end(self, outputs):
        # Aggregate metrics across all test batches
        avg_recon_loss = torch.stack([x['recon_loss'] for x in outputs]).mean()
        
        # Log final metrics
        self.log('test/avg_recon_loss', avg_recon_loss)
        
        print(f"\nTest Results:")
        print(f"Average Reconstruction Loss: {avg_recon_loss:.4f}")

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
        parser.add_argument('--image_encoder', type=str, default='resnet18', 
                          choices=['resnet18', 'resnet34', 'resnet50'])
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


# Helper class for functional transforms
class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


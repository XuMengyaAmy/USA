import pytorch_lightning as pl
from argparse import ArgumentParser
import torch
import torch.nn.functional as F

from .dynamics_model import DynamicsTransformer
from .data_action import VideoData
from .vqvae_pred_transformer import VQVAE

class DynamicsTrainer(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters()

        # Load pretrained VQVAE
        self.vqvae = VQVAE.load_from_checkpoint(args.vqvae_checkpoint)
        self.vqvae.eval()
        for param in self.vqvae.parameters():
            param.requires_grad = False

        # Initialize dynamics model
        self.dynamics_model = DynamicsTransformer(
            n_tokens=args.n_codes,
            action_dim=args.embedding_dim,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            max_seq_len=args.sequence_length
        )

    def training_step(self, batch, batch_idx):
        # Get video sequence
        video = batch['video']  # [B, C, T, H, W]
        
        with torch.no_grad():
            # Get current frame tokens
            curr_frame = video[:, :, :-1]  # [B, C, 1, H, W]
            frame_features = self.vqvae.encode_frame(curr_frame)
            vq_output = self.vqvae.codebook(frame_features)
            frame_tokens = vq_output['z_indices'].flatten(1)  # [B, H*W]
            
            # Get target frame tokens
            target_frame = video[:, :, -1:]  # Last frame
            target_features = self.vqvae.encode_frame(target_frame)
            target_vq_output = self.vqvae.codebook(target_features)
            target_tokens = target_vq_output['z_indices'].flatten(1)  # [B, H*W]
            
            # Sample action from VQVAE
            action_seq = batch['action']  # [B, C, T-1, H, W]
            action_features = self.vqvae.encode_action(action_seq[:, :, -1:])
            action_vq_output = self.vqvae.codebook(action_features)
            action_embedding = action_vq_output['z_q']  # [B, action_dim]

        # Generate mask for MaskGIT training
        B = frame_tokens.shape[0]
        L = frame_tokens.shape[-1]  # H*W
        mask_ratio = torch.rand(1).item() * 0.8 + 0.1
        mask = torch.rand(B, L) > mask_ratio
        mask = mask.to(frame_tokens.device)

        # Forward pass
        logits = self.dynamics_model(frame_tokens, action_embedding, mask)

        # Compute loss only on masked positions
        loss = F.cross_entropy(
            logits[~mask].view(-1, self.args.n_codes),
            target_tokens[~mask].view(-1)
        )

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        video = batch['video']
        action_seq = batch['action']
        
        with torch.no_grad():
            # Get current frame tokens
            curr_frame = video[:, :, -2:-1]  # Get second-to-last frame
            frame_features = self.vqvae.encode_frame(curr_frame)
            vq_output = self.vqvae.codebook(frame_features)
            frame_tokens = vq_output['z_indices'].flatten(1)  # [B, H*W]
            
            # Get actual action embedding (for supervised prediction)
            action_features = self.vqvae.encode_action(action_seq[:, :, -1:])
            action_vq_output = self.vqvae.codebook(action_features)
            action_embedding = action_vq_output['z_q']  # [B, action_dim]

            # Generate next frame with actual action
            predicted_tokens = self.dynamics_model.generate(
                frame_tokens=frame_tokens,
                action_embedding=action_embedding,
                num_steps=10,
                temperature=0.8
            )

            # Convert predicted tokens to frame
            predicted_frame = self.vqvae.decode_from_indices(predicted_tokens)

            # Compute reconstruction loss with next frame
            next_frame = video[:, :, -1:]
            val_loss = F.mse_loss(predicted_frame, next_frame)

        self.log('val_loss', val_loss, prog_bar=True)

        # Visualize first batch
        if batch_idx == 0:
            B = frame_tokens.shape[0]
            predictions = []
            
            # First add supervised prediction
            predictions.append(predicted_frame[:1])
            
            # Then add random sampled predictions
            num_samples = 3  # Generate 3 random samples
            for _ in range(num_samples):
                # Sample random action
                random_indices = torch.randint(0, self.args.n_codes, (B, 1), device=frame_tokens.device)
                random_action_embedding = self.vqvae.codebook.embedding(random_indices).squeeze(1)
                
                # Generate prediction
                pred_tokens = self.dynamics_model.generate(
                    frame_tokens=frame_tokens,
                    action_embedding=random_action_embedding,
                    num_steps=10,
                    temperature=0.8
                )
                pred_frame = self.vqvae.decode_from_indices(pred_tokens)
                predictions.append(pred_frame[:1])
            
            # Visualize: input frame -> supervised prediction -> random predictions -> ground truth
            vis_tensor = torch.cat([
                curr_frame[:1],          # Input frame
                *predictions,            # Supervised + random predictions
                next_frame[:1]          # Ground truth
            ], dim=0)
            
            self.logger.experiment.add_images(
                'frame_predictions',
                vis_tensor.clamp(-0.5, 0.5) + 0.5,
                self.global_step
            )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.dynamics_model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.args.max_epochs,
            eta_min=self.args.min_lr
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        
        # Model parameters
        parser.add_argument('--d_model', type=int, default=512)
        parser.add_argument('--n_heads', type=int, default=8)
        parser.add_argument('--n_layers', type=int, default=6)
        parser.add_argument('--dropout', type=float, default=0.1)
        
        # Training parameters
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--min_lr', type=float, default=1e-5)
        parser.add_argument('--weight_decay', type=float, default=0.01)
        parser.add_argument('--max_epochs', type=int, default=100)
        
        return parser 
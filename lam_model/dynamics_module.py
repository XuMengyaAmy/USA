import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from .dynamics_model import DynamicsTransformer, MaskGITTrainer
from .vqvae_pred_transformer import VQVAE

class DynamicsModelTrainer(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters()

        # Initialize VQVAE (video tokenizer)
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
            max_seq_len=args.max_seq_len
        )

        # Initialize MaskGIT trainer
        self.maskgit_trainer = MaskGITTrainer(
            dynamics_model=self.dynamics_model,
            video_tokenizer=self.vqvae,
            latent_action_model=self.vqvae  # Using VQVAE's action encoding
        )

    def forward(self, video_sequence, actions):
        return self.maskgit_trainer.train_step(video_sequence, actions)

    def training_step(self, batch, batch_idx):
        # Unpack batch
        video_sequence = batch['video']  # [B, C, T, H, W]
        actions = batch['actions']       # [B, T-1, action_dim]

        # Forward pass
        loss = self(video_sequence, actions)

        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        video_sequence = batch['video']
        actions = batch['actions']

        # Compute validation loss
        with torch.no_grad():
            loss = self(video_sequence, actions)

        # Log validation metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        # Generate samples for visualization (first batch only)
        if batch_idx == 0:
            self.visualize_predictions(batch)

    def visualize_predictions(self, batch):
        """Generate and log video predictions"""
        video = batch['video']
        actions = batch['actions']
        B = video.size(0)

        with torch.no_grad():
            # Get initial frame tokens
            initial_frame = video[:, :, 0:1]  # [B, C, 1, H, W]
            initial_tokens = self.vqvae.encode_frame(initial_frame)

            # Generate future frames
            predicted_tokens = self.maskgit_trainer.sample(
                initial_tokens=initial_tokens,
                actions=actions,
                num_steps=10
            )

            # Decode predicted tokens
            predicted_video = self.vqvae.decode(predicted_tokens)

            # Log videos (first 4 sequences)
            n_vis = min(4, B)
            for i in range(n_vis):
                self.logger.experiment.add_video(
                    f'generation_{i}',
                    predicted_video[i:i+1],
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
        parser.add_argument('--max_seq_len', type=int, default=256)
        
        # Training parameters
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--min_lr', type=float, default=1e-5)
        parser.add_argument('--weight_decay', type=float, default=0.01)
        parser.add_argument('--max_epochs', type=int, default=100)
        
        # Checkpoint paths
        parser.add_argument('--vqvae_checkpoint', type=str, required=True,
                          help='Path to pretrained VQVAE checkpoint')
        
        return parser

def main():
    # Initialize parser
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = DynamicsModelTrainer.add_model_specific_args(parser)
    args = parser.parse_args()

    # Initialize trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor='val_loss',
                filename='dynamics-{epoch:02d}-{val_loss:.2f}',
                save_top_k=3,
                mode='min'
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='step')
        ]
    )

    # Initialize model
    model = DynamicsModelTrainer(args)

    # Train model
    trainer.fit(model)

if __name__ == '__main__':
    main() 
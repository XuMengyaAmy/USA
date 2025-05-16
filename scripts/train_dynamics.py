import os
import sys
import pytorch_lightning as pl
from argparse import ArgumentParser
import torch
import torch.nn.functional as F
import socket
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback

from lam_model.dynamics_model import DynamicsTransformer
from lam_model.data_action import VideoData
from lam_model.vqvae_pred_transformer import VQVAE

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import datetime

class TxtLoggingCallback(Callback):
    def __init__(self, filename="dynamics_val_logs.txt"):
        self.filename = filename
        if os.path.exists(self.filename):
            os.remove(self.filename)
        self.batch_outputs = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if outputs is not None and 'val_loss' in outputs:
            self.batch_outputs.append(outputs)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.batch_outputs:
            return
        
        losses = [out['val_loss'] for out in self.batch_outputs]
        avg_loss = torch.stack(losses).mean().item()
        
        epoch = trainer.current_epoch
        log_line = f"Validation Epoch {epoch}: Average loss = {avg_loss:.6f}\n"
        
        with open(self.filename, "a") as f:
            f.write(log_line)
        
        self.batch_outputs = []

class DynamicsModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters()

        # Load pretrained VQVAE
        self.vqvae = VQVAE.load_from_checkpoint(args.vqvae_checkpoint)
        self.vqvae.eval()
        for param in self.vqvae.parameters():
            param.requires_grad = False
        print('vqvae loaded')
        
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
        video = batch['video']  # [B, C, T, H, W]
        action_seq = batch['action']  # [B, C, T-1, H, W]

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
            
            # Get action embedding
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

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
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
            
            # Get actual action embedding
            action_features = self.vqvae.encode_action(action_seq[:, :, -1:])
            action_vq_output = self.vqvae.codebook(action_features)
            action_embedding = action_vq_output['z_q']  # [B, action_dim]

            # Generate next frame
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

        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        return {'val_loss': val_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.dynamics_model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        
        # Use a simpler scheduler that doesn't need to know total steps
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.args.max_epochs,  # Use epochs instead of steps
            eta_min=self.args.min_lr
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',  # Call scheduler after each epoch
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
        parser.add_argument('--max_epochs', type=int, default=100)
        
        # Training parameters
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--min_lr', type=float, default=1e-5)
        parser.add_argument('--weight_decay', type=float, default=0.01)
        
        return parser

# Then define other functions
def ddp_setup(rank, world_size):
    """Setup for DDP training using file:// initialization"""
    try:
        # Create a temporary file for coordination
        import tempfile
        import time
        
        # Generate a unique filename for this run
        timestamp = int(time.time())
        file_path = tempfile.gettempdir() + f"/pytorch_ddp_file_{timestamp}.txt"
        print(f"Rank {rank}: Using file initialization with path: {file_path}")
        
        # Set the device
        torch.cuda.set_device(rank)
        
        # Initialize process group with file:// method
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{file_path}",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=300)
        )
        print(f"Rank {rank}: Successfully initialized process group")
    except Exception as e:
        print(f"Rank {rank}: Error initializing process group: {e}")
        import traceback
        traceback.print_exc()
        raise

def train_model(rank, world_size, args):
    print(f"Rank {rank}: Starting setup")
    # Setup DDP
    ddp_setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    print(f"Rank {rank}: DDP setup complete")
    
    try:
        # Create data module
        print(f"Rank {rank}: Creating data module")
        data = VideoData(args)
        data.setup()
        print(f"Rank {rank}: Data module setup complete")
        
        # Get datasets from the dataloader methods
        print(f"Rank {rank}: Getting dataloaders")
        train_loader = data.train_dataloader()
        val_loader = data.val_dataloader()
        
        # Extract datasets from the existing dataloaders
        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset
        
        print(f"Rank {rank}: Successfully retrieved datasets. Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
        
        # Create samplers for distributed training
        print(f"Rank {rank}: Creating samplers")
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        
        # Create new dataloaders with distributed samplers
        print(f"Rank {rank}: Creating new dataloaders")
        effective_workers = min(4, args.num_workers)
        
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=train_sampler,
            num_workers=effective_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            drop_last=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=effective_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            drop_last=True
        )
        
        # Test dataloaders by iterating over a single batch
        print(f"Rank {rank}: Testing dataloaders")
        train_iterator = iter(train_loader)
        train_batch = next(train_iterator)
        print(f"Rank {rank}: Successfully got a training batch")
        
        val_iterator = iter(val_loader)
        val_batch = next(val_iterator)
        print(f"Rank {rank}: Successfully got a validation batch")
        
        # Scale learning rate
        args.learning_rate = args.learning_rate * world_size
        print(f"Rank {rank}: Using learning rate {args.learning_rate}")
        
        # Initialize model
        print(f"Rank {rank}: Initializing model")
        model = DynamicsModel(args).to(device)
        print(f"Rank {rank}: Model initialized")
        
        # Setup callbacks
        callbacks = []
        
        if rank == 0:  # Only on main process
            print(f"Rank {rank}: Setting up callbacks")
            checkpoint_callback = ModelCheckpoint(
                dirpath=args.checkpoint_dir,
                filename='dynamics-{epoch:02d}-{val_loss:.2f}',
                monitor='val_loss',
                mode='min',
                save_top_k=3
            )
            callbacks.append(checkpoint_callback)
            
            txt_logging_callback = TxtLoggingCallback()
            callbacks.append(txt_logging_callback)
            
            lr_monitor = LearningRateMonitor(logging_interval='step')
            callbacks.append(lr_monitor)
        
        # Logger
        logger = None
        if args.use_wandb and rank == 0:
            print(f"Rank {rank}: Setting up logger")
            logger = WandbLogger(project="surgical_robot", name="dynamics_training")
        
        # Update trainer kwargs for older PyTorch Lightning versions
        print(f"Rank {rank}: Building trainer arguments")
        trainer_kwargs = {
            'max_epochs': args.max_epochs,
            'gpus': [rank],  # Use only the current GPU
            'callbacks': callbacks,
            'logger': logger,
            'log_every_n_steps': 100,
            'check_val_every_n_epoch': 1,
            'precision': 16,
            # For older PyTorch Lightning, use these instead of 'strategy'
            'accelerator': 'ddp',  # Use DDP for distributed training
            # Don't include 'strategy' parameter
        }
        
        if hasattr(args, 'resume_from_checkpoint') and args.resume_from_checkpoint:
            trainer_kwargs['resume_from_checkpoint'] = args.resume_from_checkpoint
        
        # Create trainer
        print(f"Rank {rank}: Creating trainer")
        trainer = pl.Trainer(**trainer_kwargs)
        print(f"Rank {rank}: Trainer created successfully")
        
        # Debug: verify model methods
        print(f"Rank {rank}: Model has training_step: {hasattr(model, 'training_step')}")
        print(f"Rank {rank}: Model has configure_optimizers: {hasattr(model, 'configure_optimizers')}")
        
        # For older PyTorch Lightning, use positional arguments
        print(f"Rank {rank}: Starting trainer.fit()")
        trainer.fit(model, train_loader, val_loader)
        print(f"Rank {rank}: Training completed successfully")
        
    except Exception as e:
        print(f"Rank {rank}: Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        print(f"Rank {rank}: Cleaning up process group")
        destroy_process_group()
        print(f"Rank {rank}: Process group destroyed")

def main():
    pl.seed_everything(1234)
    
    parser = ArgumentParser()
    
    # Add program level args
    parser.add_argument('--data_path', type=str,
                       help='Path to video data directory', 
                       default='/research/d1/gds/kjshi/Surgical_Robot/surgical_data/splitted_dataset')
    parser.add_argument('--vqvae_checkpoint', type=str,
                       help='Path to pretrained VQVAE checkpoint', 
                       default='/research/d1/gds/kjshi/Surgical_Robot/VideoGPT/checkpoints/vqvae_action_prediction_new_dataset_transformer_10/epoch=2-val/recon_loss=0.03.ckpt')
    parser.add_argument('--checkpoint_dir', type=str,
                       help='Directory to save model checkpoints',
                       default='/research/d1/gds/kjshi/Surgical_Robot/VideoGPT/checkpoints/dynamics_model')
    parser.add_argument('--n_codes', type=int, default=1024)
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--sequence_length', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpus', type=int, default=2)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--resolution', default=256)
    
    # Add model specific args
    parser = DynamicsModel.add_model_specific_args(parser)
    args = parser.parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Add environment variables to help with debugging
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["PYTHONFAULTHANDLER"] = "1"
    
    # Launch DDP training
    world_size = args.gpus
    try:
        mp.spawn(
            train_model,
            args=(world_size, args),
            nprocs=world_size
        )
    except Exception as e:
        import traceback
        print(f"Error during training: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main() 
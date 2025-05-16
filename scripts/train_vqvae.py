import os
import sys
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
import socket
import random

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import your modules
from lam_model.vqvae_pred_transformer import VQVAE
from lam_model.data_action import VideoData

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


class TxtLoggingCallback(Callback):
    def __init__(self, filename="val_logs.txt"):
        self.filename = filename
        if os.path.exists(self.filename):
            os.remove(self.filename)
        self.batch_outputs = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if outputs is not None and 'recon_loss' in outputs and 'ssim' in outputs:
            self.batch_outputs.append(outputs)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.batch_outputs:
            return

        # Compute averages from the collected outputs
        recon_losses = [out['recon_loss'] for out in self.batch_outputs]
        ssims = [out['ssim'] for out in self.batch_outputs]
        avg_recon_loss = torch.stack(recon_losses).mean().item()
        avg_ssim = torch.stack(ssims).mean().item()
        
        epoch = trainer.current_epoch
        log_line = (f"Validation Epoch {epoch}: Average recon_loss = {avg_recon_loss:.6f}, "
                    f"Average SSIM = {avg_ssim:.6f}\n")
        
        with open(self.filename, "a") as f:
            f.write(log_line)
        
        self.batch_outputs = []


class SetEpochCallback(Callback):
    def __init__(self, train_sampler):
        self.train_sampler = train_sampler
        
    def on_train_epoch_start(self, trainer, pl_module):
        # Set epoch for proper shuffling
        self.train_sampler.set_epoch(trainer.current_epoch)


def find_free_port():
    """Find a free port on localhost"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    # Try different ports if the default one is in use
    port = 12355
    max_tries = 10
    
    for attempt in range(max_tries):
        try:
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = str(port + attempt)
            
            # Set CUDA device before initializing process group
            torch.cuda.set_device(rank)
            
            # Try using gloo backend first, which is more reliable
            init_process_group(backend="gloo", rank=rank, world_size=world_size)
            print(f"Rank {rank}: Successfully initialized process group with gloo backend on port {port + attempt}")
            return
        except RuntimeError as e:
            if "Address already in use" in str(e) and attempt < max_tries - 1:
                print(f"Rank {rank}: Port {port + attempt} already in use, trying next port")
                continue
            else:
                raise


def train_model(rank, world_size, args):
    # Setup DDP
    ddp_setup(rank, world_size)
    
    # Set device
    device = torch.device(f'cuda:{rank}')
    
    try:
        # Create data module
        data = VideoData(args)
        data.setup()
        
        # Create samplers for distributed training
        train_sampler = DistributedSampler(
            data.train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        
        val_sampler = DistributedSampler(
            data.val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        
        # Optimize num_workers based on GPU count
        effective_workers = min(4, args.num_workers)
        
        # Create data loaders with distributed samplers and error handling
        train_loader = torch.utils.data.DataLoader(
            dataset=data.train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=train_sampler,
            num_workers=effective_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            drop_last=True,  # Drop incomplete batches
        )
        
        val_loader = torch.utils.data.DataLoader(
            dataset=data.val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=effective_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            drop_last=True,  # Drop incomplete batches
        )
        
        # Scale learning rate based on number of GPUs
        args.learning_rate = args.learning_rate * world_size
        print(f"Rank {rank}: Using learning rate {args.learning_rate}")
        
        # Initialize model
        model = VQVAE(args).to(device)
        model = DDP(model, device_ids=[rank])
        
        # Setup callbacks
        callbacks = []
        
        if rank == 0:  # Only create these on the main process
            checkpoint_callback = ModelCheckpoint(
                dirpath=f'checkpoints/{args.exp_name}',
                filename='{epoch}-{val/recon_loss:.2f}',
                monitor='val/recon_loss',
                mode='min',
                save_last=True,
                save_top_k=3,
                every_n_train_steps=args.save_every_n_steps
            )
            callbacks.append(checkpoint_callback)
            
            txt_logging_callback = TxtLoggingCallback(filename="val_logs.txt")
            callbacks.append(txt_logging_callback)
            
            if train_sampler is not None:
                epoch_callback = SetEpochCallback(train_sampler)
                callbacks.append(epoch_callback)
        
        # Logger
        logger = None
        if args.use_wandb and rank == 0:
            logger = WandbLogger(project="surgical_robot", name=args.exp_name)
        
        # Training configuration
        trainer_kwargs = {
            'callbacks': callbacks,
            'logger': logger,
            'max_steps': args.max_steps,
            'val_check_interval': args.val_check_interval,
            'gradient_clip_val': 1.0,
            'accumulate_grad_batches': 1,
            'precision': 16,
            'gpus': [rank],
            'distributed_backend': 'ddp',
        }
        
        trainer = pl.Trainer(**trainer_kwargs)
        
        # Clear cache before training
        torch.cuda.empty_cache()
        
        # Train the model
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
    except Exception as e:
        print(f"Rank {rank}: Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        destroy_process_group()


def main():
    pl.seed_everything(1234)
    
    parser = argparse.ArgumentParser()
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='our_data_path')
    parser.add_argument('--sequence_length', type=int, default=10,
                        help='Length of video sequence (T). Will use T-1 frames to predict actions')
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    
    # Training arguments
    parser.add_argument('--max_steps', type=int, default=30000)
    parser.add_argument('--save_every_n_steps', type=int, default=5000)
    parser.add_argument('--val_check_interval', type=int, default=5000)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # Add experiment tracking
    parser.add_argument('--exp_name', type=str, default='vqvae_action_prediction_new_dataset_transformer_10_test')
    parser.add_argument('--use_wandb', action='store_true', default=True,
                        help='Use Weights & Biases logging (enabled by default)')
    
    # Add memory optimization arguments
    parser.add_argument('--gradient_checkpointing', action='store_true', 
                        help='Enable gradient checkpointing to save memory')
    parser.add_argument('--mixed_precision', action='store_true', 
                        help='Use mixed precision training')
    parser.add_argument('--batch_size_per_gpu', type=int, default=None,
                        help='Batch size per GPU (overrides batch_size)')
    parser.add_argument('--vq_codebook_size', type=int, default=1024,
                        help='Size of VQ codebook')
    parser.add_argument('--vq_embed_dim', type=int, default=256,
                        help='Embedding dimension for VQ')
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Embedding dimension for VQ')
    parser.add_argument('--n_codes', type=int, default=1024,
                        help='Number of codes')
    parser.add_argument('--n_hiddens', type=int, default=256,
                        help='Number of hidden units')
    parser.add_argument('--downsample', nargs='+', type=int, default=(1, 4, 4))

    parser.add_argument('--n_res_layers', type=int, default=4,
                        help='Number of residual layers')
    parser.add_argument('--action_condition_length', type=int, default=None,
                        help='Number of previous frames to condition action prediction on. If None, uses all previous frames')

    args = parser.parse_args()
    
    # If action_condition_length not specified, use all previous frames
    if args.action_condition_length is None:
        args.action_condition_length = args.sequence_length - 1
        
    print(f"Training with sequence length {args.sequence_length}")
    print(f"Using {args.action_condition_length} previous frames to condition action prediction")
    
    # Calculate effective batch size per GPU
    if args.batch_size_per_gpu is not None:
        args.batch_size = args.batch_size_per_gpu
        print(f"Setting batch size per GPU to {args.batch_size}")
    
    # Scale learning rate based on number of GPUs
    if args.gpus > 1:
        args.learning_rate = args.learning_rate * args.gpus
        print(f"Using {args.gpus} GPUs with learning rate {args.learning_rate}")
    
    # Create data module with optimized settings
    args.num_workers = 2  # Use minimal workers but not zero for better throughput
    data = VideoData(args)
    
    # Initialize model with memory optimizations
    model = VQVAE(args)
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model.use_gradient_checkpointing()
        print("Gradient checkpointing enabled for memory efficiency")
    
    # Setup callbacks
    callbacks = []
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'checkpoints/{args.exp_name}',
        filename='{epoch}-{val/recon_loss:.2f}',
        monitor='val/recon_loss',
        mode='min',
        save_last=True,
        save_top_k=3,
        every_n_train_steps=args.save_every_n_steps
    )
    callbacks.append(checkpoint_callback)
    
    txt_logging_callback = TxtLoggingCallback(filename="val_logs.txt")
    callbacks.append(txt_logging_callback)
    
    # Logger setup
    logger = None
    if args.use_wandb:
        logger = WandbLogger(project="surgical_robot", name=args.exp_name)
        # Only add LearningRateMonitor if we're using a logger
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)
    
    # Training configuration with memory optimizations
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        max_steps=args.max_steps,
        val_check_interval=args.val_check_interval,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.gpus if args.gpus > 1 else 1,
        precision=16 if args.mixed_precision else 32,  # Use mixed precision
        gpus=min(args.gpus, 1),  # Use at most 1 GPU with accumulation
    )
    
    # Train the model
    print("Starting training with memory optimizations...")
    trainer.fit(model, data)
    print("Training completed!")


if __name__ == '__main__':
    # Set environment variables for better debugging
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["PYTHONFAULTHANDLER"] = "1"
    
    try:
        main()
    except Exception as e:
        import traceback
        print(f"Error during training: {e}")
        traceback.print_exc()



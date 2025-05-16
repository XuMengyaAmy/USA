import torch
import pytorch_lightning as pl
from lam_model.vqvae import VQVAE
from lam_model.data import VideoData
import matplotlib.pyplot as plt
import os
import sys
import argparse
import cv2
import numpy as np

sys.path.append('/research/d1/gds/kjshi/Surgical_Robot/USA')

def save_video_frames(video, recon_video, save_dir, prefix="sample"):
    """
    Save original and reconstructed video frames
    video, recon_video: [C, T, H, W] tensors in range [-0.5, 0.5]
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Move tensors to CPU and convert to numpy
    video = video.cpu()
    recon_video = recon_video.cpu()
    
    # Save frames for each timestep
    for t in range(video.shape[1]):
        # Original frame
        orig_frame = video[:, t]  # [C, H, W]
        
        # Convert from RGB to BGR and [C, H, W] to [H, W, C]
        orig_frame = orig_frame.permute(1, 2, 0).numpy()  # [H, W, C]
        # Add 0.5 to convert from [-0.5, 0.5] to [0, 1]
        orig_frame = (orig_frame + 0.5)
        orig_frame = (orig_frame * 255).clip(0, 255).astype(np.uint8)
        orig_frame = cv2.cvtColor(orig_frame, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(
            os.path.join(save_dir, f"{prefix}_orig_frame_{t:03d}.png"),
            orig_frame
        )
        
        # Reconstructed frame
        recon_frame = recon_video[:, t]
        recon_frame = recon_frame.permute(1, 2, 0).numpy()
        # Add 0.5 to convert from [-0.5, 0.5] to [0, 1]
        recon_frame = (recon_frame + 0.5)
        recon_frame = (recon_frame * 255).clip(0, 255).astype(np.uint8)
        recon_frame = cv2.cvtColor(recon_frame, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(
            os.path.join(save_dir, f"{prefix}_recon_frame_{t:03d}.png"),
            recon_frame
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/research/d1/gds/kjshi/Surgical_Robot/surgical_data/splittedDataset')
    parser.add_argument('--sequence_length', type=int, default=10)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    # Load the checkpoint
    ckpt_path = "/research/d1/gds/kjshi/Surgical_Robot/VideoGPT/checkpoints/vqvae_actino_pred/epoch=2-val/recon_loss=0.02.ckpt"
    model = VQVAE.load_from_checkpoint(ckpt_path)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    # Initialize data module
    data = VideoData(args)
    data.setup('validate')

    # Get a batch from validation set
    val_dataloader = data.val_dataloader()
    batch = next(iter(val_dataloader))
    input_video = batch['video']
    print("Input video shape:", input_video.shape)

    with torch.no_grad():
        # Move input to GPU if available
        if torch.cuda.is_available():
            input_video = input_video.cuda()
        
        # Get reconstruction
        _, x_recon, _ = model(input_video)
        
        # Save frames for each video in batch
        for b in range(input_video.shape[0]):
            save_video_frames(
                input_video[b],  # [C, T, H, W]
                x_recon[b],      # [C, T, H, W]
                save_dir="/research/d1/gds/kjshi/Surgical_Robot/VideoGPT/results/reconstruction",
                prefix=f"video_{b}"
            )

if __name__ == "__main__":
    main() 
    print('finished')
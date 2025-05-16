import torch
import pytorch_lightning as pl
from lam_model.vqvae_pred_transformer import VQVAE
from lam_model.data_action import VideoData
import matplotlib.pyplot as plt
import os
import sys
import argparse
import cv2
import numpy as np
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from pytorch_fid import fid_score
from collections import defaultdict

sys.path.append('/research/d1/gds/kjshi/Surgical_Robot/USA')

def save_prediction_frames(current_frame, future_frames, predicted_frames, save_dir, prefix="sample"):
    """
    Save original frames and predicted frames sequence
    current_frame: [C, H, W] initial frame
    future_frames: [C, T, H, W] ground truth future frames
    predicted_frames: [C, T, H, W] model's predicted future frames
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Function to process and save a single frame
    def process_and_save_frame(frame, name):
        frame = frame.cpu()
        frame = frame.permute(1, 2, 0).numpy()  # [H, W, C]
        frame = (frame + 0.5)  # Convert from [-0.5, 0.5] to [0, 1]
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, f"{name}.png"), frame)
    
    # Save initial frame
    process_and_save_frame(current_frame, "current_frame")
    
    # Save each frame in the sequences
    for t in range(future_frames.shape[1]):
        process_and_save_frame(future_frames[:, t], f"future_frame_gt_{t:02d}")
        process_and_save_frame(predicted_frames[:, t], f"future_frame_pred_{t:02d}")

def visualize_action_prediction(current_frame, future_frames, predicted_frames, save_path, num_frames=5):
    """Create a side-by-side visualization of the frame sequences"""
    # Limit to showing only `num_frames` for clarity
    frames_to_show = min(num_frames, future_frames.shape[1])
    
    # Create figure with rows for GT and prediction
    fig, axes = plt.subplots(3, frames_to_show + 1, figsize=(3 * (frames_to_show + 1), 9))
    
    # Convert tensors to numpy and adjust range
    current_np = current_frame.cpu().permute(1, 2, 0).numpy() + 0.5
    future_np = [future_frames[:, t].cpu().permute(1, 2, 0).numpy() + 0.5 for t in range(frames_to_show)]
    pred_np = [predicted_frames[:, t].cpu().permute(1, 2, 0).numpy() + 0.5 for t in range(frames_to_show)]
    
    # Plot current frame
    axes[0, 0].imshow(current_np.clip(0, 1))
    axes[0, 0].set_title('Initial Frame')
    axes[0, 0].axis('off')
    axes[1, 0].imshow(current_np.clip(0, 1))
    axes[1, 0].axis('off')
    axes[2, 0].imshow(current_np.clip(0, 1))
    axes[2, 0].axis('off')
    
    # Plot future and predicted frames
    for t in range(frames_to_show):
        axes[0, t + 1].imshow(future_np[t].clip(0, 1))
        axes[0, t + 1].set_title(f'GT Frame {t+1}')
        axes[0, t + 1].axis('off')
        
        axes[1, t + 1].imshow(pred_np[t].clip(0, 1))
        axes[1, t + 1].set_title(f'Pred Frame {t+1}')
        axes[1, t + 1].axis('off')
        
        # Difference map
        diff = np.abs(future_np[t] - pred_np[t])
        diff = diff.mean(axis=2)  # Average across channels
        diff = diff / diff.max()  # Normalize
        axes[2, t + 1].imshow(diff, cmap='hot')
        axes[2, t + 1].set_title(f'Diff Frame {t+1}')
        axes[2, t + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def calculate_metrics_sequence(pred_frames, gt_frames):
    """Calculate PSNR and SSIM between predicted and ground truth frame sequences"""
    # pred_frames: [C, T, H, W], gt_frames: [C, T, H, W]
    T = pred_frames.shape[1]
    metrics = {
        'psnr_per_frame': [],
        'ssim_per_frame': [],
        'mse_per_frame': []
    }
    
    for t in range(T):
        pred_frame = pred_frames[:, t]  # [C, H, W]
        gt_frame = gt_frames[:, t]  # [C, H, W]
        
        # Convert to numpy arrays in range [0, 1]
        pred_np = pred_frame.cpu().permute(1, 2, 0).numpy() + 0.5
        gt_np = gt_frame.cpu().permute(1, 2, 0).numpy() + 0.5
        pred_np = np.clip(pred_np, 0, 1)
        gt_np = np.clip(gt_np, 0, 1)
        
        # Calculate MSE and PSNR
        mse = F.mse_loss(pred_frame, gt_frame).item()
        psnr = -10 * np.log10(mse + 1e-8)
        
        # Calculate SSIM with explicit window size
        try:
            ssim_value = ssim(
                gt_np, pred_np,
                channel_axis=2,  # Specify channel axis
                data_range=1.0,
                win_size=3  # Use smaller window size
            )
        except ValueError:
            print(f"Warning: SSIM calculation failed for frame {t}")
            ssim_value = -1
        
        metrics['psnr_per_frame'].append(psnr)
        metrics['ssim_per_frame'].append(ssim_value)
        metrics['mse_per_frame'].append(mse)
    
    # Calculate average metrics across frames
    metrics['psnr'] = np.mean(metrics['psnr_per_frame'])
    metrics['ssim'] = np.mean(metrics['ssim_per_frame'])
    metrics['mse'] = np.mean(metrics['mse_per_frame'])
    
    return metrics

def create_comparison_video(current_frame, future_frames, predicted_frames, output_path, fps=5):
    """Create a side-by-side video comparison of ground truth vs predicted frames"""
    # Convert tensors to numpy
    current_np = (current_frame.cpu().permute(1, 2, 0).numpy() + 0.5) * 255
    current_np = current_np.astype(np.uint8)
    
    # Create video writer
    h, w = current_np.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w*3, h))
    
    # Write initial frame
    combined_frame = np.hstack([current_np, current_np, current_np])
    combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)
    video_writer.write(combined_frame)
    
    # Write each frame in the sequence
    for t in range(future_frames.shape[1]):
        # Convert frames to numpy
        gt_np = (future_frames[:, t].cpu().permute(1, 2, 0).numpy() + 0.5) * 255
        gt_np = gt_np.astype(np.uint8)
        
        pred_np = (predicted_frames[:, t].cpu().permute(1, 2, 0).numpy() + 0.5) * 255
        pred_np = pred_np.astype(np.uint8)
        
        # Create difference map
        diff = cv2.absdiff(gt_np, pred_np)
        diff = cv2.applyColorMap(cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY), cv2.COLORMAP_JET)
        
        # Combine frames side by side: [GT | Predicted | Difference]
        combined_frame = np.hstack([
            cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR),
            diff
        ])
        
        # Write combined frame to video
        video_writer.write(combined_frame)
    
    # Release video writer
    video_writer.release()
    print(f"Saved video comparison to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/research/d1/gds/kjshi/Surgical_Robot/surgical_data/splitted_dataset')
    parser.add_argument('--sequence_length', type=int, default=31)  # 1 initial + 10 to predict
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1)  # Smaller batch size for longer sequences
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='results/action_prediction_transformer-31_sequence_folder',
                        help='Directory to save results')
    parser.add_argument('--num_samples', type=int, default=30,
                        help='Number of samples to test')
    parser.add_argument('--checkpoint_path', type=str, 
                        default="/research/d1/gds/kjshi/Surgical_Robot/VideoGPT/checkpoints/vqvae_action_prediction_new_dataset_transformer_10/epoch=3-val/recon_loss=0.03.ckpt",
                        help='Path to model checkpoint')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the checkpoint
    print(f"Loading model from {args.checkpoint_path}")
    model = VQVAE.load_from_checkpoint(args.checkpoint_path)
    print('Model loaded')
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    # Initialize data module
    data = VideoData(args)
    data.setup('test')
    test_dataloader = data.test_dataloader()

    # Collect metrics
    all_metrics = defaultdict(list)
    predicted_sequences = []  # List to store predicted frame sequences
    ground_truth_sequences = []  # List to store ground truth frame sequences
    
    # Create file to save clip names and metrics
    clip_metrics_file = os.path.join(args.output_dir, 'clip_metrics.txt')
    with open(clip_metrics_file, 'w') as f:
        f.write("Clip Name, PSNR, SSIM, MSE\n")

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            if i >= args.num_samples:
                break

            if torch.cuda.is_available():
                video = batch['video'].cuda()
            
            # Process each sequence in the batch
            for b in range(video.shape[0]):
                # Get clip name if available in the batch
                clip_name = batch['clip_name'][b] if 'clip_name' in batch else f"unknown_clip_{i}_{b}"
                print(f"\nTesting on clip: {clip_name}")
                
                # Create a specific folder for this clip
                clip_folder = os.path.join(args.output_dir, clip_name)
                os.makedirs(clip_folder, exist_ok=True)
                
                # Get current and future frames
                current_frame = video[b, :, 0]  # [C, H, W]
                future_frames = video[b, :, 1:]  # [C, T-1, H, W]
                
                # Create input dictionary for the model
                model_input = {
                    'video': video[b:b+1],  # [1, C, T, H, W]
                    'action': video[b:b+1, :, 1:],  # [1, C, T-1, H, W]
                    'initial_frame': current_frame.unsqueeze(0).unsqueeze(2)  # [1, C, 1, H, W]
                }
                
                # Get model prediction
                outputs = model(model_input)
                if isinstance(outputs, tuple):
                    x_recon = outputs[0]
                    vq_output = outputs[1]
                else:
                    x_recon = outputs['future_frame_pred']
                
                # Extract predicted sequence
                predicted_frames = x_recon[0]  # [C, T, H, W]
                
                # Calculate metrics
                metrics = calculate_metrics_sequence(predicted_frames, future_frames)
                for k, v in metrics.items():
                    if k not in ['psnr_per_frame', 'ssim_per_frame', 'mse_per_frame']:
                        all_metrics[k].append(v)
                
                # Store frames for FVD calculation
                predicted_sequences.append(predicted_frames.cpu())
                ground_truth_sequences.append(future_frames.cpu())
                
                # Save the individual frames in the clip folder
                save_prediction_frames(
                    current_frame,
                    future_frames,
                    predicted_frames,
                    save_dir=clip_folder,
                    prefix="frame"
                )
                
                # Create visualization in the clip folder
                visualize_action_prediction(
                    current_frame,
                    future_frames,
                    predicted_frames,
                    save_path=os.path.join(clip_folder, "comparison.png")
                )
                
                # Print per-sample metrics
                print(f"Clip: {clip_name}")
                print(f"Average PSNR: {metrics['psnr']:.2f} dB")
                print(f"Average SSIM: {metrics['ssim']:.4f}")
                print(f"PSNR per frame: {[f'{p:.2f}' for p in metrics['psnr_per_frame']]}")
                print(f"SSIM per frame: {[f'{s:.4f}' for s in metrics['ssim_per_frame']]}")
                
                # Save metrics to main metrics file
                with open(clip_metrics_file, 'a') as f:
                    f.write(f"{clip_name}, {metrics['psnr']:.2f}, {metrics['ssim']:.4f}, {metrics['mse']:.6f}\n")
                
                # Create frame-by-frame metrics file in the clip folder
                frame_metrics_file = os.path.join(clip_folder, "frame_metrics.txt")
                with open(frame_metrics_file, 'w') as f:
                    f.write("Frame, PSNR, SSIM, MSE\n")
                    for t in range(len(metrics['psnr_per_frame'])):
                        f.write(f"{t+1}, {metrics['psnr_per_frame'][t]:.2f}, {metrics['ssim_per_frame'][t]:.4f}, {metrics['mse_per_frame'][t]:.6f}\n")
                
                # Create a video file in the clip folder
                pred_video_path = os.path.join(clip_folder, "comparison_video.mp4")
                create_comparison_video(
                    current_frame, 
                    future_frames, 
                    predicted_frames, 
                    pred_video_path
                )

        # Calculate average PSNR and SSIM
        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
        print("\nAverage Metrics:")
        print(f"PSNR: {avg_metrics['psnr']:.2f} dB")
        print(f"SSIM: {avg_metrics['ssim']:.4f}")
        
        # Add average metrics to file
        with open(clip_metrics_file, 'a') as f:
            f.write(f"AVERAGE, {avg_metrics['psnr']:.2f}, {avg_metrics['ssim']:.4f}, {avg_metrics['mse']:.6f}\n")
        
    print('Testing completed. Results saved in:', args.output_dir)
    print(f'Clip-by-clip metrics saved to: {clip_metrics_file}')

if __name__ == "__main__":
    main()
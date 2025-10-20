#!/usr/bin/env python
"""
Massive RNN Inference Script
Loads a trained model, generates inference sequences, and evaluates performance.
"""

import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'

import matplotlib
matplotlib.use('Agg')

import torch
import numpy as np
import tensorstore as ts
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from scipy.stats import zscore
import argparse
from tqdm import tqdm
import subprocess
import cv2

from massive_rnn_train import MassiveRNNModule
from sparse_gru import SparseGRUBrain


def load_model_and_data(checkpoint_path, override_cfg):
    """Load trained model and all necessary data."""
    print(f"Loading checkpoint metadata from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    training_cfg = checkpoint['hyper_parameters']
    
    # Load connectivity graph first (required for model initialization)
    connectivity_path = training_cfg.get('connectivity_path', 'connectivity_graph_pos.pkl')
    print(f"Loading connectivity graph from {connectivity_path}...")
    
    if not os.path.exists(connectivity_path):
        raise FileNotFoundError(f"Connectivity graph not found: {connectivity_path}")
    
    with open(connectivity_path, 'rb') as f:
        connectivity_graph = pickle.load(f)
    
    print(f"Connectivity graph loaded: {len(connectivity_graph)} neurons")
    
    # Extract num_neurons from checkpoint (default to None for old checkpoints)
    num_neurons = training_cfg.get('num_neurons', None)
    print(f"Number of neurons from checkpoint: {num_neurons}")
    
    # Now load the model with the connectivity graph
    print(f"Loading model...")
    model = MassiveRNNModule.load_from_checkpoint(
        checkpoint_path,
        cfg=training_cfg,
        connectivity_graph=connectivity_graph,
        num_neurons=num_neurons
    )
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Model loaded on {device}")
    
    print("\nTraining config loaded from checkpoint:")
    print(f"  normalize_traces: {training_cfg.get('normalize_traces', False)}")
    print(f"  sequence_length: {training_cfg.get('sequence_length', 32)}")
    print(f"  stimulus_dim: {training_cfg.get('stimulus_dim', 10)}")
    print(f"  hidden_dim: {training_cfg.get('hidden_dim', 4)}")
    print(f"  condition_name: {training_cfg.get('condition_name', None)}")
    print(f"  max_neurons: {training_cfg.get('max_neurons', None)}")
    
    # Merge configs: training config as base, override with user preferences
    cfg = {**training_cfg, **override_cfg}
    print(f"\nUsing inference_steps: {cfg['inference_steps']}")
    print(f"Output directory: {cfg['output_dir']}\n")
    
    print(f"Loading traces from {cfg['traces_path']}...")
    ds_traces = ts.open({
        'driver': 'zarr3',
        'kvstore': cfg['traces_path']
    }).result()
    
    if cfg.get('condition_name') is None:
        traces = ds_traces[:, :].read().result()
    else:
        from zapbench import constants, data_utils
        condition_idx = constants.CONDITION_NAMES.index(cfg['condition_name'])
        trace_min, trace_max = data_utils.get_condition_bounds(condition_idx)
        traces = ds_traces[trace_min:trace_max, :].read().result()
    
    if cfg.get('max_neurons'):
        traces = traces[:, :cfg['max_neurons']]
    
    print(f"Loaded traces: {traces.shape}")
    
    if cfg.get('normalize_traces', False):
        print("Z-scoring traces...")
        traces = zscore(traces, axis=0)
        traces = np.nan_to_num(traces)
    
    print(f"Loading stimulus from {cfg['stimulus_path']}...")
    raw_stimulus = np.fromfile(cfg['stimulus_path'], dtype=np.float32).reshape(-1, 10)
    
    n_trace_samples = traces.shape[0]
    n_stim_samples = raw_stimulus.shape[0]
    
    if cfg.get('condition_name') is not None:
        scale_factor = n_stim_samples / ds_traces.shape[0]
        stim_min = int(trace_min * scale_factor)
        stim_max = int(trace_max * scale_factor)
        raw_stimulus = raw_stimulus[stim_min:stim_max, :]
    
    stimulus_tensor = torch.from_numpy(raw_stimulus).float().permute(1, 0).unsqueeze(0)
    resampled_stimulus = torch.nn.functional.interpolate(
        stimulus_tensor, size=n_trace_samples, mode='linear', align_corners=False
    )
    stimulus = resampled_stimulus.squeeze(0).permute(1, 0).numpy()
    print(f"Resampled stimulus to shape: {stimulus.shape}")
    
    segmentation = None
    positions = None
    
    if os.path.exists(cfg['segmentation_path']):
        print(f"Loading segmentation from {cfg['segmentation_path']}...")
        ds_seg = ts.open({
            'driver': 'zarr3',
            'kvstore': f"file://{cfg['segmentation_path']}"
        }).result()
        segmentation = ds_seg.read().result()
        print(f"Segmentation shape: {segmentation.shape}")
    
    if os.path.exists(cfg['neuron_positions_path']):
        print(f"Loading neuron positions from {cfg['neuron_positions_path']}...")
        with open(cfg['neuron_positions_path'], 'rb') as f:
            positions = pickle.load(f)
        print(f"Loaded {len(positions)} neuron positions")
    
    return model, traces, stimulus, segmentation, positions, device, cfg


def select_random_timepoint(traces, inference_steps, sequence_length):
    """Select a random valid timepoint for inference."""
    total_timesteps = traces.shape[0]
    min_idx = sequence_length
    max_idx = total_timesteps - inference_steps
    
    if max_idx <= min_idx:
        raise ValueError(f"Not enough data: need at least {sequence_length + inference_steps} timesteps")
    
    start_idx = np.random.randint(min_idx, max_idx)
    print(f"Selected start index: {start_idx} (valid range: [{min_idx}, {max_idx}))")
    return start_idx


def prepare_inference_input(model, traces, stimulus, start_idx, sequence_length, inference_steps, device):
    """Prepare input for inference with warmup."""
    warmup_traces = traces[start_idx - sequence_length:start_idx, :]
    warmup_stimulus = stimulus[start_idx - sequence_length:start_idx, :]
    inference_stimulus = stimulus[start_idx:start_idx + inference_steps, :]
    ground_truth = traces[start_idx:start_idx + inference_steps, :]
    
    warmup_traces_t = torch.from_numpy(warmup_traces).float().unsqueeze(0).to(device)
    warmup_stimulus_t = torch.from_numpy(warmup_stimulus).float().unsqueeze(0).to(device)
    inference_stimulus_t = torch.from_numpy(inference_stimulus).float().unsqueeze(0).to(device)
    
    print("Running warmup to initialize hidden state...")
    hidden = model.model.init_hidden(1, device=device)
    
    with torch.no_grad():
        for t in range(sequence_length):
            calcium_t = warmup_traces_t[:, t, :]
            stim_t = warmup_stimulus_t[:, t, :]
            _, hidden = model.model(calcium_t, hidden, stim_t)
    
    initial_calcium = warmup_traces_t[:, -1, :]
    
    return initial_calcium, hidden, inference_stimulus_t, ground_truth


def run_inference(model, initial_calcium, initial_hidden, stimulus_seq, inference_steps):
    """Run inference to generate predictions."""
    print(f"Running inference for {inference_steps} timesteps...")
    with torch.no_grad():
        predictions = model.infer(initial_calcium, initial_hidden, stimulus_seq, inference_steps)
    return predictions


def calculate_metrics(ground_truth, predictions):
    """Calculate MSE and MAE metrics."""
    predictions_np = predictions.cpu().numpy().squeeze(0)
    
    mse_per_step = np.mean((ground_truth - predictions_np) ** 2, axis=1)
    mae_per_step = np.mean(np.abs(ground_truth - predictions_np), axis=1)
    
    overall_mse = np.mean(mse_per_step)
    overall_mae = np.mean(mae_per_step)
    
    per_neuron_mse = np.mean((ground_truth - predictions_np) ** 2, axis=0)
    per_neuron_mae = np.mean(np.abs(ground_truth - predictions_np), axis=0)
    
    metrics = {
        'mse_per_step': mse_per_step,
        'mae_per_step': mae_per_step,
        'overall_mse': overall_mse,
        'overall_mae': overall_mae,
        'per_neuron_mse': per_neuron_mse,
        'per_neuron_mae': per_neuron_mae,
    }
    
    print(f"Overall MSE: {overall_mse:.6f}")
    print(f"Overall MAE: {overall_mae:.6f}")
    
    return metrics


def plot_timeseries_metrics(mse_per_step, mae_per_step, output_dir):
    """Plot MSE and MAE over time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(mse_per_step, linewidth=1.5)
    ax1.set_ylabel('MSE')
    ax1.set_title('Mean Squared Error Over Time')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(mae_per_step, linewidth=1.5, color='orange')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('MAE')
    ax2.set_title('Mean Absolute Error Over Time')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/timeseries_metrics.png', dpi=150)
    plt.close()
    print(f"Saved timeseries metrics to {output_dir}/timeseries_metrics.png")


def plot_random_neurons(ground_truth, predictions, num_neurons, output_dir):
    """Plot ground truth vs predictions for random neurons."""
    predictions_np = predictions.cpu().numpy().squeeze(0)
    n_total_neurons = ground_truth.shape[1]
    
    neuron_indices = np.random.choice(n_total_neurons, size=min(num_neurons, n_total_neurons), replace=False)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, neuron_idx in enumerate(neuron_indices):
        if i >= len(axes):
            break
        
        ax = axes[i]
        gt = ground_truth[:, neuron_idx]
        pred = predictions_np[:, neuron_idx]
        
        ax.plot(gt, label='Ground Truth', linewidth=1.5, alpha=0.8)
        ax.plot(pred, label='Predicted', linewidth=1.5, alpha=0.8)
        
        mse = np.mean((gt - pred) ** 2)
        mae = np.mean(np.abs(gt - pred))
        
        ax.set_title(f'Neuron {neuron_idx}\nMSE: {mse:.4f}, MAE: {mae:.4f}')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Activity')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/random_neurons_comparison.png', dpi=150)
    plt.close()
    print(f"Saved random neurons comparison to {output_dir}/random_neurons_comparison.png")


def plot_per_neuron_error_distribution(per_neuron_mse, per_neuron_mae, output_dir):
    """Plot distribution of per-neuron errors."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.hist(per_neuron_mse, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(per_neuron_mse), color='red', linestyle='--', label=f'Mean: {np.mean(per_neuron_mse):.4f}')
    ax1.axvline(np.median(per_neuron_mse), color='green', linestyle='--', label=f'Median: {np.median(per_neuron_mse):.4f}')
    ax1.set_xlabel('MSE')
    ax1.set_ylabel('Number of Neurons')
    ax1.set_title('Per-Neuron MSE Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.hist(per_neuron_mae, bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax2.axvline(np.mean(per_neuron_mae), color='red', linestyle='--', label=f'Mean: {np.mean(per_neuron_mae):.4f}')
    ax2.axvline(np.median(per_neuron_mae), color='green', linestyle='--', label=f'Median: {np.median(per_neuron_mae):.4f}')
    ax2.set_xlabel('MAE')
    ax2.set_ylabel('Number of Neurons')
    ax2.set_title('Per-Neuron MAE Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/per_neuron_error_distribution.png', dpi=150)
    plt.close()
    print(f"Saved per-neuron error distribution to {output_dir}/per_neuron_error_distribution.png")


def create_activity_projection(activity, seg_2d, max_neurons):
    """Project neuron activity onto 2D X-Y plane using pre-computed segmentation."""
    activity_image = np.zeros_like(seg_2d, dtype=np.float32)
    
    seg_2d_clipped = seg_2d.copy()
    if max_neurons is not None:
        seg_2d_clipped[seg_2d > max_neurons] = 0
    
    valid_mask = (seg_2d_clipped > 0) & (seg_2d_clipped <= len(activity))
    valid_indices = seg_2d_clipped[valid_mask] - 1
    activity_image[valid_mask] = activity[valid_indices]
    
    return activity_image


def save_activity_frames(ground_truth, predictions, segmentation, positions, max_neurons, output_dir):
    """Save individual activity frames as images."""
    print("Generating activity frames...")
    predictions_np = predictions.cpu().numpy().squeeze(0)
    num_steps = ground_truth.shape[0]

    # Create nested folder structure
    frames_dir = os.path.join(output_dir, 'frames')
    gt_dir = os.path.join(frames_dir, 'ground_truth')
    pred_dir = os.path.join(frames_dir, 'predicted')
    diff_dir = os.path.join(frames_dir, 'difference')
    combined_dir = os.path.join(frames_dir, 'combined')

    for dir_path in [frames_dir, gt_dir, pred_dir, diff_dir, combined_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Pre-compute 2D segmentation projection once
    seg_2d = np.max(segmentation, axis=2)

    # Calculate global min/max directly from traces for consistent scaling
    vmin = min(ground_truth.min(), predictions_np.min())
    vmax = max(ground_truth.max(), predictions_np.max())

    # Calculate difference range directly from traces
    diff_vmax = max(abs(ground_truth.min() - predictions_np.min()), 
                    abs(ground_truth.max() - predictions_np.max()))

    # Generate and save frames
    for t in tqdm(range(num_steps), desc="Saving frames"):
        gt_frame = create_activity_projection(ground_truth[t], seg_2d, max_neurons)
        pred_frame = create_activity_projection(predictions_np[t], seg_2d, max_neurons)
        diff_frame = gt_frame - pred_frame

        # Normalize frames to 0-255 range for saving
        gt_normalized = ((gt_frame - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        pred_normalized = ((pred_frame - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        diff_normalized = ((diff_frame + diff_vmax) / (2 * diff_vmax) * 255).astype(np.uint8)

        # Apply colormaps
        gt_colored = cv2.applyColorMap(gt_normalized, cv2.COLORMAP_HOT)
        pred_colored = cv2.applyColorMap(pred_normalized, cv2.COLORMAP_HOT)
        diff_colored = cv2.applyColorMap(diff_normalized, cv2.COLORMAP_JET)

        # Save individual frames
        cv2.imwrite(os.path.join(gt_dir, f'frame_{t:04d}.png'), gt_colored)
        cv2.imwrite(os.path.join(pred_dir, f'frame_{t:04d}.png'), pred_colored)
        cv2.imwrite(os.path.join(diff_dir, f'frame_{t:04d}.png'), diff_colored)

        # Create combined frame (horizontal concatenation)
        separator = np.ones((gt_colored.shape[0], 5, 3), dtype=np.uint8) * 255  # White separator
        combined_frame = np.hstack([gt_colored, separator, pred_colored, separator, diff_colored])
        cv2.imwrite(os.path.join(combined_dir, f'frame_{t:04d}.png'), combined_frame)

    print(f"Saved {num_steps} frames to {frames_dir}/")
    return frames_dir


def create_video_from_frames(frames_dir, output_dir, fps=10):
    """Create videos from saved frames using ffmpeg."""
    print("Creating videos with ffmpeg...")

    # Video output paths
    gt_video = os.path.join(output_dir, 'ground_truth.mp4')
    pred_video = os.path.join(output_dir, 'predicted.mp4')
    diff_video = os.path.join(output_dir, 'difference.mp4')
    combined_video = os.path.join(output_dir, 'combined.mp4')

    # ffmpeg commands
    videos_to_create = [
        (os.path.join(frames_dir, 'ground_truth', 'frame_%04d.png'), gt_video),
        (os.path.join(frames_dir, 'predicted', 'frame_%04d.png'), pred_video),
        (os.path.join(frames_dir, 'difference', 'frame_%04d.png'), diff_video),
        (os.path.join(frames_dir, 'combined', 'frame_%04d.png'), combined_video)
    ]

    for input_pattern, output_path in videos_to_create:
        cmd = [
            'ffmpeg', '-y',  # -y to overwrite existing files
            '-r', str(fps),
            '-i', input_pattern,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',  # High quality
            output_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"Created video: {os.path.basename(output_path)}")
        except subprocess.CalledProcessError as e:
            print(f"Error creating {output_path}: {e.stderr}")
        except FileNotFoundError:
            print("ffmpeg not found. Please install ffmpeg to create videos.")
            print("Individual frames are saved in the frames/ directory.")
            return

    print(f"Videos saved to {output_dir}/")


def create_activity_videos(ground_truth, predictions, segmentation, positions, max_neurons, output_dir, fps=10):
    """Create activity videos using frame-based approach with ffmpeg."""
    frames_dir = save_activity_frames(ground_truth, predictions, segmentation, positions, max_neurons, output_dir)
    create_video_from_frames(frames_dir, output_dir, fps)


def save_results_summary(metrics, cfg, output_dir, checkpoint_path):
    """Save a text summary of results."""
    summary_path = f'{output_dir}/results_summary.txt'
    
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("MASSIVE RNN INFERENCE RESULTS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Configuration:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Inference steps: {cfg['inference_steps']}\n")
        f.write(f"Condition: {cfg.get('condition_name', 'All data')}\n")
        f.write(f"Max neurons: {cfg.get('max_neurons', 'All')}\n")
        f.write(f"Normalize traces: {cfg.get('normalize_traces', False)}\n")
        f.write(f"Sequence length: {cfg.get('sequence_length', 32)}\n\n")
        
        f.write("Overall Metrics:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Overall MSE: {metrics['overall_mse']:.6f}\n")
        f.write(f"Overall MAE: {metrics['overall_mae']:.6f}\n\n")
        
        f.write("Per-Step Statistics:\n")
        f.write("-" * 60 + "\n")
        f.write(f"MSE - Mean: {metrics['mse_per_step'].mean():.6f}, Std: {metrics['mse_per_step'].std():.6f}\n")
        f.write(f"MSE - Min: {metrics['mse_per_step'].min():.6f}, Max: {metrics['mse_per_step'].max():.6f}\n")
        f.write(f"MAE - Mean: {metrics['mae_per_step'].mean():.6f}, Std: {metrics['mae_per_step'].std():.6f}\n")
        f.write(f"MAE - Min: {metrics['mae_per_step'].min():.6f}, Max: {metrics['mae_per_step'].max():.6f}\n\n")
        
        f.write("Per-Neuron Statistics:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Per-Neuron MSE - Mean: {metrics['per_neuron_mse'].mean():.6f}, Median: {np.median(metrics['per_neuron_mse']):.6f}\n")
        f.write(f"Per-Neuron MAE - Mean: {metrics['per_neuron_mae'].mean():.6f}, Median: {np.median(metrics['per_neuron_mae']):.6f}\n")
    
    print(f"Saved results summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Run inference with trained Massive RNN')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--inference_steps', type=int, default=200, help='Number of timesteps to infer')
    parser.add_argument('--num_random_neurons', type=int, default=8, help='Number of neurons to plot')
    parser.add_argument('--output_dir', type=str, default='inference_results', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--video_fps', type=int, default=10, help='Video FPS')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Only specify override config (user preferences, not model/data config)
    override_cfg = {
        'traces_path': 'file:///home/v/proj/zebra/data/traces',
        'stimulus_path': '/home/v/proj/zebra/data/stimuli_raw/stimuli_and_ephys.10chFlt',
        'segmentation_path': '/home/v/proj/zebra/data/segmentation',
        'neuron_positions_path': 'neuron_positions.pkl',
        'inference_steps': args.inference_steps,
        'num_random_neurons': args.num_random_neurons,
        'output_dir': args.output_dir,
        'video_fps': args.video_fps,
    }
    
    os.makedirs(override_cfg['output_dir'], exist_ok=True)
    
    # Load model and get merged config from checkpoint + overrides
    model, traces, stimulus, segmentation, positions, device, cfg = load_model_and_data(
        args.checkpoint, override_cfg
    )
    
    start_idx = select_random_timepoint(traces, cfg['inference_steps'], cfg['sequence_length'])
    
    initial_calcium, initial_hidden, inference_stimulus, ground_truth = prepare_inference_input(
        model, traces, stimulus, start_idx, cfg['sequence_length'], cfg['inference_steps'], device
    )
    
    predictions = run_inference(model, initial_calcium, initial_hidden, inference_stimulus, cfg['inference_steps'])
    
    metrics = calculate_metrics(ground_truth, predictions)
    
    plot_timeseries_metrics(metrics['mse_per_step'], metrics['mae_per_step'], cfg['output_dir'])
    plot_random_neurons(ground_truth, predictions, cfg['num_random_neurons'], cfg['output_dir'])
    plot_per_neuron_error_distribution(metrics['per_neuron_mse'], metrics['per_neuron_mae'], cfg['output_dir'])
    
    if segmentation is not None and positions is not None:
        max_neurons = cfg.get('max_neurons', traces.shape[1])
        create_activity_videos(ground_truth, predictions, segmentation, positions, max_neurons,
                             cfg['output_dir'], cfg['video_fps'])
    else:
        print("Skipping video generation (segmentation or positions not available)")
    
    save_results_summary(metrics, cfg, cfg['output_dir'], args.checkpoint)
    
    print("\nInference complete!")
    print(f"Results saved to {cfg['output_dir']}/")


if __name__ == '__main__':
    main()

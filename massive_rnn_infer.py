#!/usr/bin/env python
"""
Massive RNN Inference Script
Loads a trained model, generates inference sequences, and evaluates performance.
"""

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
import os

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


def create_activity_projection(activity, segmentation, positions, max_neurons):
    """Project 3D neuron activity onto 2D X-Y plane."""
    seg_2d = np.max(segmentation, axis=0)
    activity_image = np.zeros_like(seg_2d, dtype=np.float32)
    
    unique_labels = np.unique(seg_2d)
    unique_labels = unique_labels[unique_labels != 0]
    
    for neuron_id in unique_labels:
        if neuron_id > max_neurons:
            continue
        
        neuron_idx = neuron_id - 1
        if neuron_idx >= len(activity):
            continue
        
        activity_val = activity[neuron_idx]
        mask = seg_2d == neuron_id
        activity_image[mask] = activity_val
    
    return activity_image


def create_activity_video(ground_truth, predictions, segmentation, positions, max_neurons, output_dir, fps=10):
    """Create side-by-side video of ground truth and predicted activity."""
    print("Creating activity videos...")
    predictions_np = predictions.cpu().numpy().squeeze(0)
    num_steps = ground_truth.shape[0]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    vmin = min(ground_truth.min(), predictions_np.min())
    vmax = max(ground_truth.max(), predictions_np.max())
    
    frames = []
    for t in tqdm(range(num_steps), desc="Generating frames"):
        gt_frame = create_activity_projection(ground_truth[t], segmentation, positions, max_neurons)
        pred_frame = create_activity_projection(predictions_np[t], segmentation, positions, max_neurons)
        diff_frame = gt_frame - pred_frame
        
        ax1.clear()
        ax2.clear()
        ax3.clear()
        
        im1 = ax1.imshow(gt_frame, cmap='hot', vmin=vmin, vmax=vmax)
        ax1.set_title(f'Ground Truth (t={t})')
        ax1.axis('off')
        
        im2 = ax2.imshow(pred_frame, cmap='hot', vmin=vmin, vmax=vmax)
        ax2.set_title(f'Predicted (t={t})')
        ax2.axis('off')
        
        diff_vmax = max(abs(diff_frame.min()), abs(diff_frame.max()))
        im3 = ax3.imshow(diff_frame, cmap='seismic', vmin=-diff_vmax, vmax=diff_vmax)
        ax3.set_title(f'Difference (t={t})')
        ax3.axis('off')
        
        if t == 0:
            plt.colorbar(im1, ax=ax1, fraction=0.046)
            plt.colorbar(im2, ax=ax2, fraction=0.046)
            plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
    
    print(f"Saving video with {len(frames)} frames...")
    
    Writer = animation.writers['pillow']
    writer = Writer(fps=fps, bitrate=1800)
    
    anim_fig = plt.figure(figsize=(18, 6))
    im = plt.imshow(frames[0])
    plt.axis('off')
    
    def update(frame_idx):
        im.set_array(frames[frame_idx])
        return [im]
    
    anim = animation.FuncAnimation(anim_fig, update, frames=len(frames), interval=1000/fps, blit=True)
    anim.save(f'{output_dir}/activity_comparison.gif', writer=writer)
    plt.close(anim_fig)
    plt.close(fig)
    
    print(f"Saved activity video to {output_dir}/activity_comparison.gif")


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
        create_activity_video(ground_truth, predictions, segmentation, positions, max_neurons, 
                            cfg['output_dir'], cfg['video_fps'])
    else:
        print("Skipping video generation (segmentation or positions not available)")
    
    save_results_summary(metrics, cfg, cfg['output_dir'], args.checkpoint)
    
    print("\nInference complete!")
    print(f"Results saved to {cfg['output_dir']}/")


if __name__ == '__main__':
    main()

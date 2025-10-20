#!/usr/bin/env python
"""
Massive RNN Training Script
Trains a sparse GRU network on all neuron calcium traces simultaneously
using connectivity graphs from either position-based or cross-correlation analysis.

USAGE:
------
1. First, ensure you have a connectivity graph by running one of:
   - python pos_based_con.py  (for position-based connectivity)
   - python cross_corr3.py    (for cross-correlation connectivity)

2. Test the setup:
   python test_massive_rnn.py

3. Start training:
   python massive_rnn_train.py

CONFIGURATION:
--------------
Key parameters to adjust in the cfg dict in main():
- connectivity_path: Path to your connectivity graph pickle file
- max_neurons: Start with 100-1000 for testing, then set to None for all
- batch_size: Reduce if you get OOM errors (try 16, 8, or 4)
- sequence_length: Shorter sequences use less memory (try 16 or 32)
- hidden_dim: Smaller dimensions use less memory (try 16 or 32)

MEMORY TIPS:
------------
If you run out of memory:
1. Reduce batch_size
2. Reduce sequence_length
3. Reduce hidden_dim
4. Use gradient_accumulation_steps (e.g., 2, 4, or 8)
5. Use fewer neurons with max_neurons

Gradient accumulation simulates larger batch sizes:
- batch_size=8 with accumulate_grad_batches=4 → effective batch_size=32
- Reduces memory usage while maintaining training stability

MONITORING:
-----------
The script uses Weights & Biases for logging. Make sure to run:
   wandb login
before training, or set WANDB_MODE=offline for local logging.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import numpy as np
import tensorstore as ts
import pickle
import os
import random
from scipy.stats import zscore
from tqdm import tqdm
import wandb
import argparse
from pprint import pprint
from datetime import datetime

import bitsandbytes as bnb

# Import our sparse GRU model
from sparse_gru import SparseGRUBrain

def parse_param_value(key, value_str, cfg):
    """Parse a parameter value string to the appropriate type."""
    if value_str == 'None':
        return None
    if value_str == 'True':
        return True
    if value_str == 'False':
        return False
    
    # If key exists in cfg, try to match its type
    if key in cfg:
        original_type = type(cfg[key])
        if original_type == bool:
            return value_str.lower() in ('true', '1', 'yes')
        if original_type == int:
            return int(value_str)
        if original_type == float:
            return float(value_str)
        if cfg[key] is None:
            # Original was None, infer from string
            try:
                return int(value_str)
            except ValueError:
                try:
                    return float(value_str)
                except ValueError:
                    return value_str
    
    # Try to infer type from string
    try:
        if '.' in value_str:
            return float(value_str)
        return int(value_str)
    except ValueError:
        return value_str

def parse_params(params_list, cfg):
    """Parse --params arguments and return dict of overrides."""
    overrides = {}
    for param in params_list:
        if '=' not in param:
            raise ValueError(f"Invalid param format: {param}. Expected key=value")
        key, value_str = param.split('=', 1)
        overrides[key] = parse_param_value(key, value_str, cfg)
    return overrides

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)


class CalciumTracesDataset(Dataset):
    """Dataset for calcium traces with stimulus."""
    
    def __init__(self, traces, stimulus, sequence_length, noise_std=0.0, apply_clip=True):
        """
        Args:
            traces: (n_timesteps, n_neurons) array of calcium traces
            stimulus: (n_timesteps, stimulus_dim) array of stimulus data
            sequence_length: length of input sequences
            noise_std: standard deviation of Gaussian noise to add (default 0.0)
            apply_clip: whether to clip to non-negative values (default True)
        """
        self.traces = torch.from_numpy(traces).float()
        self.stimulus = torch.from_numpy(stimulus).float()
        self.sequence_length = sequence_length
        self.noise_std = noise_std
        self.apply_clip = apply_clip
        
        self.n_timesteps, self.n_neurons = self.traces.shape
        self.n_samples = self.n_timesteps - sequence_length
        
        print(f"Dataset initialized: {self.n_samples} samples, {self.n_neurons} neurons")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Get sequence window
        start_idx = idx
        end_idx = start_idx + self.sequence_length
        
        # Extract clean sequences
        calcium_seq_clean = self.traces[start_idx:end_idx, :]  # (seq_len, n_neurons)
        stimulus_seq = self.stimulus[start_idx:end_idx, :]  # (seq_len, stimulus_dim)
        
        # Create noisy version
        calcium_seq_noisy = calcium_seq_clean
        if self.noise_std > 0:
            calcium_seq_noisy = calcium_seq_noisy + torch.randn_like(calcium_seq_noisy) * self.noise_std

        if self.apply_clip:
            calcium_seq_clean = torch.clamp(calcium_seq_clean, min=0)
            calcium_seq_noisy = torch.clamp(calcium_seq_noisy, min=0)
        
        # Return: noisy input, stimulus, clean target
        return calcium_seq_noisy, stimulus_seq, calcium_seq_clean


class CalciumDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for calcium traces."""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.connectivity_graph = None
        self.num_neurons = None
        
    def setup(self, stage=None):
        """Load and prepare data."""
        # Skip if already set up
        if hasattr(self, 'train_dataset') and self.train_dataset is not None:
            return
        
        print("Loading data...")
        
        # Load connectivity graph
        if not os.path.exists(self.cfg['connectivity_path']):
            raise FileNotFoundError(f"Connectivity graph not found at {self.cfg['connectivity_path']}")
        
        print(f"Loading connectivity graph from {self.cfg['connectivity_path']}...")
        with open(self.cfg['connectivity_path'], 'rb') as f:
            self.connectivity_graph = pickle.load(f)
        
        # Print connectivity graph info
        if isinstance(self.connectivity_graph, dict):
            n_post_synaptic = len(self.connectivity_graph)
            total_connections = sum(len(v) for v in self.connectivity_graph.values())
            print(f"Loaded connectivity graph: {n_post_synaptic} post-synaptic neurons, {total_connections} total connections")
        
        # Load traces
        print(f"Loading traces from {self.cfg['traces_path']}...")
        ds_traces = ts.open({
            'driver': 'zarr3',
            'kvstore': self.cfg['traces_path']
        }).result()
        
        # Load all traces or subset based on condition
        if self.cfg.get('condition_name') is None:
            traces = ds_traces[:, :].read().result()
        else:
            # Import zapbench if using conditions
            from zapbench import constants, data_utils
            condition_idx = constants.CONDITION_NAMES.index(self.cfg['condition_name'])
            trace_min, trace_max = data_utils.get_condition_bounds(condition_idx)
            traces = ds_traces[trace_min:trace_max, :].read().result()
        
        # Limit number of neurons if specified
        if self.cfg.get('max_neurons'):
            traces = traces[:, :self.cfg['max_neurons']]
            print(f"Limited to {self.cfg['max_neurons']} neurons")
        
        self.num_neurons = traces.shape[1]
        print(f"Loaded traces: {traces.shape}")
        
        # Z-score normalize traces
        if self.cfg.get('normalize_traces', True):
            print("Z-scoring traces...")
            traces = zscore(traces, axis=0)
            traces = np.nan_to_num(traces)
        
        # Load and resample stimulus data
        print(f"Loading stimulus from {self.cfg['stimulus_path']}...")
        raw_stimulus = np.fromfile(self.cfg['stimulus_path'], dtype=np.float32).reshape(-1, 10)
        
        # Resample stimulus to match traces
        n_trace_samples = traces.shape[0]
        n_stim_samples = raw_stimulus.shape[0]
        
        if self.cfg.get('condition_name') is not None:
            # Scale stimulus indices to match condition
            scale_factor = n_stim_samples / ds_traces.shape[0]
            stim_min = int(trace_min * scale_factor)
            stim_max = int(trace_max * scale_factor)
            raw_stimulus = raw_stimulus[stim_min:stim_max, :]
        
        # Resample stimulus using interpolation
        stimulus_tensor = torch.from_numpy(raw_stimulus).float().permute(1, 0).unsqueeze(0)
        resampled_stimulus = torch.nn.functional.interpolate(
            stimulus_tensor, size=n_trace_samples, mode='linear', align_corners=False
        )
        stimulus = resampled_stimulus.squeeze(0).permute(1, 0).numpy()
        
        print(f"Resampled stimulus to shape: {stimulus.shape}")
        
        # Train/val split
        split_idx = int(n_trace_samples * self.cfg['train_val_split'])
        train_traces = traces[:split_idx, :]
        val_traces = traces[split_idx:, :]
        train_stimulus = stimulus[:split_idx, :]
        val_stimulus = stimulus[split_idx:, :]
        
        # Create datasets
        # Train dataset gets noise, validation doesn't
        self.train_dataset = CalciumTracesDataset(
            train_traces, train_stimulus,
            self.cfg['sequence_length'],
            noise_std=self.cfg.get('noise_std', 0.0),
            apply_clip=self.cfg.get('nonzero_calc', True)
        )
        self.val_dataset = CalciumTracesDataset(
            val_traces, val_stimulus,
            self.cfg['sequence_length'],
            noise_std=0.0,  # No noise for validation
            apply_clip=self.cfg.get('nonzero_calc', True)
        )
        
        print(f"Train samples: {len(self.train_dataset)}, Val samples: {len(self.val_dataset)}")
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg['batch_size'],
            shuffle=True,
            num_workers=self.cfg.get('num_workers', 4),
            prefetch_factor=self.cfg.get('prefetch_factor', 2),
            pin_memory=self.cfg.get('pin_memory', True),
            persistent_workers=True if self.cfg.get('num_workers', 4) > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg['batch_size'],
            shuffle=False,
            num_workers=self.cfg.get('num_workers', 4),
            prefetch_factor=self.cfg.get('prefetch_factor', 2),
            pin_memory=self.cfg.get('pin_memory', True),
            persistent_workers=True if self.cfg.get('num_workers', 4) > 0 else False
        )


class MassiveRNNModule(pl.LightningModule):
    """PyTorch Lightning module for training the sparse GRU network."""
    
    def __init__(self, cfg, connectivity_graph, num_neurons=None):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.save_hyperparameters({'num_neurons': num_neurons})
        self.num_neurons = num_neurons
        self.use_gradient_checkpointing = cfg.get('use_gradient_checkpointing', False)
        
        # Create the sparse GRU model from connectivity graph
        # Determine if connectivity graph uses 0-indexed or 1-indexed neuron IDs
        # Graphs from pos_based_con.py and cross_corr3.py typically use 1-indexed IDs
        assume_zero_indexed = cfg.get('assume_zero_indexed', False)
        
        self.model = SparseGRUBrain.from_connectivity_graph(
            connectivity_graph,
            hidden_dim=cfg['hidden_dim'],
            stimulus_dim=cfg.get('stimulus_dim', 10),  # Default 10 for standard stimulus
            include_self_connections=cfg.get('include_self_connections', True),
            min_strength=cfg.get('min_connection_strength', 0.0),
            bias=cfg.get('use_bias', True),
            num_neurons=self.num_neurons,  # Use the actual number of neurons from data
            assume_zero_indexed=assume_zero_indexed,
            shared_stim_proj=cfg.get('shared_stim_proj', False)
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
        # Training settings
        self.teacher_forcing_ratio = cfg.get('teacher_forcing_ratio', 1.0)
        self.autoregressive_steps = cfg.get('autoregressive_val_steps', 5)
        
    def forward(self, calcium_seq, stimulus_seq):
        """Forward pass through the sequence."""
        batch_size, seq_len, _ = calcium_seq.shape
        hidden = self.model.init_hidden(batch_size, device=self.device)
        
        outputs = []
        
        for t in range(seq_len):
            calcium_t = calcium_seq[:, t, :]
            stimulus_t = stimulus_seq[:, t, :] if stimulus_seq is not None else None
            
            # Forward through GRU
            calcium_pred, hidden = self.model(calcium_t, hidden, stimulus_t)
            outputs.append(calcium_pred)
        
        # Stack predictions
        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, n_neurons)
        return outputs
    
    def training_step(self, batch, batch_idx):
        calcium_seq_noisy, stimulus_seq, calcium_seq_clean = batch
        batch_size, seq_len, n_neurons = calcium_seq_noisy.shape
        
        # Initialize hidden state
        hidden = self.model.init_hidden(batch_size, device=self.device)
        
        # Get BPTT chunk size
        bptt_chunk_size = self.hparams.get('bptt_chunk_size', 0)
        
        # Teacher forcing training with optional BPTT
        losses = []
        for t in range(seq_len - 1):
            # Truncated BPTT: detach hidden state periodically
            if bptt_chunk_size > 0 and t > 0 and t % bptt_chunk_size == 0:
                hidden = hidden.detach()
            
            calcium_t = calcium_seq_noisy[:, t, :]  # Noisy input
            stimulus_t = stimulus_seq[:, t, :]
            target_t = calcium_seq_clean[:, t + 1, :]  # Clean target (next timestep)
            
            # Forward pass with optional gradient checkpointing
            if self.use_gradient_checkpointing:
                calcium_pred, hidden = checkpoint(
                    self.model, calcium_t, hidden, stimulus_t,
                    use_reentrant=False
                )
            else:
                calcium_pred, hidden = self.model(calcium_t, hidden, stimulus_t)
            
            # Compute loss and store
            loss = self.mse_loss(calcium_pred, target_t)
            losses.append(loss)
        
        # Average loss over sequence
        avg_loss = torch.stack(losses).mean()
        
        # Logging
        self.log('train_loss', avg_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return avg_loss
    
    def validation_step(self, batch, batch_idx):
        calcium_seq_noisy, stimulus_seq, calcium_seq_clean = batch
        batch_size, seq_len, n_neurons = calcium_seq_clean.shape
        
        # Initialize hidden state
        hidden = self.model.init_hidden(batch_size, device=self.device)
        
        # Teacher forcing evaluation (use clean inputs for validation)
        tf_losses = []
        mae_losses = []
        for t in range(seq_len - 1):
            calcium_t = calcium_seq_clean[:, t, :]  # Clean input for validation
            stimulus_t = stimulus_seq[:, t, :]
            target_t = calcium_seq_clean[:, t + 1, :]  # Clean target
            
            calcium_pred, hidden = self.model(calcium_t, hidden, stimulus_t)
            tf_losses.append(self.mse_loss(calcium_pred, target_t))
            mae_losses.append(self.mae_loss(calcium_pred, target_t))
        
        avg_tf_loss = torch.stack(tf_losses).mean()
        avg_mae = torch.stack(mae_losses).mean()
        
        # Autoregressive evaluation (predict multiple steps ahead)
        if self.autoregressive_steps > 1 and seq_len > self.autoregressive_steps:
            hidden_ar = self.model.init_hidden(batch_size, device=self.device)
            calcium_ar = calcium_seq_clean[:, 0, :]  # Start from first timestep
            
            ar_loss = 0
            for t in range(min(self.autoregressive_steps, seq_len - 1)):
                stimulus_t = stimulus_seq[:, t, :]
                target_t = calcium_seq_clean[:, t + 1, :]
                
                # Predict next step
                calcium_ar, hidden_ar = self.model(calcium_ar, hidden_ar, stimulus_t)
                ar_loss += self.mse_loss(calcium_ar, target_t)
            
            avg_ar_loss = ar_loss / min(self.autoregressive_steps, seq_len - 1)
            self.log('val_ar_loss', avg_ar_loss, on_epoch=True, prog_bar=False, logger=True)
        
        # Logging
        self.log('val_loss', avg_tf_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mae', avg_mae, on_epoch=True, prog_bar=True, logger=True)
        
        return avg_tf_loss
    
    def infer(self, initial_calcium, initial_hidden, stimulus_seq, num_steps):
        """
        Autoregressively generate neuron activity for a given number of timesteps.
        
        Args:
            initial_calcium: (batch_size, n_neurons) - Initial calcium values
            initial_hidden: (batch_size, n_neurons, hidden_dim) - Initial hidden state
            stimulus_seq: (batch_size, num_steps, stimulus_dim) - Stimulus sequence for all timesteps
            num_steps: int - Number of timesteps to generate
            
        Returns:
            predictions: (batch_size, num_steps, n_neurons) - Predicted calcium activity over time
        """
        batch_size = initial_calcium.shape[0]
        n_neurons = initial_calcium.shape[1]
        
        predictions = torch.zeros(batch_size, num_steps, n_neurons, device=self.device)
        current_calcium = initial_calcium
        current_hidden = initial_hidden
        
        for t in tqdm(range(num_steps), 'Inference'):
            stimulus_t = stimulus_seq[:, t, :] if stimulus_seq is not None else None
            calcium_pred, hidden_new = self.model(current_calcium, current_hidden, stimulus_t)
            predictions[:, t, :] = calcium_pred
            current_calcium = calcium_pred
            current_hidden = hidden_new
        
        return predictions
    
    def configure_optimizers(self):
        # Optimizer
        use_8bit = self.hparams.get('use_8bit_optimizer', False)
        
        if use_8bit:
            print("Using 8-bit AdamW optimizer")
            optimizer = bnb.optim.AdamW8bit(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.get('weight_decay', 1e-5)
            )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.get('weight_decay', 1e-5)
            )
        
        # Learning rate scheduler
        scheduler_type = self.hparams.get('scheduler_type', 'cosine')
        lr_interval = self.hparams.get('lr_interval', 'step')
        
        if scheduler_type == 'cosine':
            # Get total training steps from trainer
            total_steps = self.trainer.estimated_stepping_batches
            warmup_steps = self.hparams.get('warmup_steps', 0)
            
            print(f"LR Scheduler: total_steps={total_steps}, warmup_steps={warmup_steps}, interval={lr_interval}")
            
            if warmup_steps > 0:
                # Cosine annealing with warmup using SequentialLR
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.01,  # Start at 1% of base LR
                    end_factor=1.0,     # Reach 100% of base LR
                    total_iters=warmup_steps
                )
                cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=total_steps - warmup_steps,
                    eta_min=self.hparams.get('min_lr', 1e-6)
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_steps]
                )
            else:
                # Standard cosine annealing without warmup
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=total_steps,
                    eta_min=self.hparams.get('min_lr', 1e-6)
                )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': lr_interval
                }
            }
        elif scheduler_type == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=3,
                min_lr=self.hparams.get('min_lr', 1e-6)
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch'
                }
            }
        else:
            return optimizer


def main():
    """Main training function."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train Massive RNN on calcium traces')
    parser.add_argument('--params', nargs='*', default=[], help='Override cfg params: key1=value1 key2=value2')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Configuration
    cfg = {
        # Data paths
        'traces_path': 'file:///home/v/proj/zebra/data/traces',
        'stimulus_path': '/home/v/proj/zebra/data/stimuli_raw/stimuli_and_ephys.10chFlt',
        'connectivity_path': 'connectivity_graph_global_threshold.pkl',  # or connectivity_graph_pos.pkl
        'assume_zero_indexed': False,  # Set to True if neuron IDs in connectivity graph are 0-indexed
        
        # Data parameters
        'condition_name': None,  # None for all data, or 'turning', 'swimming', etc.
        'max_neurons': None,  # Start with small subset for testing, None for all neurons
        'sequence_length': 16,  # TODO Length of input sequences
        'train_val_split': 0.9,  # Train/validation split ratio
        'normalize_traces': False,  # Whether to z-score normalize traces
        'noise_std': 0.08,  # Standard deviation of Gaussian noise to add to traces
        'nonzero_calc': True,  # Clip calcium traces to non-negative values
        
        # Model parameters
        'hidden_dim': 8,  # Hidden dimension for GRUs
        'stimulus_dim': 10,  # TODO Dimension of stimulus input
        'include_self_connections': True,  # Add self-connections to graph
        'min_connection_strength': 0.52,  # Minimum strength to include connection
        'use_bias': True,  # Use bias in GRU gates
        'shared_stim_proj': True,  # Use single shared stimulus projection (saves 67% params: 5.6M vs 16.8M)
        
        # Training parameters
        'batch_size': 2,  # Batch size
        'accumulate_grad_batches': 2,  # Gradient accumulation steps (1=no accumulation, 2/4/8 for memory savings)
        'learning_rate': 1e-3,  # Initial learning rate
        'weight_decay': 1e-5,  # L2 regularization
        'max_epochs': 10,  # Maximum training epochs
        'teacher_forcing_ratio': 1.0,  # Teacher forcing ratio (1.0 = always use ground truth)
        'autoregressive_val_steps': 1,  # Steps for autoregressive validation
        'use_8bit_optimizer': True,  # Use 8-bit AdamW (saves ~60% optimizer memory, requires bitsandbytes)
        'use_gradient_checkpointing': False,  # Trade compute for memory (40-50% VRAM savings)
        'bptt_chunk_size': 0,  # Truncated BPTT: detach hidden state every N steps (0=disabled, 8-16 recommended)
        'scheduler_type': 'cosine',  # 'cosine' or 'reduce_on_plateau'
        'warmup_steps': 100,  # Number of warmup steps (0 for no warmup)
        'min_lr': 1e-6,  # Minimum learning rate
        'lr_interval': 'step',  # 'step' or 'epoch' - how often to update LR
        
        # Hardware and performance
        'accelerator': 'gpu',  # 'gpu' or 'cpu'
        'devices': 1,  # Number of GPUs
        'strategy': 'auto',  # Training strategy: 'auto', 'ddp', 'deepspeed_stage_2', 'deepspeed_stage_3'
        'precision': '32',  # '16-mixed',  # Mixed precision training
        'gradient_clip_val': 1.0,  # Gradient clipping
        'num_workers': 4,  # DataLoader workers
        'prefetch_factor': 2,  # DataLoader prefetch factor
        'pin_memory': True,  # Pin memory for faster GPU transfer (uses more memory)
        
        # Logging and checkpointing
        'project_name': 'zebra-neuron-forecasting',  # W&B project name
        'experiment_name': None,  # Experiment name
        'checkpoint_dir': 'checkpoints',  # Checkpoint directory
        'log_every_n_steps': 10,  # Logging frequency
        
        # Reproducibility
        'seed': 42,
    }
    
    # Override cfg with command-line params
    if args.params:
        overrides = parse_params(args.params, cfg)
        print(f"Overriding cfg with: {overrides}")
        cfg.update(overrides)
    
    pprint(cfg)
    
    # Set random seed
    set_seed(cfg['seed'])
    
    # Set float32 matmul precision for better performance
    torch.set_float32_matmul_precision('medium')
    
    # Initialize data module
    print("Initializing data module...")
    data_module = CalciumDataModule(cfg)
    data_module.setup()
    
    # Initialize model with actual number of neurons from data
    print("Initializing model...")
    model = MassiveRNNModule(cfg, data_module.connectivity_graph, num_neurons=data_module.num_neurons)
    
    # Initialize logger
    wandb_logger = WandbLogger(
        project=cfg['project_name'],
        name=cfg['experiment_name'],
        config=cfg,
        log_model=False
    )
    
    # Create unique checkpoint directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = f"checkpoints/massive-rnn_{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='massive-rnn-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg['max_epochs'],
        accelerator=cfg['accelerator'],
        devices=cfg['devices'],
        strategy=cfg.get('strategy', 'auto'),
        precision=cfg['precision'],
        gradient_clip_val=cfg['gradient_clip_val'],
        accumulate_grad_batches=cfg['accumulate_grad_batches'],
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        log_every_n_steps=cfg['log_every_n_steps'],
        enable_progress_bar=True,
        enable_model_summary=False,
        val_check_interval=0.2,
    )
    
    # Print training configuration
    effective_batch_size = cfg['batch_size'] * cfg['accumulate_grad_batches']
    print(f"\nTraining Configuration:")
    print(f"  Batch size: {cfg['batch_size']}")
    print(f"  Gradient accumulation steps: {cfg['accumulate_grad_batches']}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Number of neurons: {data_module.num_neurons}")
    print(f"  Sequence length: {cfg['sequence_length']}")
    print(f"  Hidden dimension: {cfg['hidden_dim']}")
    
    # Train the model
    if args.resume:
        print(f"\nResuming training from checkpoint: {args.resume}")
    else:
        print("\nStarting training...")
    trainer.fit(model, data_module, ckpt_path=args.resume)
    
    # Final evaluation
    print("Running final validation...")
    trainer.validate(model, data_module)
    
    print("Training complete!")


if __name__ == '__main__':
    main()

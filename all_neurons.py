import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import numpy as np
import tensorstore as ts
import os
import random
import math
from pytorch_lightning.loggers import WandbLogger
from zapbench import constants, data_utils

# --- Configuration ---
cfg = {
    # Data parameters
    'traces_path': 'file:///home/v/v/zebra/data/traces_zip/traces',
    'embedding_path': 'file:///home/v/v/zebra/data/position_embedding',
    'stimulus_path': '/home/v/v/zebra/data/stimuli_raw/stimuli_and_ephys.10chFlt',
    'condition_name': 'turning',  # Condition to train on
    'sequence_length': 150,
    'prediction_horizon': 1,
    'train_val_split_ratio': 0.8,
    'batch_size': 2048,
    'noise_std': 0.01,
    'max_neurons': 7879,  # Set to a number to limit neurons for testing (e.g., 100)

    # Training parameters
    'seed': 42,
    'max_epochs': 20,
    'accelerator': 'gpu',
    'devices': 1,

    # Model parameters
    'model_params': {
        'input_size': 11,  # 1 (activity) + 192 (embedding) + 10 (stimulus)
        'hidden_size': 16, # Increased hidden size for more complex data
        'num_layers': 2,
        'output_size': 1,
        'num_neurons': None,  # Will be set dynamically based on data
    },

    # Optimizer and scheduler parameters
    'optimizer_params': {
        'lr': 1e-3,
    },
    'scheduler_params': {
        'factor': 0.2,
        'patience': 3,
        'min_lr': 1e-6,
    },
}

# --- Reproducibility ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)

# --- Performance Optimizations ---
torch.set_float32_matmul_precision('medium')

# --- Dataset ---
class AllNeuronsDataset(Dataset):
    def __init__(self, traces, embeddings, stimulus, sequence_length, prediction_horizon, noise_std=0.0):
        self.traces = torch.from_numpy(traces).float()  # Shape: (time, neurons)
        self.embeddings = torch.from_numpy(embeddings).float()  # Shape: (neurons, embed_dim)
        self.stimulus = torch.from_numpy(stimulus).float() # Shape: (time, 10)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.noise_std = noise_std

        self.num_timesteps, self.num_neurons = self.traces.shape
        self.samples_per_neuron = self.num_timesteps - self.sequence_length - self.prediction_horizon + 1

    def __len__(self):
        return self.num_neurons * self.samples_per_neuron

    def __getitem__(self, idx):
        neuron_idx = idx // self.samples_per_neuron
        time_idx = idx % self.samples_per_neuron

        # Get activity sequence and target
        start_idx = time_idx
        end_idx = start_idx + self.sequence_length
        target_idx = end_idx + self.prediction_horizon - 1

        activity_seq = self.traces[start_idx:end_idx, neuron_idx].unsqueeze(-1) # (seq_len, 1)
        stimulus_seq = self.stimulus[start_idx:end_idx, :] # (seq_len, 10)
        target = self.traces[target_idx, neuron_idx].unsqueeze(-1) # (1,)

        # Get the corresponding static embedding
        embedding = self.embeddings[neuron_idx] # (embed_dim,)

        # Add noise to activity if specified
        if self.noise_std > 0:
            activity_seq += torch.randn_like(activity_seq) * self.noise_std

        # Return neuron_idx as well to route to the correct LSTM
        return (activity_seq, embedding, stimulus_seq, neuron_idx), target


# --- DataModule ---
class NeuronDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        # Load full traces to get length
        ds_traces = ts.open({'driver': 'zarr3', 'kvstore': self.cfg['traces_path']}).result()
        num_trace_samples = ds_traces.shape[0]

        # Load raw stimulus data
        raw_stimulus = np.fromfile(self.cfg['stimulus_path'], dtype=np.float32).reshape(-1, 10)
        num_stimulus_samples = raw_stimulus.shape[0]

        # --- Align and Resample Stimulus Data ---
        # Get condition bounds for the low-res ephys clock
        condition_idx = constants.CONDITION_NAMES.index(self.cfg['condition_name'])
        trace_min, trace_max = data_utils.get_condition_bounds(condition_idx)

        # Calculate the high-res stimulus clock bounds
        scale_factor = num_stimulus_samples / num_trace_samples
        stim_min = int(trace_min * scale_factor)
        stim_max = int(trace_max * scale_factor)

        # Slice both datasets to the 'turning' condition
        traces = ds_traces[trace_min:trace_max, :].read().result()
        stimulus_sliced = raw_stimulus[stim_min:stim_max, :]

        # Resample stimulus data to match the number of trace samples
        stimulus_tensor = torch.from_numpy(stimulus_sliced).float().permute(1, 0).unsqueeze(0) # (1, 10, time)
        resampled_stimulus = torch.nn.functional.interpolate(
            stimulus_tensor, size=traces.shape[0], mode='linear', align_corners=False
        )
        stimulus_final = resampled_stimulus.squeeze(0).permute(1, 0).numpy() # (time, 10)

        # Load embeddings
        ds_embed = ts.open({'driver': 'zarr', 'kvstore': self.cfg['embedding_path']}).result()
        embeddings = ds_embed.read().result()

        # Ensure neuron counts match
        num_trace_neurons = traces.shape[1]
        num_embed_neurons = embeddings.shape[0]
        if num_trace_neurons != num_embed_neurons:
            print(f"Warning: Mismatch in neuron count. Traces: {num_trace_neurons}, Embeddings: {num_embed_neurons}")
            min_neurons = min(num_trace_neurons, num_embed_neurons)
            traces = traces[:, :min_neurons]
            embeddings = embeddings[:min_neurons, :]
        
        # Optionally limit the number of neurons for testing
        if self.cfg.get('max_neurons') is not None:
            max_neurons = self.cfg['max_neurons']
            traces = traces[:, :max_neurons]
            embeddings = embeddings[:max_neurons, :]
            print(f"Limiting to {max_neurons} neurons for testing")

        # Split time-series data
        split_idx = int(traces.shape[0] * self.cfg['train_val_split_ratio'])
        train_traces = traces[:split_idx, :]
        val_traces = traces[split_idx:, :]

        # We also need to split the final stimulus data
        train_stimulus = stimulus_final[:split_idx]
        val_stimulus = stimulus_final[split_idx:]


        # Store the number of neurons for the model
        self.num_neurons = train_traces.shape[1]
        print(f"Creating individual LSTMs for {self.num_neurons} neurons")
        
        # Create datasets
        self.train_data = AllNeuronsDataset(train_traces, embeddings, train_stimulus, self.cfg['sequence_length'], self.cfg['prediction_horizon'], self.cfg['noise_std'])
        self.val_data = AllNeuronsDataset(val_traces, embeddings, val_stimulus, self.cfg['sequence_length'], self.cfg['prediction_horizon'])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.cfg['batch_size'], shuffle=True,
                          num_workers=8, pin_memory=True, prefetch_factor=4)    

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.cfg['batch_size'], shuffle=False,
                          num_workers=8, pin_memory=True, prefetch_factor=4)

# --- LightningModule ---
class LSTMForecaster(pl.LightningModule):
    def __init__(self, cfg, num_neurons):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model_params = cfg['model_params']
        self.num_neurons = num_neurons
        
        print(f"Initializing LSTMForecaster with {self.num_neurons} individual LSTMs...")

        # Create one LSTM for each neuron
        self.lstms = nn.ModuleList([
            nn.LSTM(
                input_size=self.model_params['input_size'],
                hidden_size=self.model_params['hidden_size'],
                num_layers=self.model_params['num_layers'],
                batch_first=True,
                dropout=0.2 if self.model_params['num_layers'] > 1 else 0
            )
            for _ in range(self.num_neurons)
        ])
        
        # Create one linear layer for each neuron
        self.linear = nn.Linear(self.model_params['hidden_size'], self.model_params['output_size'])
        
        self.act = nn.ReLU()
        
        print(f"Model initialized with {self.num_neurons} LSTMs and linear layers")

        self.mae_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    @torch.compile
    def process_neuron_batch(self, neuron_activity, neuron_embedding, neuron_stimulus, lstm, linear):
        """Compiled hot path for processing a batch of samples for a single neuron."""
        # Expand embedding to match the sequence length
        embedding_seq = neuron_embedding.unsqueeze(1).expand(-1, neuron_activity.size(1), -1)
        # Concatenate inputs
        x = torch.cat([neuron_activity, neuron_stimulus], dim=2)  # removed neuron_embedding
        # Pass through LSTM
        lstm_out, _ = lstm(x)
        # Extract last timestep and pass through linear layer with activation
        return self.act(linear(lstm_out[:, -1, :]))

    def forward(self, activity_seq, embedding, stimulus_seq, neuron_indices):
        batch_size = activity_seq.size(0)
        predictions = torch.zeros(batch_size, 1, device=activity_seq.device, dtype=torch.float16)
        
        # Process samples by grouping them by neuron for efficiency
        unique_neurons = torch.unique(neuron_indices)
        
        for neuron_idx in unique_neurons:
            # Find all samples belonging to this neuron
            mask = neuron_indices == neuron_idx
            if not mask.any():
                continue
                
            # Get the samples for this neuron
            neuron_activity = activity_seq[mask]
            neuron_embedding = embedding[mask]
            neuron_stimulus = stimulus_seq[mask]
            
            # Use the compiled function for the hot path
            neuron_predictions = self.process_neuron_batch(
                neuron_activity, 
                neuron_embedding, 
                neuron_stimulus,
                self.lstms[neuron_idx], 
                self.linear
            )
            
            # Place predictions in the correct positions
            predictions[mask] = neuron_predictions
        
        return predictions

    def training_step(self, batch, batch_idx):
        (activity_seq, embedding, stimulus_seq, neuron_indices), y = batch
        y_hat = self(activity_seq, embedding, stimulus_seq, neuron_indices)
        loss = self.mae_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (activity_seq, embedding, stimulus_seq, neuron_indices), y = batch
        y_hat = self(activity_seq, embedding, stimulus_seq, neuron_indices)
        mae = self.mae_loss(y_hat, y)
        mse = self.mse_loss(y_hat, y)
        self.log('val_mae', mae, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mse', mse, on_epoch=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.optimizer_params['lr'])
        
        # Cosine Annealing with Warmup
        warmup_steps = 50
        # Estimate total training steps (this is approximate)
        # You may want to calculate this more precisely based on your dataset size
        steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
        total_steps = steps_per_epoch * self.trainer.max_epochs
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine annealing after warmup
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # Update learning rate at each step
                'frequency': 1
            }
        }

# --- Main Execution ---
if __name__ == '__main__':
    set_seed(cfg['seed'])

    # --- W&B Logger ---
    # Make sure to login with `wandb login` in your terminal before running.
    wandb_logger = WandbLogger(project="zebra-neuron-forecasting", log_model="all")
    wandb_logger.experiment.config.update(cfg)

    data_module = NeuronDataModule(cfg)
    # First setup the data module to get the number of neurons
    data_module.setup()
    model = LSTMForecaster(cfg, num_neurons=data_module.num_neurons)

    # Note: torch.compile may not work well with dynamic ModuleList iteration
    # Commenting out for now as it may cause issues with per-neuron LSTM processing
    # model = torch.compile(model)

    trainer = pl.Trainer(
        max_epochs=cfg['max_epochs'],
        accelerator=cfg['accelerator'],
        devices=cfg['devices'],
        precision='16-mixed',
        benchmark=True,
        log_every_n_steps=1,  # Log every step instead of default 50
        logger=wandb_logger,  # Add W&B logger
        callbacks=[
            pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
            # ModelCheckpoint is now managed by WandbLogger when log_model is used
            pl.callbacks.EarlyStopping(monitor='val_mae', patience=cfg['scheduler_params']['patience'] + 2, mode='min')
        ],
    )
    trainer.fit(model, data_module)

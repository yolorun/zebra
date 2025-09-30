import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import numpy as np
import tensorstore as ts
import os
import pickle
import random
import math
import wandb
from pytorch_lightning.loggers import WandbLogger
from zapbench import constants, data_utils

# --- Configuration ---
cfg = {
    # Data parameters
    'traces_path': 'file:///home/v/proj/zebra/data/traces',
    'embedding_path': 'file:///home/v/proj/zebra/data/position_embedding',
    'stimulus_path': '/home/v/proj/zebra/data/stimuli_raw/stimuli_and_ephys.10chFlt',
    'connectivity_path': 'connectivity_graph.pkl',
    'condition_name': 'turning',  # Condition to train on
    'sequence_length': 64,
    'prediction_horizon': 1,
    'train_val_split_ratio': 0.8,
    'batch_size': 256,
    'noise_std': 0.01,
    'max_neurons': 200,  # TODO Set to a number to limit neurons for testing (e.g., 100)

    # Training parameters
    'seed': 42,
    'max_epochs': 20,
    'accelerator': 'gpu',
    'devices': 1,
    'benchmark': False,

    # Model parameters
    'model_params': {
        'hidden_size': 16, # Increased hidden size for more complex data
        'num_layers': 1,
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
    def __init__(self, traces, stimulus, num_modelled_neurons, sequence_length, prediction_horizon, noise_std=0.0):
        self.traces = torch.from_numpy(traces).float()  # Shape: (time, all_neurons)
        self.stimulus = torch.from_numpy(stimulus).float() # Shape: (time, 10)
        self.num_modelled_neurons = num_modelled_neurons
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.noise_std = noise_std

        self.num_timesteps, self.total_num_neurons = self.traces.shape
        self.samples_per_neuron = self.num_timesteps - self.sequence_length - self.prediction_horizon + 1

    def __len__(self):
        # The length is based on the number of neurons we are modeling, not the total number
        return self.num_modelled_neurons * self.samples_per_neuron

    def __getitem__(self, idx):
        neuron_idx = idx // self.samples_per_neuron
        time_idx = idx % self.samples_per_neuron

        start_idx = time_idx
        end_idx = start_idx + self.sequence_length
        target_idx = end_idx + self.prediction_horizon - 1

        # Get the activity slice for ALL neurons in the time window
        all_activity_seq = self.traces[start_idx:end_idx, :] # Shape: [seq_len, num_all_neurons]

        # Get stimulus and the single target value
        stimulus_seq = self.stimulus[start_idx:end_idx, :] # Shape: [seq_len, 10]
        target = self.traces[target_idx, neuron_idx].unsqueeze(-1) # Shape: [1,]

        # Add noise to activity if specified
        if self.noise_std > 0:
            all_activity_seq += torch.randn_like(all_activity_seq) * self.noise_std

        # Return the full activity slice and the target neuron's index
        # The model will be responsible for selecting neighbor activities
        return (all_activity_seq, stimulus_seq, neuron_idx), target


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
        if self.cfg['condition_name'] is None:
            trace_min, trace_max = 0, num_trace_samples
        else:
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

        # Load connectivity graph
        if os.path.exists(self.cfg['connectivity_path']):
            with open(self.cfg['connectivity_path'], 'rb') as f:
                connectivity = pickle.load(f)
            print(f"Loaded connectivity graph from {self.cfg['connectivity_path']}")
        else:
            connectivity = None
            print(f"Warning: Connectivity graph not found at {self.cfg['connectivity_path']}. Proceeding without it.")

        # Ensure neuron counts match
        num_trace_neurons = traces.shape[1]
        num_embed_neurons = embeddings.shape[0]
        if num_trace_neurons != num_embed_neurons:
            print(f"Warning: Mismatch in neuron count. Traces: {num_trace_neurons}, Embeddings: {num_embed_neurons}")
            min_neurons = min(num_trace_neurons, num_embed_neurons)
            traces = traces[:, :min_neurons]
            embeddings = embeddings[:min_neurons, :]
        
        # The number of neurons to model (i.e., to use as prediction targets)
        num_modelled_neurons = self.cfg.get('max_neurons') or traces.shape[1]
        print(f"Modeling {num_modelled_neurons} neurons.")
        # Note: We do NOT slice the traces tensor. It must contain all neurons
        # so that neighbor activity can be looked up.

        # Split time-series data
        split_idx = int(traces.shape[0] * self.cfg['train_val_split_ratio'])
        train_traces = traces[:split_idx, :]
        val_traces = traces[split_idx:, :]

        # We also need to split the final stimulus data
        train_stimulus = stimulus_final[:split_idx]
        val_stimulus = stimulus_final[split_idx:]


        # The number of neurons to be passed to the model is the number of neurons we are modeling.
        self.num_neurons = num_modelled_neurons

        # Create neighbor map for the neurons we are modeling.
        self.neighbor_map = []
        for i in range(num_modelled_neurons):
            neuron_id = i + 1
            if connectivity and neuron_id in connectivity:
                # Get all neighbors from the full connectivity graph.
                # Their data will be available in the full traces tensor.
                neighbors = sorted([neighbor_id for neighbor_id, dist in connectivity[neuron_id]])
                self.neighbor_map.append([n_id - 1 for n_id in neighbors])
            else:
                self.neighbor_map.append([])

        # Create datasets, passing the FULL traces and the number of neurons to model.
        self.train_data = AllNeuronsDataset(train_traces, train_stimulus, num_modelled_neurons, self.cfg['sequence_length'], self.cfg['prediction_horizon'], self.cfg['noise_std'])
        self.val_data = AllNeuronsDataset(val_traces, val_stimulus, num_modelled_neurons, self.cfg['sequence_length'], self.cfg['prediction_horizon'])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.cfg['batch_size'], shuffle=True,
                          num_workers=2, pin_memory=True, prefetch_factor=4)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.cfg['batch_size'], shuffle=False,
                          num_workers=2, pin_memory=True, prefetch_factor=4)

# --- LightningModule ---
class LSTMForecaster(pl.LightningModule):
    def __init__(self, cfg, num_neurons, neighbor_map):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model_params = cfg['model_params']
        self.num_neurons = num_neurons
        self.neighbor_map = neighbor_map

        print(f"Initializing LSTMForecaster with {self.num_neurons} individual LSTMs...")

        # Create one LSTM for each neuron with a dynamic input size
        self.lstms = nn.ModuleList()
        for i in range(self.num_neurons):
            num_neighbors = len(self.neighbor_map[i])
            # input_size = num_neighbors + 1 (self) + 10 (stimulus)
            input_size = num_neighbors + 1 + 10
            self.lstms.append(
                nn.LSTM(
                    input_size=input_size,
                    hidden_size=self.model_params['hidden_size'],
                    num_layers=self.model_params['num_layers'],
                    batch_first=True,
                    dropout=0.2 if self.model_params['num_layers'] > 1 else 0
                )
            )

        self.linear = nn.Linear(self.model_params['hidden_size'], self.model_params['output_size'])
        self.act = nn.ReLU()

        print(f"Model initialized with {self.num_neurons} LSTMs and linear layers")

        self.mae_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, all_activity_seq, stimulus_seq, neuron_indices):
        batch_size = all_activity_seq.size(0)
        # Initialize predictions tensor with the correct dtype and device for mixed precision
        predictions = torch.zeros(batch_size, 1, device=self.device, dtype=self.dtype)

        for neuron_idx_tensor in torch.unique(neuron_indices):
            neuron_idx = neuron_idx_tensor.item()
            mask = (neuron_indices == neuron_idx)
            
            # Get data for the current neuron group
            group_all_activity = all_activity_seq[mask]
            group_stimulus = stimulus_seq[mask]

            # Define the columns to select for this neuron's input
            neighbor_cols = self.neighbor_map[neuron_idx]
            self_col = [neuron_idx]
            input_cols = neighbor_cols + self_col

            # Select activity data for the neuron and its neighbors
            group_specific_activity = group_all_activity[:, :, input_cols]

            # Create the final input tensor for the LSTM
            x = torch.cat([group_specific_activity, group_stimulus], dim=2)

            # Get the correct LSTM for this neuron
            lstm = self.lstms[neuron_idx]

            # Defensive check: Ensure the input size matches the LSTM's expected size
            expected_input_size = lstm.input_size
            actual_input_size = x.shape[2]
            if actual_input_size != expected_input_size:
                raise ValueError(
                    f"Shape mismatch for neuron {neuron_idx}: "
                    f"LSTM expected input_size={expected_input_size}, but got {actual_input_size}."
                )

            # Process the sequence
            lstm_out, _ = lstm(x)
            prediction = self.act(self.linear(lstm_out[:, -1, :]))

            # Store the prediction, ensuring dtypes match
            predictions[mask] = prediction.to(predictions.dtype)

        return predictions

    def training_step(self, batch, batch_idx):
        (all_activity_seq, stimulus_seq, neuron_indices), y = batch
        y_hat = self(all_activity_seq, stimulus_seq, neuron_indices)
        loss = self.mae_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (all_activity_seq, stimulus_seq, neuron_indices), y = batch
        y_hat = self(all_activity_seq, stimulus_seq, neuron_indices)
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


def main():
    # --- W&B Logger ---
    wandb_logger = WandbLogger(project="zebra-neuron-forecasting", log_model="all", config=cfg)

    # Initialize data and model
    data_module = NeuronDataModule(cfg)
    data_module.setup()

    # Pass the neighbor map from the data module to the model
    model = LSTMForecaster(cfg, num_neurons=data_module.num_neurons, neighbor_map=data_module.neighbor_map)

    # Initialize Trainer to get rank information
    trainer = pl.Trainer(
        max_epochs=cfg['max_epochs'],
        accelerator=cfg['accelerator'],
        precision='16-mixed',
        benchmark=cfg['benchmark'],
        log_every_n_steps=1,
        devices=cfg['devices'],
        logger=wandb_logger,
        callbacks=[
            pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
            pl.callbacks.EarlyStopping(monitor='val_mae', patience=cfg['scheduler_params']['patience'] + 2, mode='min')
        ],
    )

    # Start training
    trainer.fit(model, data_module)


# --- Main Execution ---
if __name__ == '__main__':
    main()

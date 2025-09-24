import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import numpy as np
import tensorstore as ts
import os
import random

# --- Configuration ---
cfg = {
    # Data parameters
    'data_path': 'file:///home/v/v/zebra/data/traces_zip/traces',
    'neuron_index': 100,  # Index of the neuron to train on
    'sequence_length': 150,  # Number of time steps in each input sequence
    'prediction_horizon': 1,  # Number of time steps to predict into the future
    'train_val_split_ratio': 0.8,  # 80% for training, 20% for validation
    'batch_size': 512,
    'noise_std': 0.01,  # Standard deviation of Gaussian noise for augmentation

    # Training parameters
    'seed': 42,
    'max_epochs': 20,
    'accelerator': 'gpu',
    'devices': 1,

    # Model parameters
    'model_params': {
        'input_size': 1,
        'hidden_size': 64,
        'num_layers': 2,
        'output_size': 1,
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
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)

# --- Dataset ---
class NeuronActivityDataset(Dataset):
    """Custom PyTorch Dataset for neuron activity time-series."""
    def __init__(self, data, sequence_length, prediction_horizon, noise_std=0.0):
        self.data = torch.from_numpy(data).float().unsqueeze(-1)  # Shape: (time, 1)
        # self.data = np.clip(self.data, 0, None)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.noise_std = noise_std

    def __len__(self):
        return len(self.data) - self.sequence_length - self.prediction_horizon + 1

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = start_idx + self.sequence_length
        target_idx = end_idx + self.prediction_horizon - 1

        input_sequence = self.data[start_idx:end_idx]
        target = self.data[target_idx]

        if self.noise_std > 0:
            # Add Gaussian noise for data augmentation
            noise = torch.randn_like(input_sequence) * self.noise_std
            input_sequence += noise

        return input_sequence, target


# --- DataModule ---
class NeuronDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for handling neuron activity data."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.traces = None
        self.train_data = None
        self.val_data = None

    def setup(self, stage=None):
        # Load the full dataset
        ds_traces_local = ts.open({
            'open': True,
            'driver': 'zarr3',
            'kvstore': self.cfg['data_path']
        }).result()
        self.traces = ds_traces_local.read().result()

        # Extract data for the selected neuron
        neuron_data = self.traces[:, self.cfg['neuron_index']]

        # Split data into training and validation sets
        split_idx = int(len(neuron_data) * self.cfg['train_val_split_ratio'])
        train_raw = neuron_data[:split_idx]
        val_raw = neuron_data[split_idx:]

        # Create datasets
        self.train_data = NeuronActivityDataset(
            train_raw,
            self.cfg['sequence_length'],
            self.cfg['prediction_horizon'],
            noise_std=self.cfg['noise_std']
        )
        self.val_data = NeuronActivityDataset(
            val_raw,
            self.cfg['sequence_length'],
            self.cfg['prediction_horizon']
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.cfg['batch_size'],
            shuffle=True,
            num_workers=os.cpu_count(),
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.cfg['batch_size'],
            shuffle=False,
            num_workers=os.cpu_count(),
            pin_memory=True
        )


# --- LightningModule ---
class LSTMForecaster(pl.LightningModule):
    """PyTorch Lightning module for the LSTM forecasting model."""
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.model_params = self.cfg['model_params']
        self.optimizer_params = self.cfg['optimizer_params']
        self.scheduler_params = self.cfg['scheduler_params']

        self.lstm = nn.LSTM(
            input_size=self.model_params['input_size'],
            hidden_size=self.model_params['hidden_size'],
            num_layers=self.model_params['num_layers'],
            batch_first=True
        )
        self.linear = nn.Linear(self.model_params['hidden_size'], self.model_params['output_size'])
        self.act = nn.ReLU()

        self.mae_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # We only need the output from the last time step
        last_time_step_out = lstm_out[:, -1, :]
        predictions = self.act(self.linear(last_time_step_out))
        return predictions

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.mae_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        mae = self.mae_loss(y_hat, y)
        mse = self.mse_loss(y_hat, y)
        self.log('val_mae', mae, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mse', mse, on_epoch=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.optimizer_params['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            'min',
            factor=self.scheduler_params['factor'], 
            patience=self.scheduler_params['patience'], 
            min_lr=self.scheduler_params['min_lr']
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_mae',
            }
        }


# --- Main Execution ---
if __name__ == '__main__':
    # Set seed for reproducibility
    set_seed(cfg['seed'])

    # Initialize DataModule
    data_module = NeuronDataModule(cfg)

    # Initialize Model
    model = LSTMForecaster(cfg)

    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=cfg['max_epochs'],
        accelerator=cfg['accelerator'],
        devices=cfg['devices'],
        callbacks=[
            pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
            pl.callbacks.ModelCheckpoint(monitor='val_mae', save_top_k=1, mode='min'),
            pl.callbacks.EarlyStopping(monitor='val_mae', patience=cfg['scheduler_params']['patience'] + 2, mode='min')
        ]
    )

    # Train the model
    trainer.fit(model, data_module)

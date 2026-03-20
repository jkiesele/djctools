# import as from djctools.training

import torch
from .module_extensions import flush_all_plotting
from .wandb_tools import wandb_wrapper
import numpy as np
import os
from torch.nn import DataParallel

from typing import Any, Dict, Optional, Sequence, Tuple, Union
from .parallel import make_replicas, check_replicas_equal, train_step_threaded, val_step_threaded



class Trainer:
    """
    Trainer class for multi-GPU training using custom parallel utilities.
    
    This Trainer class handles the initialization, manual batch distribution, 
    and model synchronization across multiple GPUs, allowing flexibility for complex data structures 
    and control over data loading. Compatible with both single and multi-GPU configurations, 
    and can fall back to CPU if no GPU is available or `num_gpus=0` is specified.
    
    Attributes:
        model (torch.nn.Module): The main model for training.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        num_gpus (int): Number of GPUs to use for training. Set to 0 for CPU training.
        device_ids (list of int): List of GPU device IDs to use for training. Defaults to `[0, 1, ..., num_gpus-1]`.
        device (str): The primary device for training, either a specified GPU or 'cpu'.
        verbose_level (int): Controls verbosity of output, with `> 0` printing batch-wise loss updates.
    
    Methods:
        _data_to_device(data, device):
            Recursively moves data (tensors, lists, dictionaries) to the specified device.
        
        create_batches(data_iterator):
            Manually creates and distributes batches across GPUs or CPU from the provided data iterator.
    
        train_loop(train_loader):
            Executes the training loop over one epoch. Handles forward, backward passes, 
            gradient updates, and logging of training losses.
        
        val_loop(val_loader):
            Executes the validation loop, computing and logging validation losses. Runs without gradient updates.
    
        save_model(filepath):
            Saves the model to a file. 
    
        load_model(filepath):
            Loads model from a file. 

        train_batch_callback(model, batch_number, batch_data):
            Callback function that is called after each batch is processed during training.
            The function should take the model, the batch number, and the batch data as arguments.
            This function can be used to perform custom operations on the model or the data after each batch
            and should be implemented by the user through inheritance. Please do not use for logging purposes,
            use the wandb_wrapper.log() function instead.

        val_batch_callback(model, batch_number, batch_data):
            Callback function that is called after each batch is processed during validation.
            The function should take the model, the batch number, and the batch data as arguments.
            This function can be used to perform custom operations on the model or the data after each batch
            and should be implemented by the user through inheritance. Please do not use for logging purposes,
            use the wandb_wrapper.log() function instead.

    
    Example Usage:
    --------------
    >>> model = MyModel() # The model must use LossModule to define the loss(es)
    >>> optimizer = torch.optim.Adam(model.parameters())
    >>> trainer = Trainer(model, optimizer, num_gpus=2, verbose_level=1)
    
    >>> for epoch in range(num_epochs):
    >>>     trainer.train_loop(train_loader)
    >>>     trainer.val_loop(val_loader)
    
    >>> trainer.save_model("model_weights.pth")
    
    Notes:
    ------
    - The `Trainer` class assumes single-process execution. Each batch is moved manually to the correct device,
      allowing full control over batch distribution.
    - This class is optimized for cases where each batch may consist of complex nested structures 
      (e.g., lists of dictionaries or tuples). It can be used with both standard PyTorch data loaders 
      and custom data iterators.
    """
    def __init__(self, model, optimizer, num_gpus=1, device_ids=None, verbose_level=0):
        
        # Initialize distributed process group
        self.num_gpus = num_gpus
        if torch.cuda.is_available() and num_gpus > 0:
            # Set up gpu
            self.device_ids = device_ids if device_ids is not None else list(range(num_gpus))
            self.device = f'cuda:{self.device_ids[0]}'
            self.devices = [f'cuda:{device_id}' for device_id in self.device_ids]
        else:
            # Fall back to CPU if no GPU available or num_gpus is 1
            self.device = 'cpu'
            self.devices = ['cpu']
            self.device_ids = []
            self.num_gpus = 0
            print("Warning: CUDA not available or num_gpus=0. Using CPU.")

        #make devices a Sequence of torch.device objects
        self.devices = [torch.device(d) for d in self.devices]

        # if 'model' is a valid file path and not a torch.nn.Module, load the model from the file
        if isinstance(model, str) and os.path.isfile(model):
            print(f"Loading model from file: {model}")
            model = torch.load(model)
        elif isinstance(model, torch.nn.Module):
            pass
        else:
            raise ValueError("Model must be either a torch.nn.Module or a valid file path to a saved model.")
        
        print(f"Creating replicas using devices: {self.devices}")

        self.model_replicas = make_replicas(model, self.devices)
        self.optimizer = optimizer
        self.verbose_level = verbose_level

    def save_model(self, filepath):
        """
        Saves the model (not just weights) to a file. 

        Args:
            filepath (str): The path to the file where the model weights will be saved.
        """
        torch.save(self.model_replicas[0], filepath) #the first replica is always the master

    def load_model(self, filepath):
        """
        Loads model from a file. slim wrapper

        Args:
            filepath (str): The path to the file from which the model weights will be loaded.
        """
        loaded_model = torch.load(filepath, map_location=self.device)
        self.model_replicas = make_replicas(loaded_model, self.devices)

    @property
    def model(self):
        """
        Returns the master model (the first replica).

        Returns:
            torch.nn.Module: The master model.
        """
        return self.model_replicas[0]
        

    def _data_to_device(self, data, device):
        """
        Moves data to the specified device.

        Args:
            data (tensor, dict, list, or tuple): The data to move.
            device (str): The target device.

        Returns:
            The data moved to the target device, maintaining the same structure.
        """
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, dict):
            return {key: self._data_to_device(value, device) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._data_to_device(item, device) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._data_to_device(item, device) for item in data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def create_batches(self, data_iterator):
        """
        Manually distributes data to each device by loading a batch for each device.

        Args:
            data_iterator: Iterator over the dataset.

        Returns:
            List of batches moved to each device.
        """
        batches = []
        for d in self.devices:
            try:
                data = next(data_iterator)
                data = self._data_to_device(data, d)
                batches.append(data)
            except StopIteration:
                return []  # End of data
        return batches

    def train_loop(self, train_loader):
        """
        Runs the training loop for one epoch.

        Args:
            train_loader (DataLoader): The data loader for training data.
        """
        for r in self.model_replicas:
            r.train()  # Set all replicas to training mode

        self.optimizer.zero_grad()  # Zero gradients before starting the epoch
        data_iterator = iter(train_loader)
        batch_idx = 0

        while True:
            batches = self.create_batches(data_iterator)
            if not batches:
                break  # End of epoch

            info = train_step_threaded(self.model_replicas, self.optimizer, batches, self.devices, check_sync=False)
            flush_all_plotting(self.model_replicas[0])
            self.train_batch_callback(self.model_replicas[0], batch_idx, batches)

            loss = sum([i.loss_value for i in info if i.loss_value is not None])#ok, they have been scaled for this to work before
            # Logging and printing
            wandb_wrapper.log("total_loss", loss)
            if self.verbose_level > 0 and batch_idx % 10 == 0:
                print(f'Batch {batch_idx}: Loss {loss}')
            batch_idx += 1
            wandb_wrapper.flush()

    def val_loop(self, val_loader):
        """
        Runs the validation loop.

        Args:
            val_loader (DataLoader): The data loader for validation data.
        """
        for r in self.model_replicas:
            r.eval()  # Set all replicas to evaluation mode
        data_iterator = iter(val_loader)
        batch_idx = 0

        with torch.no_grad():
            while True:
                batches = self.create_batches(data_iterator)
                if not batches:
                    break  # End of epoch
                
                info = val_step_threaded(self.model_replicas, batches, self.devices)
                flush_all_plotting(self.model_replicas[0])
                self.val_batch_callback(self.model_replicas[0], batch_idx, batches)
                loss = sum([i.loss_value for i in info if i.loss_value is not None]) #ok, they have been scaled for this to work before
    
                # Logging and printing
                wandb_wrapper.log("total_loss", loss)
                if self.verbose_level > 0 and batch_idx % 10 == 0:
                    print(f'Validation Batch {batch_idx}: Loss {loss}')
                batch_idx += 1
                wandb_wrapper.flush(prefix="val_")


    def train_batch_callback(self, model, batch_number, batch_data):
        """
        Callback function that is called after each batch is processed during training.
        The function should take the model, the batch number, and the batch data as arguments.
        This function can be used to perform custom operations on the model or the data after each batch
        and should be implemented by the user through inheritance. Please do not use for logging purposes,
        use the wandb_wrapper.log() function instead.
        """
        pass

    def val_batch_callback(self, model, batch_number, batch_data):
        """
        Callback function that is called after each batch is processed during validation.
        The function should take the model, the batch number, and the batch data as arguments.
        This function can be used to perform custom operations on the model or the data after each batch
        and should be implemented by the user through inheritance. Please do not use for logging purposes,
        use the wandb_wrapper.log() function instead.
        """
        pass
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from .module_extensions import sum_all_losses, clear_all_losses
from .wandb_tools import wandb_wrapper
import numpy as np
import os
from torch.nn import DataParallel

from typing import Any, Dict, Optional, Sequence, Tuple, Union

class _CustomDataParallel(DataParallel):
    def scatter(
        self,
        inputs: list,
        kwargs: Optional[Dict[str, Any]],
        device_ids: Sequence[Union[int, torch.device]],
    ) -> Any:
        """
        Custom scatter method that handles the input structures provided by the Trainer class,
        such as lists of dictionaries or lists of lists of tensors.
        
        Args:
            inputs (Tuple[Any, ...]): The input to be scattered - here this is a tuple with one entry. 
                                      The latter entry is the list mentioned above.
            kwargs (Optional[Dict[str, Any]]): Keyword arguments.
            device_ids (Sequence[Union[int, torch.device]]): Target devices.

        Returns:
            Tuple of scattered inputs and kwargs for each device.
        """
        # Implement custom logic to split `inputs` and `kwargs` based on your structure.
        # Example: if inputs is a list of dicts, scatter each dict entry to the devices.
        
        # Example pseudo-code:
        scattered_inputs = []
        scattered_kwargs = []
        #print('inputs len',len(inputs))
        #print('inputs types',[type(i) for i in inputs])
        ##nested
        #print('inputs[0]',[type(i) for i in inputs[0]])

        for i, device_id in enumerate(device_ids):
            # Create device-specific slices of the input.
            device_input = [ inputs[0][i] ] #wrap in one extra list as in apply_parallel the list is unpacked for some reason, so to mitigate this, we have to wrap it in another list
            device_kwargs = kwargs
            
            scattered_inputs.append(device_input)  # Convert to tuple if needed by forward.
            scattered_kwargs.append(device_kwargs)
        
        return scattered_inputs, tuple(scattered_kwargs)



class Trainer:
    """
    Trainer class for multi-GPU training using PyTorch Distributed Data Parallel (DDP).
    
    This Trainer class handles the initialization of DDP, manual batch distribution, 
    and model synchronization across multiple GPUs, allowing flexibility for complex data structures 
    and control over data loading. Compatible with both single and multi-GPU configurations, 
    and can fall back to CPU if no GPU is available or `num_gpus=0` is specified.
    
    Attributes:
        model (torch.nn.Module): The main model for training, wrapped in DDP if using multiple GPUs.
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
            Saves the model weights to a file. For DDP-wrapped models, uses `model.module.state_dict()`.
    
        load_model(filepath):
            Loads model weights from a file. For DDP-wrapped models, loads weights into `model.module`.
    
        cleanup():
            Cleans up the DDP process group after training. Recommended when using multiple training sessions 
            in a single script to release GPU resources properly.
    
    Example Usage:
    --------------
    >>> model = MyModel() # The model must use LossModule to define the loss(es)
    >>> optimizer = torch.optim.Adam(model.parameters())
    >>> trainer = Trainer(model, optimizer, num_gpus=2, verbose_level=1)
    
    >>> for epoch in range(num_epochs):
    >>>     trainer.train_loop(train_loader)
    >>>     trainer.val_loop(val_loader)
    
    >>> trainer.save_model("model_weights.pth")
    >>> trainer.cleanup()  # Call when using multi-GPU to release resources
    
    Notes:
    ------
    - The `Trainer` class assumes single-process execution. Each batch is moved manually to the correct device,
      allowing full control over batch distribution.
    - For DDP, the model is wrapped with `DistributedDataParallel`, which handles gradient synchronization and
      weight updates across GPUs. Manual gradient averaging is not required.
    - This class is optimized for cases where each batch may consist of complex nested structures 
      (e.g., lists of dictionaries or tuples). It can be used with both standard PyTorch data loaders 
      and custom data iterators.
    - `DistributedDataParallel` uses `nccl` backend by default for multi-GPU setups. If running on a single GPU 
      or CPU, DDP is bypassed, and the model is trained in a standard non-parallel setup.
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

        # Move model to device and wrap with DDP
        model.to(self.device)
        self.model = _CustomDataParallel(model, device_ids=self.device_ids) if len(self.device_ids) > 1 else model
        self.optimizer = optimizer
        self.verbose_level = verbose_level

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
        self.model.train()
        data_iterator = iter(train_loader)
        batch_idx = 0

        while True:
            self.optimizer.zero_grad()
            batches = self.create_batches(data_iterator)
            if not batches:
                break  # End of epoch

            if len(batches) == 1:
                batches = batches[0]

            outputs = self.model(batches)  # DataParallel handles passing data to each GPU
            loss = sum_all_losses(self.model)

            loss.backward()
            self.optimizer.step()
            clear_all_losses(self.model)

            # Logging and printing
            wandb_wrapper.log("total_loss", loss.item())
            if self.verbose_level > 0 and batch_idx % 10 == 0:
                print(f'Batch {batch_idx}: Loss {loss.item()}')
            batch_idx += 1
            wandb_wrapper.flush()

    def val_loop(self, val_loader):
        """
        Runs the validation loop.

        Args:
            val_loader (DataLoader): The data loader for validation data.
        """
        self.model.eval()
        data_iterator = iter(val_loader)
        batch_idx = 0

        with torch.no_grad():
            while True:
                batches = self.create_batches(data_iterator)
                if not batches:
                    break  # End of epoch
                if len(batches) == 1: #no multi gpu
                    batches = batches[0]
    
                outputs = self.model(batches)  # DataParallel handles passing data to each GPU
                loss = sum_all_losses(self.model)
                clear_all_losses(self.model)
    
                # Logging and printing
                wandb_wrapper.log("total_loss", loss.item())
                if self.verbose_level > 0 and batch_idx % 10 == 0:
                    print(f'Validation Batch {batch_idx}: Loss {loss.item()}')
                batch_idx += 1
                wandb_wrapper.flush(prefix="val_")


    def save_model(self, filepath):
        """Saves the model weights to a file."""
        torch.save(self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict(), filepath)

    def load_model(self, filepath):
        """Loads model weights from a file."""
        state_dict = torch.load(filepath)
        self.model.module.load_state_dict(state_dict) if hasattr(self.model, "module") else self.model.load_state_dict(state_dict)

    def cleanup(self):
        pass
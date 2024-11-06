import torch
import copy
from .module_extensions import LossModule
from .wandb_tools import wandb_wrapper


class Trainer:
    def __init__(self, model, optimizer, num_gpus=1, device_ids=None):
        """
        Initializes the Trainer for manual data parallelism.

        Args:
            model (torch.nn.Module): The model to train, containing LossModule instances.
            optimizer (torch.optim.Optimizer): The optimizer for updating the model parameters.
            num_gpus (int): The number of GPUs to use.
            device_ids (list of int, optional): The GPU device IDs to use. If None, uses [0, 1, ..., num_gpus-1].
        """
        # Check CUDA availability before device initialization
        if not torch.cuda.is_available():
            self.device_ids = [0]
            self.devices = ['cpu']
            self.num_gpus = 1
            print("Warning: CUDA is not available. Using CPU instead.")
        else:
            self.num_gpus = num_gpus
            self.device_ids = device_ids if device_ids is not None else list(range(num_gpus))
            self.device_ids = self.device_ids[:num_gpus]  # Ensure the correct number of GPUs
            self.devices = [f'cuda:{device_id}' for device_id in self.device_ids]

        # Move the original model to the first device and create replicas for other devices
        self.models = []
        self.models.append(model.to(self.devices[0]))
        # Only enable logging on the first model
        if hasattr(self.models[0], 'switch_logging'):
            self.models[0].switch_logging(True)

        # Clone the model to other devices
        for i in range(1, len(self.devices)):
            model_clone = self._clone_model(self.models[0], device=self.devices[i])
            if hasattr(model_clone, 'switch_logging'):
                model_clone.switch_logging(False)
            self.models.append(model_clone)

        self.optimizer = optimizer
        self.device = self.devices[0]

    def _clone_model(self, model, device):
        """
        Clones the model to a specified device.

        Args:
            model (torch.nn.Module): The model to clone.
            device (str): The device to move the model to.

        Returns:
            torch.nn.Module: The cloned model on the specified device.
        """
        model_clone = copy.deepcopy(model)
        model_clone = model_clone.to(device)
        return model_clone

    def _data_to_device(self, data, device):
        """
        Moves data to the specified device.

        Args:
            data (list of dicts or tensors): The data to move.
            device (str): The target device.

        Returns:
            list: The data moved to the target device.
        """
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                for key in item:
                    item[key] = item[key].to(device)
            else:
                data[idx] = item.to(device)
        return data

    def create_batches(self, data_iterator):
        batches = []
        # Collect batches for each device
        for device in self.devices:
            try:
                data = next(data_iterator)
                data = self._data_to_device(data, device)
                batches.append(data)
            except StopIteration:
                # If any iterator is exhausted, end the epoch
                return []
        return batches

    def train_loop(self, train_loader):
        """
        Runs the training loop for one epoch.

        Args:
            train_loader (DataLoader): The data loader for training data.
            epoch (int): The current epoch number.
        """
        for model in self.models:
            model.train()

        data_iterator = iter(train_loader)
        batch_idx = 0
        while True:
            # Clear gradients on all models
            self.optimizer.zero_grad()

            batches = self.create_batches(data_iterator)
            if not batches:
                break  # done

            losses = []
            # Run forward and backward passes on each device
            for data_i, model_i in zip(batches, self.models):
                # Clear previous losses
                LossModule.clear_all_losses(model_i)

                # Forward pass
                output = model_i(data_i)

                # Collect losses from LossModule instances
                loss = LossModule.sum_all_losses(model_i)
                losses.append(loss)

            # Launch backward passes asynchronously
            for loss in losses:
                loss.backward()

            # Average gradients manually
            self._average_gradients()

            # Update optimizer on the main model
            self.optimizer.step()

            # Synchronize weights to other models
            self._synchronize_models()

            total_loss = sum([loss.item() for loss in losses]) / self.num_gpus
            #log the total loss
            wandb_wrapper.log("total_loss", total_loss)

            # Optionally, print progress - remove or adjust for actual use
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}: Loss {total_loss}')

            batch_idx += 1

            wandb_wrapper.flush()

    def _average_gradients(self):
        """
        Averages gradients from all models and sets them in the main model.
        """
        # Get parameter lists
        main_params = list(self.models[0].parameters())
        other_params = [list(model.parameters()) for model in self.models[1:]]

        # For each parameter in the main model
        for idx, param in enumerate(main_params):
            if param.grad is None:
                continue  # Skip if no gradient

            # Collect gradients from all models
            grads = [param.grad.data.clone()]
            for params in other_params:
                if params[idx].grad is not None:
                    grads.append(params[idx].grad.data)

            # Average the gradients
            avg_grad = sum(grads) / self.num_gpus

            # Set the averaged gradient in the main model
            param.grad.data = avg_grad

    def _synchronize_models(self):
        """
        Synchronizes the weights from the main model to all other models.
        """
        state_dict = self.models[0].state_dict()
        for i in range(1, self.num_gpus):
            self.models[i].load_state_dict(state_dict)

    def val_loop(self, val_loader):
        """
        Runs the validation loop.

        Args:
            val_loader (DataLoader): The data loader for validation data.
        """
        for model in self.models:
            model.eval()

        data_iterator = iter(val_loader)
        batch_idx = 0
        with torch.no_grad():
            while True:
                batches = self.create_batches(data_iterator)
                if not batches:
                    break  # done

                losses = []
                # Run forward passes on each device
                for data_i, model_i in zip(batches, self.models):
                    # Clear previous losses
                    LossModule.clear_all_losses(model_i)

                    # Forward pass
                    output = model_i(data_i)

                    # Collect losses from LossModule instances
                    loss = LossModule.sum_all_losses(model_i)
                    losses.append(loss)

                # Average the losses
                total_loss = sum([loss.item() for loss in losses]) / self.num_gpus
                
                wandb_wrapper.log("total_loss", total_loss)

                # Optionally, print progress
                if batch_idx % 10 == 0:
                    print(f'Validation Batch {batch_idx}: Loss {total_loss}')

                batch_idx += 1

                wandb_wrapper.flush("val_")

    def save_model(self, filepath):
        """
        Saves the model weights to a file.

        Args:
            filepath (str): The path to save the model.
        """
        torch.save(self.models[0].state_dict(), filepath)

    def load_model(self, filepath):
        """
        Loads model weights from a file.

        Args:
            filepath (str): The path to the saved model.
        """
        state_dict = torch.load(filepath)
        self.models[0].load_state_dict(state_dict)
        self._synchronize_models()

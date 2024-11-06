

# djctools

`djctools` is a package designed to simplify logging, loss management, and multi-GPU training for deep learning models, specifically focusing on nested module structures and irregular data handling. It includes a wrapper for [Weights & Biases (wandb)](https://wandb.ai/) logging and custom logging-capable PyTorch modules, such as `LoggingModule` and `LossModule`, allowing fine-grained control over logging and modular loss calculation within complex model hierarchies. Additionally, it provides a `Trainer` class to facilitate manual data parallelism with irregular data across multiple GPUs, explicitly designed to work seamlessly with the [djcdata data loader](https://github.com/jkiesele/djcdata/blob/master/src/djcdata/torch_interface.py).

## Features

- Singleton wrapper around `wandb` for controlled logging across models.
- `LoggingModule` class with integrated logging capabilities that can be toggled on or off at different levels in the model.
- `LossModule` class for modular and toggled loss calculation, with built-in support for aggregation across multiple loss terms.
- `Trainer` class for manual data parallelism, supporting irregular data and custom data loaders, including compatibility with the `djcdata` data loader.
- Batch logging with buffered flushing to optimize performance.

---

## Installation

After cloning this repository, install the package in editable mode:

```bash
pip install -e .
```

---

## wandb_wrapper

`wandb_wrapper` is a singleton class that controls all `wandb` logging, buffering metrics, and activating/deactivating logging across an entire model. It allows selective initialization of `wandb`, including automatic API key loading from a file (`~/private/wandb_api.sh`).

### Basic Usage

```python
from djctools.wandb_tools import wandb_wrapper

# Initialize wandb (with automatic API key detection)
wandb_wrapper.init(project="my_project")

# Activate or deactivate logging
wandb_wrapper.activate()  # Enable logging
wandb_wrapper.deactivate()  # Disable logging

# Log metrics
wandb_wrapper.log("accuracy", 0.95)
wandb_wrapper.log("loss", 0.1)

# Flush buffered logs to wandb
wandb_wrapper.flush()
```

### API Key Loading

If an API key is not provided, `wandb_wrapper` will attempt to load it from a file located at `~/private/wandb_api.sh`. The file should contain a line like:

```bash
WANDB_API_KEY="your_api_key_here"
```

When the key is found, it is automatically set in the environment for `wandb` to use. This feature allows seamless logging in interactive sessions without exposing sensitive information directly in the code.

---

## LoggingModule

`LoggingModule` is a subclass of `torch.nn.Module` that integrates logging capabilities at the module level. It enables fine-grained control of logging, allowing logging to be toggled on or off for each module or across nested submodules.

### Basic Usage

```python
from djctools.module_extensions import LoggingModule

# Create a logging-enabled module
module = LoggingModule(is_logging_module=True)
module.log("example_metric", 123)  # Logs to wandb_wrapper

# Disable logging for the module
module.switch_logging(False)
module.log("example_metric", 456)  # This will not be logged
```

### Automatic Name Assignment

If no name is provided during initialization, `LoggingModule` automatically assigns unique names like `LoggingModule1`, `LoggingModule2`, etc. These names are used as prefixes for logged metrics, making it easy to identify the source of each log entry.

### Nested Module Logging

`LoggingModule` supports nested logging, meaning that each module can contain submodules with independent logging states. The `switch_logging` method is recursively applied to all `LoggingModule` instances within a parent module, making it easy to toggle logging across complex models.

```python
# Example model with nested LoggingModules
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = LoggingModule(is_logging_module=True)
        self.layer2 = LoggingModule(is_logging_module=False)

# Toggle logging for all LoggingModule instances in a standard torch.nn.Module
LoggingModule.switch_all_logging(MyModel(), enable_logging=True)
```

### Additional Features

- **Logging Prefixes**: Each `LoggingModule` instance prefixes metrics with its name, which is particularly useful for identifying metrics in nested structures.
- **Static Method for Global Logging Control**: The `switch_all_logging` static method allows toggling logging on or off for all `LoggingModule` instances within any `torch.nn.Module` hierarchy.

---

## LossModule

`LossModule` is a subclass of `LoggingModule` that adds functionality for computing, recording, and aggregating loss terms within a model. Each `LossModule` can store its own computed losses, which can later be aggregated across multiple instances within a model. It provides toggling functionality for enabling or disabling specific loss terms, allowing for modular and customizable loss calculations.

### Basic Usage

```python
from djctools.module_extensions import LossModule

# Define a custom loss by subclassing LossModule
class MyCustomLoss(LossModule):
    def compute_loss(self, predictions, targets):
        # Define a specific loss computation, e.g., Mean Squared Error
        return torch.nn.functional.mse_loss(predictions, targets)

# Use the custom loss in a model
model = torch.nn.Module()
model.loss_layer = MyCustomLoss(is_logging_module=True, is_loss_active=True)

# Forward pass with loss calculation
predictions = torch.randn(10, 5)
targets = torch.randn(10, 5)
model.loss_layer(predictions, targets)
```

### Toggling Loss Calculation

You can enable or disable loss calculation for each `LossModule` instance or for all instances within a model. Disabling a loss module bypasses its computation, making it efficient for experiments where only selected loss terms are needed.

```python
# Disable loss calculation
model.loss_layer.switch_loss_calculation(False)
assert not model.loss_layer.is_loss_active

# Enable loss calculation
model.loss_layer.switch_loss_calculation(True)
assert model.loss_layer.is_loss_active
```

### Aggregating Losses

`LossModule` provides static methods to sum and clear all accumulated losses across `LossModule` instances within a model:

```python
# Sum all losses across LossModule instances
total_loss = LossModule.sum_all_losses(model)

# Clear losses after an optimization step
LossModule.clear_all_losses(model)
```

### Additional Features

- **Instance-Level Loss Storage**: Each `LossModule` instance stores its own computed losses, enabling flexible and modular loss management.
- **Static Methods for Global Loss Control**: The `switch_all_losses`, `sum_all_losses`, and `clear_all_losses` methods allow for global management of loss calculations across a model.
- **Read-Only `is_loss_active` Property**: This property provides information on whether a given `LossModule` instance is currently set to compute losses.

This setup provides a comprehensive structure for handling multiple loss terms in deep learning models, supporting modular loss calculation, efficient toggling, and aggregated loss management for complex architectures.

---

## Trainer

The `Trainer` class is designed to facilitate manual data parallelism across multiple GPUs, especially when dealing with irregular data or custom data loaders that are not compatible with PyTorch's built-in data parallelism utilities. It allows you to distribute your model and data across multiple GPUs, perform parallel computations, and manage gradient averaging and model synchronization.

### Key Features

- **Manual Data Parallelism**: Distributes data and model computations manually across multiple GPUs.
- **Irregular Data Handling**: Supports data loaders that yield irregular or variable-sized batches.
- **Compatibility with djcdata Data Loader**: Explicitly designed to work seamlessly with the [`djcdata` data loader](https://github.com/jkiesele/djcdata/blob/master/src/djcdata/torch_interface.py), facilitating efficient training with datasets that produce lists of dictionaries or tensors.
- **Custom Model Integration**: Works with any `torch.nn.Module`, including those using `LossModule` instances.
- **Gradient Averaging**: Averages gradients from all GPUs before performing the optimization step.
- **Model Synchronization**: Synchronizes model weights across all GPUs after each update.
- **Logging Control**: Only enables logging on the main model to prevent redundant logging entries.

### Integration with djcdata Data Loader

The `Trainer` class is explicitly designed to work well with the `djcdata` data loader, which yields data as lists of dictionaries or tensors. This compatibility ensures seamless data handling and efficient training when using the `djcdata` framework for loading and preprocessing your datasets.

- **Data Format**: The `DJCDataLoader` returns data as a list of tensors or a list of dictionaries. Importantly, the length of the list is **not** the batch size; instead, it represents the number of inputs to the model (e.g., a dictionary for features and a dictionary for truth quantities). This structure facilitates the use of `LossModule` instances.
- **Other DataLoaders**: It is not required to use the DJCDataloader from the djcdata package as long as the data loader returns a list of dictionaries or a list of tensors - emphasis is on the list.

### Model Requirements

Your model should:

- Define a `forward` method that accepts data in the format provided by the `djcdata` data loader (typically a list of dictionaries or tensors).
- Use `LossModule` instances to compute and store losses during the forward pass.

### Training and Validation Methods

#### `train_loop(train_loader)`

Runs the training loop. Note that the `Trainer` functions have been updated, and there is no longer an `epoch` parameter in the training function.

- **Parameters**:
  - `train_loader`: Data loader for the training data, compatible with `djcdata`.

#### `validate_loop(val_loader)`

Runs the validation loop.

- **Parameters**:
  - `val_loader`: Data loader for the validation data, compatible with `djcdata`.

### Basic Usage

```python
from djctools.trainer import Trainer
from djcdata.torch_interface import DJCDataLoader

# Define your model using LossModule
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.loss_module = CustomLossModule(is_logging_module=True)
    
    def forward(self, data):
        self.loss_module(data)

# Initialize model and optimizer
model = CustomModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Initialize the Trainer
trainer = Trainer(model, optimizer, num_gpus=2)

# Create data loaders using djcdata
train_loader = DJCDataLoader(data_path='path/to/train_data.djcdc', batch_size=32, shuffle=True)
val_loader = DJCDataLoader(data_path='path/to/val_data.djcdc', batch_size=32)

# Training loop
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch+1}")
    trainer.train_loop(train_loader)
    trainer.validate_loop(val_loader)
```

### Data Handling in the Trainer

- **Data Format**: The `Trainer` expects the data loader to return a list of tensors or a list of dictionaries. The list's length is **not** the batch size but corresponds to the number of inputs to the model.
- **Data Distribution**: The `Trainer` fetches separate batches for each GPU and moves the data to the appropriate device.
- **Data Moving Function**: The `_data_to_device` method in the `Trainer` is designed to handle the data format provided by the `djcdata` data loader.

### Important Notes

- **Trainer Function Updates**: The `Trainer` functions have been updated to reflect the latest changes:
  - `def train_loop(self, train_loader)`
  - `def validate_loop(self, val_loader)`
- **Assumption of Similar Batch Sizes**: The `Trainer` assumes that batches across GPUs are similar in size. If there's significant variation, you might need to adjust the gradient averaging method to weight gradients appropriately.
- **Data List Dimension**: The list dimension in the data returned by the data loader is not the batch size but represents the number of inputs to the model.
- **LossModule Integration**: The data format facilitates the use of `LossModule` instances, allowing for modular loss computation.

### Customization

- **Adjusting Data Handling**: If your data loader provides data in a different format, you may need to adjust the `_data_to_device` method in the `Trainer` class to accommodate your data structure.
- **Error Handling**: Implement additional error handling as needed for your specific use case.
- **Extensibility**: The `Trainer` can be extended with additional features such as learning rate schedulers, checkpointing, early stopping, etc.

### Limitations

- **Communication Overhead**: Manually aggregating gradients and synchronizing models can introduce overhead compared to using PyTorch's built-in `DistributedDataParallel` (DDP). For large-scale applications, consider using DDP.
- **Single Data Loader Iteration**: The `Trainer` fetches data for each GPU by iterating over the data loader multiple times. Ensure that your data loader can handle multiple concurrent iterations if necessary.


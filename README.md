

# djctools

`djctools` is a package designed to simplify logging and loss management for deep learning models, specifically focusing on multi-GPU training and nested module structures. It includes a wrapper for [Weights & Biases (wandb)](https://wandb.ai/) logging and custom logging-capable PyTorch modules, such as `LoggingModule` and `LossModule`, allowing fine-grained control over logging and modular loss calculation within complex model hierarchies.

## Features

- Singleton wrapper around `wandb` for controlled logging across models.
- `LoggingModule` class with integrated logging capabilities that can be toggled on or off at different levels in the model.
- `LossModule` class for modular and toggled loss calculation, with built-in support for aggregation across multiple loss terms.
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

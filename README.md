# djctools

`djctools` is a Python package designed to simplify logging, loss management, and multi-GPU training for deep learning models with complex, nested structures. It includes components that integrate seamlessly with PyTorch and [Weights & Biases (wandb)](https://wandb.ai/), providing tools for:

- Fine-grained control over logging with `LoggingModule`.
- Modular and toggleable loss calculation with `LossModule`.
- Efficient multi-GPU training with `Trainer`, can be used with standard torch `DataLoader` instances, and is additionally optimized for use with irregular data and custom data loaders like from `djcdata`.
- jit-compatibility despite flexible in-model logging functionality.

## General concept

To facilitate these features, the general concept is that the loss functions and metrics calculations are part of the main model (inheriting from `torch.nn.Module`).
For example, the model could be passed the truth targets in addition to the features in training and validation mode:
```
m = MyModel()
out = m([features, targets])
```
However, if all included `LossModule` and `LoggingModule` instances are turned off for pure inference mode without any truth information, the call would become:
```
out = m([features, None])
```
This concept allows for generic training loops, transparent GPU parallelism, and fine grained control over the logging and loss modules.
The details on how this is implemented and should be used can be found below.


## Detailed Features

- **Singleton wandb Wrapper**: Centralized, buffered logging for `wandb`.
- **LoggingModule**: Integrates logging into PyTorch modules with toggleable logging functionality.
- **LossModule**: Modular loss calculation with support for aggregation and toggling specific loss terms.
- **Trainer**: Manual data parallelism, handling irregular data and enabling custom batch distribution for multi-GPU training.
- **Compatibility with djcdata**: Supports custom data loaders, including `djcdata`, which outputs lists of dictionaries or tensors.

---

## Basic Usage Example with MNIST

For the impatient, there is an example using the `Trainer` with a standard dataset like MNIST in the `examples` directory in this repository.


## Installation

```bash
pip install git+https://github.com/jkiesele/djctools
```

### Dependencies

- `torch>=1.8.0`
- `wandb>=0.12.0`
- `djcdata` (optional, will not be installed by default, see https://github.com/jkiesele/djcdata)

---

## wandb_wrapper

`wandb_wrapper` is a singleton class that manages all `wandb` logging for the model. It buffers log metrics, provides a centralized control for logging activation, and initializes `wandb` with optional API key loading from `~/private/wandb_api.sh`.

### Basic Usage

```python
from djctools.wandb_tools import wandb_wrapper

# Initialize wandb
wandb_wrapper.init(project="my_project")

# Activate or deactivate logging
wandb_wrapper.activate()
wandb_wrapper.deactivate()

# Log metrics
wandb_wrapper.log("accuracy", 0.95)
wandb_wrapper.log("loss", 0.1)

# Flush buffered logs to wandb
wandb_wrapper.flush()

# Finish the wandb run
wandb_wrapper.finish()
```

### API Key Loading

If no API key is provided, `wandb_wrapper` will look for a file at `~/private/wandb_api.sh` containing:

```bash
WANDB_API_KEY="your_api_key_here"
```

This feature supports secure logging in interactive sessions without exposing sensitive information in code.

---

## LoggingModule

`LoggingModule` is a subclass of `torch.nn.Module` with integrated logging. The `logging_active` attribute allows you to toggle logging for specific modules or entire model hierarchies.

### Basic Usage

```python
from djctools.module_extensions import LoggingModule

# Create a logging-enabled module
module = LoggingModule(logging_active=True)
module.log("example_metric", 123)  # Logs to wandb_wrapper

# Disable logging for the module
module.switch_logging(False)
module.log("example_metric", 456)  # This will not be logged
```

### Automatic Name Assignment

If no name is provided, `LoggingModule` automatically assigns unique names (`LoggingModule1`, `LoggingModule2`, etc.), which are used as metric prefixes for easy identification.

### Nested Module Logging

`LoggingModule` supports nested logging. Using `switch_logging`, you can toggle logging for all `LoggingModule` instances within a parent module.

```python
# Example model with nested LoggingModules
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = LoggingModule(logging_active=True)
        self.layer2 = LoggingModule(logging_active=False)

# Toggle logging for all LoggingModule instances
model = MyModel()
switch_all_logging(model, enable_logging=True)
```

---

## LossModule

`LossModule`, a subclass of `LoggingModule`, provides modular loss management by allowing each instance to store computed losses, which can be toggled with `loss_active`.

### Basic Usage

```python
from djctools.module_extensions import LossModule

# Define a custom loss by subclassing LossModule
class MyCustomLoss(LossModule):
    def compute_loss(self, predictions, targets):
        '''
        This function will only be called if loss is set to active. 
        '''
        loss = torch.nn.functional.mse_loss(predictions, targets)
        self.log("mse_loss", loss)
        return loss

# Use the custom loss in a model
model = torch.nn.Module()
model.loss_layer = MyCustomLoss(logging_active=True, loss_active=True)

# Forward pass with loss calculation
predictions = torch.randn(10, 5)
targets = torch.randn(10, 5)
model.loss_layer(predictions, targets)
```

### Toggling Loss Calculation

Enable or disable loss calculation with `switch_loss_calculation`:

```python
# Disable loss calculation
model.loss_layer.switch_loss_calculation(False)
assert not model.loss_layer.loss_active

# Enable loss calculation
model.loss_layer.switch_loss_calculation(True)
assert model.loss_layer.loss_active
```

### Aggregating Losses

`module_extensions` includes static methods to manage all logging and losses across instances in a model recursively:

```python
# Sum all losses across LossModule instances
total_loss = sum_all_losses(model)

# Clear losses after an optimization step
clear_all_losses(model)

switch_all_logging(model, False) #disables all logging

switch_all_losses(model, False) #disables all losses
```

This is particularly interesting if a model should be prepared for pure inference mode, and should not depend on truth information anymore.
If all logging and losses are turned off, and the model was configured to use truth information only in logging or loss modules, then 
the truth information fed to the model can be None.

---

## Trainer

The `Trainer` class enables manual data parallelism, distributing computations across multiple GPUs while handling irregular data from custom data loaders, like `djcdata`.

### Key Features

- **Manual Data Parallelism**: Distributes data across multiple GPUs with explicit control over batch distribution.
- **Custom Data Handling**: Compatible with data loaders like `djcdata`, which return lists of dictionaries or tensors.
- **Gradient Averaging**: Averages gradients across GPUs before the optimization step.
- **Model Synchronization**: Syncs model weights across GPUs after updates.


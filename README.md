# djctools

`djctools` is a Python package designed to simplify logging, loss management, and multi-GPU training for deep learning models with complex, nested structures. It includes components that integrate seamlessly with PyTorch and [Weights & Biases (wandb)](https://wandb.ai/), providing tools for:

- Fine-grained control over logging with `LoggingModule`.
- Modular and toggleable loss calculation with `LossModule`.
- Efficient multi-GPU training with `Trainer`, can be used with standard torch `DataLoader` instances, and is additionally optimized for use with irregular data and custom data loaders like from `djcdata`.

## Features

- **Singleton wandb Wrapper**: Centralized, buffered logging for `wandb`.
- **LoggingModule**: Integrates logging into PyTorch modules with toggleable logging functionality.
- **LossModule**: Modular loss calculation with support for aggregation and toggling specific loss terms.
- **Trainer**: Manual data parallelism, handling irregular data and enabling custom batch distribution for multi-GPU training.
- **Compatibility with djcdata**: Supports custom data loaders, including `djcdata`, which outputs lists of dictionaries or tensors.

---

## Installation

After cloning the repository, install `djctools` in editable mode:

```bash
pip install -e .
```

### Dependencies

- `torch>=1.8.0`
- `wandb>=0.12.0`
- `djcdata` (optional, can be installed from GitHub)

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
LoggingModule.switch_all_logging(MyModel(), enable_logging=True)
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
        loss = torch.nn.functional.mse_loss(predictions, targets)
        self.log("mse_loss", loss.item())
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

`LossModule` includes static methods to manage all losses across instances in a model:

```python
# Sum all losses across LossModule instances
total_loss = LossModule.sum_all_losses(model)

# Clear losses after an optimization step
LossModule.clear_all_losses(model)
```

---

## Trainer

The `Trainer` class enables manual data parallelism, distributing computations across multiple GPUs while handling irregular data from custom data loaders, like `djcdata`.

### Key Features

- **Manual Data Parallelism**: Distributes data across multiple GPUs with explicit control over batch distribution.
- **Custom Data Handling**: Compatible with data loaders like `djcdata`, which return lists of dictionaries or tensors.
- **Gradient Averaging**: Averages gradients across GPUs before the optimization step.
- **Model Synchronization**: Syncs model weights across GPUs after updates.

### Compatibility with djcdata Data Loader

The `Trainer` class is designed to work with `djcdata`, which outputs lists of dictionaries or tensors. Importantly, the list dimension does not represent the batch size but the number of inputs to the model.

#### Example of djcdata Data Loader

```python
from djcdata.torch_interface import DJCDataLoader

train_loader = DJCDataLoader(data_path="path/to/data", batch_size=32, shuffle=True, dict_output=True)
```

### Basic Usage Example with MNIST

Here’s an example using the `Trainer` with a standard dataset like MNIST:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from djctools.module_extensions import LossModule
from djctools.trainer import Trainer
from djctools.wandb_tools import wandb_wrapper

# Initialize wandb
wandb_wrapper.init(project="mnist_trainer_example")

# Define a custom LossModule
class MNISTLossModule(LossModule):
    def __init__(self, **kwargs):
        super(MNISTLossModule, self).__init__(**kwargs)
        self.criterion = nn.CrossEntropyLoss()
    
    def compute_loss(self, outputs, targets):
        loss = self.criterion(outputs, targets)
        self.log("train/loss", loss.item())
        return loss

# Define the model
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.loss_module = MNISTLossModule(logging_active=True, loss_active=True)
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 14 * 14, 10)
    
    def forward(self, data):
        inputs, targets = data
        x = inputs
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = x.view(-1, 32 * 14 * 14)
        outputs = self.fc1(x)
        self.loss_module(outputs, targets)
        return outputs

# Training loop
def main():
    batch_size = 64
    num_epochs = 2
    learning_rate = 0.001
    num_gpus = min(torch.cuda.device_count(), 2)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = MNISTModel()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    trainer = Trainer(model, optimizer, num_gpus=num_gpus)

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}")
        trainer.train_loop

(train_loader)
        trainer.validate_loop(val_loader)

    trainer.save_model("mnist_model.pth")
    wandb_wrapper.finish()

if __name__ == "__main__":
    main()
```

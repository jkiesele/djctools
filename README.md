# djctools

`djctools` streamlines logging, loss handling and multi-GPU training for PyTorch models. It integrates with [Weights & Biases](https://wandb.ai/) and keeps models JIT compatible even with complex, nested structures.

## Features

- **Singleton `wandb_wrapper`** – buffered metric logging with optional API key loading.
- **`LoggingModule`** – drop-in base class that lets any submodule log metrics and toggle logging on or off.
- **`LossModule`** – modular loss computation with helpers to sum or clear losses across a model.
- **`PlottingModule`** – optional base class for asynchronous plotting from within a model.
- **`Trainer`** – minimal training loop supporting irregular data structures and manual multi-GPU batch distribution.

## Installation

```bash
pip install git+https://github.com/jkiesele/djctools
```

Dependencies: `torch>=1.8.0`, `wandb>=0.12.0`, and `numpy`. The `djcdata` package is optional and can be installed separately.

## Quick example

```python
import torch
from djctools.module_extensions import LossModule, sum_all_losses

class MyLoss(LossModule):
    def compute_loss(self, pred, target):
        loss = torch.nn.functional.mse_loss(pred, target)
        self.log("mse", loss)
        return loss

model = torch.nn.Module()
model.loss = MyLoss(logging_active=True)

pred = torch.randn(10, 5)
truth = torch.randn(10, 5)
model.loss(pred, truth)

# aggregate losses from all LossModule instances
total_loss = sum_all_losses(model)
```

Training on multiple GPUs:

```python
from djctools.training import Trainer

optimizer = torch.optim.Adam(model.parameters())
trainer = Trainer(model, optimizer, num_gpus=2)
trainer.train_loop(train_loader)
```

## Component overview

### `wandb_wrapper`
A singleton helper that buffers metrics and only contacts wandb when `flush()` is called. It can load the API key from `~/private/wandb_api.sh`.

### `LoggingModule`
Base class providing a `log` method and a `switch_logging` toggle. When logging is disabled the module becomes a no-op, keeping the model JIT friendly.

### `LossModule`
Extends `LoggingModule` with storage for computed losses and utilities such as `sum_all_losses` and `clear_all_losses`. Loss calculation can be disabled dynamically, allowing inference without truth information.

### `PlottingModule`
Caches data during forward passes and spawns a thread to create plots when `flush()` is called.

### `Trainer`
Handles manual batch distribution across devices and works with both standard `DataLoader` objects and irregular structures like lists of dictionaries. See the `examples` directory for an MNIST training script.

## License

This project is licensed under the MIT License.


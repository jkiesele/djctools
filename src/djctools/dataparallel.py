"""
Prototype single-host multi-GPU trainer skeleton for local reasoning.

Goals:
- one master model + optimizer
- one replica per GPU
- uneven batches allowed
- semantics intended to match concatenated global batch, assuming local losses
  are represented as sample-summed contributions before final normalization
- per-forward thread-local loss context
- loss modules disappear cleanly when truth is None

This is intentionally a compact prototype, not production code.
"""

from __future__ import annotations

import copy
import threading
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from djctools.module_extensions import switch_all_logging, LoggingModule


# ---------------------------------------------------------------------------
# Forward context
# ---------------------------------------------------------------------------

class ForwardContext:
    def __init__(self, sample_count=0):
        self.sample_count = sample_count
        self.losses = []
        self._closed = False

    def add_loss(self, value):
        if self._closed:
            raise RuntimeError("Cannot add loss to closed ForwardContext")
        if not isinstance(value, torch.Tensor):
            raise TypeError("Loss must be a torch.Tensor")
        if value.dim() != 0:
            raise ValueError("Loss must be a scalar tensor")
        self.losses.append(value)

    def total_loss(self):
        if not self.losses:
            return None
        out = self.losses[0]
        for l in self.losses[1:]:
            out = out + l
        return out

    def close(self):
        self._closed = True


_tls = threading.local()

def get_current_context():
    return getattr(_tls, "ctx", None)


@contextmanager
def forward_context(sample_count=0):
    if getattr(_tls, "ctx", None) is not None:
        raise RuntimeError("Nested ForwardContext is not allowed")

    ctx = ForwardContext(sample_count=sample_count)
    _tls.ctx = ctx
    try:
        yield ctx
    finally:
        _tls.ctx = None
        ctx.close()





# ---------------------------------------------------------------------------
# Base loss module
# ---------------------------------------------------------------------------



class LossModule(LoggingModule):
    def __init__(self, name=None, logging_active=False, loss_active=True):
        """
        LossModule extends LoggingModule to enable modular loss computation, allowing 
        fine-grained control over loss terms in complex models.

        It is a PyTorch module designed to compute and record individual loss terms, inheriting from LoggingModule.
        This module allows toggling loss calculation on or off for efficient handling of multiple loss terms within
        complex model structures. Each LossModule instance stores its own computed losses, which can later be aggregated
        across a model.
    
        Attributes:
            _losses (list): An instance-level list that stores computed losses for the module, enabling
                            retrieval and aggregation of losses when needed.
            loss_active (bool): A property that returns whether loss calculation is enabled or disabled.
    
        Args:
            name (str, optional): Optional name for the module. If None, a unique name will be assigned.
            logging_active (bool): If True, enables logging for this module.
            loss_active (bool): If True, enables loss calculation for this module. Default is True.
    
        Methods:
            forward(*args, **kwargs): Computes the loss by calling compute_loss and appends it to the instance's loss list.
                                      This method is dynamically reassigned based on the `loss_active` state.
            compute_loss(*args, **kwargs): Should be implemented in subclasses.
                                           Returns a single scalar tensor representing the loss.
            switch_loss_calculation(enable_loss): Enables or disables loss calculation, dynamically assigning forward to
                                                  either compute_loss or a no-op function for JIT compatibility.
            clear_losses(): Clears all recorded losses for the instance, useful for resetting losses after aggregation.
            
            sum_all_losses(module): Recursively collects and sums losses from all LossModule instances within a given module.
                                    Returns the total loss as a single scalar tensor.
            switch_all_losses(module, enable_loss): Recursively enables or disables loss calculation for all LossModule
                                                    instances within a given module.
        """
        super(LossModule, self).__init__(name=name, logging_active=logging_active)
        self.switch_loss_calculation(loss_active)

    @property
    def loss_active(self):
        """Read-only property to access the loss calculation state."""
        return self.forward == self._compute_loss_and_record

    def _compute_loss_and_record(self, *args, **kwargs):
        """Compute the loss and append to the instance's loss list."""
        loss = self.compute_loss(*args, **kwargs)

        if loss is None:
            return None

        ctx = get_current_context()
        if ctx is None:
            raise RuntimeError(
                f"LossModule '{self.__class__.__name__}' returned a loss but no ForwardContext is active"
            )

        ctx.add_loss(loss)
        return loss
    
    def _no_op(self, *args, **kwargs):
        """A no-op function used when loss calculation is disabled."""
        return None

    def compute_loss(self, *args, **kwargs):
        """
        Placeholder for the actual loss computation. Should be implemented in subclasses.
        This function will be called by `forward` when the loss calculation is enabled.

        Must return a single scalar tensor representing the loss.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses of LossModule must implement the compute_loss method.")
    
    def switch_logging(self, logging_active):
        """
        Enables or disables logging for this module and all nested submodules.
        This only affects calls to the `log` method, not the forward method.

        Args:
            logging_active (bool): True to enable logging, False to disable it.
        """
        self.log = self._log if logging_active else self._no_op
        for child in self.children():
            if isinstance(child, LoggingModule): # now these are all nested logging modules
                child.switch_logging(logging_active)

    def switch_loss_calculation(self, loss_active):
        """
        Enables or disables the loss calculation for this module, dynamically setting `forward` to either
        `_compute_loss_and_record` (enabled) or `_no_op` (disabled) for JIT compatibility.

        Args:
            loss_active (bool): True to enable loss calculation, False to disable it.

        Note:
            This method applies recursively to all nested LossModule instances within the module.
        """
        self.forward = self._compute_loss_and_record if loss_active else self._no_op

        # Recursively apply to all child modules
        for child in self.children():
            if isinstance(child, LossModule):
                child.switch_loss_calculation(loss_active)

    def clear_losses(self):
        raise DeprecationWarning("This method should not be used anymore and is not needed.")
    




# ---------------------------------------------------------------------------
# Replica management
# ---------------------------------------------------------------------------

def make_replicas(master_model: nn.Module, devices: Sequence[torch.device]) -> List[nn.Module]:
    """
    Create one replica per device.
    Assumes master_model is already in the desired initial state.
    """
    master_model.to('cpu')  # ensure master starts on CPU for clean copying
    replicas: List[nn.Module] = []
    for i, dev in enumerate(devices):
        if i:
            replica = copy.deepcopy(master_model)
        else:
            replica = master_model
        replica.train(master_model.training)
        if i > 0:
            pass
            #switch_all_logging(replica, False)
        replicas.append(replica)
    
    #move all to device
    for replica, dev in zip(replicas, devices):
        replica.to(dev)

    sync_from_master(replicas)

    # direct parameter check, no state_dict, sanity check
    master_params = dict(replicas[0].named_parameters())
    replica_params = dict(replicas[1].named_parameters())
    
    for name in master_params:
        if not torch.equal(master_params[name].cpu(), replica_params[name].cpu()):
            raise RuntimeError(f"Direct parameter mismatch right after make_replicas for {name}")
    
    return replicas


# FIXME: set to non blocking at some point, and sync at the end of the function
@torch.no_grad()
def sync_from_master_no_working(replicas: Sequence[nn.Module], blocking: bool = True) -> None:
    """
    Copy parameters and buffers from master to all other replicas.
    replicas[0] may be master itself; it is skipped.
    """
    master_params = dict(replicas[0].named_parameters())
    master_buffers = dict(replicas[0].named_buffers())

    for replica in replicas[1:]:
        replica_params = dict(replica.named_parameters())
        replica_buffers = dict(replica.named_buffers())

        for name, p in master_params.items():
            replica_params[name].copy_(p, non_blocking =  not blocking)
            #replica_params[name].data.copy_(p.data, non_blocking=not blocking)
            # check if they are actually the same
            if not torch.equal(replica_params[name].cpu(), p.cpu()): #needs to be on same device
                raise RuntimeError(f"sync_from_master: Parameter '{name}' differs between master and replica after sync_from_master: values: {p} vs {replica_params[name]}, shapes: {p.shape} vs {replica_params[name].shape}")

        for name, b in master_buffers.items():
            replica_buffers[name].copy_(b, non_blocking =  not blocking)
            # check if they are actually the same
            if not torch.equal(replica_buffers[name].cpu(), b.cpu()): #needs to be on same device
                raise RuntimeError(f"sync_from_master: Buffer '{name}' differs between master and replica after sync_from_master: values: {b} vs {replica_buffers[name]}, shapes: {b.shape} vs {replica_buffers[name].shape}")
            
@torch.no_grad()
def sync_from_master(replicas: Sequence[nn.Module]) -> None:
    """
    Copy parameters and buffers from master to all other replicas via CPU staging.
    replicas[0] is the master.
    """
    master_params = dict(replicas[0].named_parameters())
    master_buffers = dict(replicas[0].named_buffers())

    for replica in replicas[1:]:
        replica_params = dict(replica.named_parameters())
        replica_buffers = dict(replica.named_buffers())

        for name, p in master_params.items():
            tmp = p.detach().cpu()
            replica_params[name].data.copy_(tmp.to(replica_params[name].device))
            if not torch.equal(replica_params[name].detach().cpu(), tmp):
                raise RuntimeError(
                    f"Parameter '{name}' differs between master and replica after sync_from_master: "
                    f"values: {p} vs {replica_params[name]}, shapes: {p.shape} vs {replica_params[name].shape}"
                )

        for name, b in master_buffers.items():
            tmp = b.detach().cpu()
            replica_buffers[name].copy_(tmp.to(replica_buffers[name].device))
            if not torch.equal(replica_buffers[name].detach().cpu(), tmp):
                raise RuntimeError(
                    f"Buffer '{name}' differs between master and replica after sync_from_master: "
                    f"values: {b} vs {replica_buffers[name]}, shapes: {b.shape} vs {replica_buffers[name].shape}"
                )

@torch.no_grad()
def check_replicas_equal(replicas: Sequence[nn.Module]) -> None:
    master_params = dict(replicas[0].named_parameters())
    master_buffers = dict(replicas[0].named_buffers())

    for i, replica in enumerate(replicas[1:], start=1):
        replica_params = dict(replica.named_parameters())
        replica_buffers = dict(replica.named_buffers())

        for name, p in master_params.items():
            other = replica_params[name]
            if not torch.equal(p.detach().cpu(), other.detach().cpu()):
                raise RuntimeError(
                    f"Replica {i} differs from master parameter '{name}': "
                    f"values: {p} vs {other}, shapes: {p.shape} vs {other.shape}"
                )

        for name, b in master_buffers.items():
            other = replica_buffers[name]
            if not torch.equal(b.detach().cpu(), other.detach().cpu()):
                raise RuntimeError(
                    f"Replica {i} differs from master buffer '{name}': "
                    f"values: {b} vs {other}, shapes: {b.shape} vs {other.shape}"
                )

# ---------------------------------------------------------------------------
# Local worker step
# ---------------------------------------------------------------------------

@dataclass
class LocalStepInfo:
    batch_size: int
    loss_value: Optional[float]
    ctx: ForwardContext


def _infer_batch_size(x: Any) -> int:
    if hasattr(x, "__len__"):
        return len(x)
    raise TypeError("Cannot infer batch size from input batch")


def local_worker(model: nn.Module, batch: Tuple[Any, Any], device: torch.device) -> LocalStepInfo:
    x, y = batch

    if hasattr(x, "to"):
        x = x.to(device, non_blocking=True)
    if y is not None and hasattr(y, "to"):
        y = y.to(device, non_blocking=True)

    bs = _infer_batch_size(x)

    model.zero_grad(set_to_none=True)

    with forward_context(sample_count=bs) as ctx:
        _ = model(x, truth=y)
        loss = ctx.total_loss()

    if loss is not None:
        loss.backward()

    return LocalStepInfo(
        batch_size=bs,
        loss_value=None if loss is None else float(loss.detach().cpu()),
        ctx=ctx,
    )


def threaded_local_steps(
    replicas: Sequence[nn.Module],
    batches: Sequence[Tuple[Any, Any]],
    devices: Sequence[torch.device],
    max_workers: Optional[int] = None,
) -> List[LocalStepInfo]:
    assert len(replicas) == len(batches) == len(devices)

    with ThreadPoolExecutor(max_workers=max_workers or len(devices)) as pool:
        futures = [
            pool.submit(local_worker, model, batch, dev)
            for model, batch, dev in zip(replicas, batches, devices)
        ]
        infos = [f.result() for f in futures]

    return infos


# ---------------------------------------------------------------------------
# Gradient aggregation and optimizer step
# ---------------------------------------------------------------------------

@torch.no_grad()
def aggregate_grads_to_master(
    replicas: Sequence[nn.Module],
    batch_sizes: Sequence[int],
) -> None:
    """
    Aggregate gradients from all replicas onto replicas[0].

    Assumes:
    - local losses correspond to sample-summed contributions
    - final desired semantics are global mean over all samples
    """
    master = replicas[0]
    total_bs = sum(batch_sizes)
    if total_bs <= 0:
        raise RuntimeError("Total batch size must be > 0")

    master_params = list(master.parameters())
    replica_params = [list(m.parameters()) for m in replicas]

    for p in master_params:
        p.grad = None

    for p_idx, master_p in enumerate(master_params):
        acc = None

        for r_idx in range(len(replicas)):
            g = replica_params[r_idx][p_idx].grad
            if g is None:
                continue

            g_on_master = g.to(master_p.device, non_blocking=True)

            if acc is None:
                acc = g_on_master.clone()
            else:
                acc.add_(g_on_master)

        if acc is None:
            continue

        acc.div_(total_bs)
        master_p.grad = acc


def master_step( optimizer: torch.optim.Optimizer) -> None:
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


# ---------------------------------------------------------------------------
# One full multi-GPU train step
# ---------------------------------------------------------------------------

def train_step_threaded(
    replicas: Sequence[nn.Module],
    optimizer: torch.optim.Optimizer,
    batches: Sequence[Tuple[Any, Any]],
    devices: Sequence[torch.device],
    check_sync: bool = False,
) -> List[LocalStepInfo]:
    infos = threaded_local_steps(replicas, batches, devices)
    batch_sizes = [info.batch_size for info in infos]


    torch.cuda.synchronize()
    aggregate_grads_to_master(replicas, batch_sizes)
    master_step(optimizer)
    torch.cuda.synchronize()
    sync_from_master(replicas)
    torch.cuda.synchronize()

    if check_sync:
        check_replicas_equal(replicas)

    return infos


# ---------------------------------------------------------------------------
# Example toy loss module
# ---------------------------------------------------------------------------

class MSELossModule(LossModule):
    def compute_loss(self, pred, truth=None):
        if truth is None:
            return None
        # sample-summed contribution
        return ((pred - truth) ** 2).sum()


# ---------------------------------------------------------------------------
# Example toy model
# ---------------------------------------------------------------------------

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.aux_loss = MSELossModule("mse")

    def forward(self, x, truth=None):
        pred = self.linear(x)
        self.aux_loss(pred, truth=truth)
        return pred


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

def example_setup(num_gpus: int = 2):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this prototype")

    devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]
    master_model = ToyModel(16, 4).to(devices[0])
    replicas = make_replicas(master_model, devices)
    optimizer = torch.optim.Adam(replicas[0].parameters(), lr=1e-3)

    # check replica sync
    print("Checking replica synchronization... before training")
    check_replicas_equal(replicas)
    print("Replicas are initially synchronized.")

    return replicas, optimizer, devices


def example_batches(devices: Sequence[torch.device]):
    batches = []
    for i, _dev in enumerate(devices):
        bs = 32 + i  # intentionally uneven
        x = torch.randn(bs, 16)
        y = torch.randn(bs, 4)
        batches.append((x, y))
    return batches


if __name__ == "__main__":
    replicas, optimizer, devices = example_setup(num_gpus=2)

    batches = example_batches(devices)

    print("Stepping")

    infos = train_step_threaded(
        replicas=replicas,
        optimizer=optimizer,
        batches=batches,
        devices=devices,
        check_sync=True,
    )

    for i, info in enumerate(infos):
        print(f"Replica {i}: batch_size={info.batch_size}, loss={info.loss_value}")

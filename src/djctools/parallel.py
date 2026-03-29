import copy
from concurrent.futures import ThreadPoolExecutor
import torch
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple
import torch.nn as nn
from .threading_context import ForwardContext, forward_context

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

    #check if devices exist and raise if not already here with proper error
    for d in devices:
        if d.type == 'cuda' and d.index >= torch.cuda.device_count():
            raise RuntimeError(f"Device {d} is not available. Only {torch.cuda.device_count()} CUDA devices detected.")

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
    if len(replicas) <= 1:
        return # nothing to do
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

    # set device context explicitly to avoid multi-GPU race conditions
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    
    x, y = batch

    bs = _infer_batch_size(x)

    model.zero_grad(set_to_none=True)

    with forward_context(sample_count=bs) as ctx:
        _ = model(batch)
        loss = ctx.total_loss()

    if loss is not None:
        loss.backward()

    #print gradients for all parameters in this local worker
    #for name, p in model.named_parameters():
    #    print(f"Local worker on device {device}: parameter '{name}' grad value: {p.grad.detach().cpu() if p.grad is not None else None}")

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


def create_splitbatch_scalers(batch_sizes: Sequence[int]) -> List[float]:
    total_bs = sum(batch_sizes)
    if total_bs <= 0:
        raise RuntimeError("Total batch size must be > 0")
    scalers = [bs / total_bs for bs in batch_sizes]
    return scalers

@torch.no_grad()
def aggregate_grads_to_master(
    replicas: Sequence[nn.Module],
    batch_sizes: Sequence[int],
    scalers: Optional[Sequence[float]] = None,
) -> None:
    """
    Aggregate gradients from all replicas onto replicas[0].

    Assumes:
    - local losses correspond to sample-summed contributions
    - final desired semantics are global mean over all samples
    """
    if len(replicas) != len(batch_sizes):
        raise ValueError("Length of replicas and batch_sizes must match")
    if len(replicas) == 1:
        return [1.0]  # nothing to do
    
    master = replicas[0]
    total_bs = sum(batch_sizes)
    if total_bs <= 0:
        raise RuntimeError("Total batch size must be > 0")

    if scalers is None:
        scalers = create_splitbatch_scalers(batch_sizes)

    master_params = list(master.parameters())
    replica_params = [list(m.parameters()) for m in replicas]

    for p_idx, master_p in enumerate(master_params):
        acc = None

        for r_idx in range(len(replicas)):
            g = replica_params[r_idx][p_idx].grad
            if g is None:
                continue

            g_on_master = g.detach().cpu().to(master_p.device) #room for optimisation here if peer access is available
            g_on_master.mul_(scalers[r_idx])

            if acc is None:
                acc = g_on_master.clone()
            else:
                acc.add_(g_on_master)

        if acc is None:
            continue

        master_p.grad = acc



def master_step( optimizer: torch.optim.Optimizer) -> None:
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


# ---------------------------------------------------------------------------
# One full multi-GPU train step; also works for single GPU or CPU
# ---------------------------------------------------------------------------

def train_step_threaded(
    replicas: Sequence[nn.Module],
    optimizer: torch.optim.Optimizer,
    batches: Sequence[Tuple[Any, Any]],
    devices: Sequence[torch.device],
    check_sync: bool = False,
    has_cuda: bool = torch.cuda.is_available(),
) -> List[LocalStepInfo]:
    infos = threaded_local_steps(replicas, batches, devices)
    batch_sizes = [info.batch_size for info in infos]

    scalers = create_splitbatch_scalers(batch_sizes)
    #torch.cuda.synchronize()
    aggregate_grads_to_master(replicas, batch_sizes, scalers)
    #print master parameters
    #for name, p in replicas[0].named_parameters():
        #print(f"Before master step: {name} param value: {p.detach().cpu()}")
        #print(f"Before master step: {name} grad value: {p.grad.detach().cpu() if p.grad is not None else None}")
    master_step(optimizer)
    #torch.cuda.synchronize()
    sync_from_master(replicas)
    if has_cuda:
        torch.cuda.synchronize() #check if needed

    if check_sync:
        check_replicas_equal(replicas)

    #scale the info loss values accordingly
    for info, scaler in zip(infos, scalers):
        if info.loss_value is not None:
            info.loss_value *= scaler

    return infos


def val_step_threaded(
    replicas: Sequence[nn.Module],
    batches: Sequence[Tuple[Any, Any]],
    devices: Sequence[torch.device],
) -> List[LocalStepInfo]:
    infos = threaded_local_steps(replicas, batches, devices)
    batch_sizes = [info.batch_size for info in infos]
    scalers = scalers = create_splitbatch_scalers(batch_sizes)
    for info, scaler in zip(infos, scalers):
        if info.loss_value is not None:
            info.loss_value *= scaler

    return infos


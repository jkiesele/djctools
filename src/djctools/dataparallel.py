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

import torch
from djctools.parallel import make_replicas, check_replicas_equal, train_step_threaded
import torch.nn as nn
from djctools.module_extensions import LossModule
from typing import Sequence

from djctools.threading_context import forward_context # just for testing here


# ---------------------------------------------------------------------------
# Example toy loss module
# ---------------------------------------------------------------------------

class MSELossModule(LossModule):
    def compute_loss(self, pred, truth=None):
        '''
        to be put into the documentation: loss *must* be mean over batch size
        '''
        if truth is None:
            return None
        # sample-summed contribution
        return ((pred - truth) ** 2).mean()


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
    master_model = ToyModel(2, 2).to(devices[0])
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
        x = torch.randn(bs, 2)
        y = torch.randn(bs, 2)
        batches.append((x, y))
    return batches



def run_equivalence_tests(num_steps: int = 4, atol: float = 1e-6, rtol: float = 1e-6) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for equivalence tests")

    if torch.cuda.device_count() < 2:
        raise RuntimeError("At least 2 CUDA devices required for test 2")

    def set_seed(seed: int) -> None:
        import random
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @torch.no_grad()
    def assert_models_close(model_a: nn.Module, model_b: nn.Module, msg: str) -> None:
        params_a = dict(model_a.named_parameters())
        params_b = dict(model_b.named_parameters())
        bufs_a = dict(model_a.named_buffers())
        bufs_b = dict(model_b.named_buffers())

        for name, p in params_a.items():
            q = params_b[name]
            if not torch.allclose(p.detach().cpu(), q.detach().cpu(), atol=atol, rtol=rtol):
                diff = (p.detach().cpu() - q.detach().cpu()).abs().max().item()
                raise RuntimeError(f"{msg}: parameter '{name}' differs, max abs diff = {diff}")

        for name, b in bufs_a.items():
            q = bufs_b[name]
            if not torch.allclose(b.detach().cpu(), q.detach().cpu(), atol=atol, rtol=rtol):
                diff = (b.detach().cpu() - q.detach().cpu()).abs().max().item()
                raise RuntimeError(f"{msg}: buffer '{name}' differs, max abs diff = {diff}")

    def clone_batches_to_device(batches, device):
        out = []
        for x, y in batches:
            x2 = x.clone().to(device)
            y2 = None if y is None else y.clone().to(device)
            out.append((x2, y2))
        return out

    def make_split_batches(step_idx: int):
        # deterministic but not identical across steps
        bs0 = 8 + (step_idx % 3)
        bs1 = 11 + ((2 * step_idx) % 4)
        x0 = torch.randn(bs0, 2)
        y0 = torch.randn(bs0, 2)
        x1 = torch.randn(bs1, 2)
        y1 = torch.randn(bs1, 2)
        return [(x0, y0), (x1, y1)]

    def make_concat_batch(split_batches):
        xs = [b[0] for b in split_batches]
        ys = [b[1] for b in split_batches]
        return (torch.cat(xs, dim=0), torch.cat(ys, dim=0))

    def single_gpu_reference_step(model, optimizer, batch, device):
        x, y = batch
        x = x.to(device)
        y = None if y is None else y.to(device)

        optimizer.zero_grad(set_to_none=True)
        with forward_context(sample_count=len(x)) as ctx:
            _ = model(x, truth=y)
            loss = ctx.total_loss()
        if loss is not None:
            loss = loss 
            loss.backward()

        ## print all parameters (all of them, not just max or min) before step
        #for name, p in model.named_parameters():
        #    print(f"Before step: {name} param value: {p.detach().cpu()}")
        ## print all gradients before step
        #for name, p in model.named_parameters():
        #    print(f"Before step: {name} grad value: {p.grad.detach().cpu() if p.grad is not None else None}")
        optimizer.step()

    # ------------------------------------------------------------------
    # Test 1: plain single-GPU reference vs threaded path with 1 GPU
    # ------------------------------------------------------------------
    print("Running test 1: single-GPU reference vs threaded 1-GPU")

    seed = 12345
    set_seed(seed)
    ref_model_1 = ToyModel(2, 2).to("cuda:0")
    ref_opt_1 = torch.optim.Adam(ref_model_1.parameters(), lr=1e-2)

    set_seed(seed)
    threaded_model_1 = ToyModel(2, 2).to("cuda:0")



    threaded_replicas_1 = make_replicas(threaded_model_1, [torch.device("cuda:0")])
    threaded_opt_1 = torch.optim.Adam(threaded_model_1.parameters(), lr=1e-2)

    #sanity check if models are same at this stage
    check_replicas_equal([ref_model_1]+ threaded_replicas_1)
    print("models are the same to begin with")

    #check optimisers

    set_seed(seed + 1)
    shared_batches_test1 = []
    for step in range(num_steps):
        bs = 10 # + (step % 4)
        x = torch.randn(bs, 2)
        y = torch.randn(bs, 2)
        shared_batches_test1.append((x, y))

    for batch in shared_batches_test1:
        single_gpu_reference_step(ref_model_1, ref_opt_1, batch, torch.device("cuda:0"))

    for batch in shared_batches_test1:
        local_batches = clone_batches_to_device([batch], torch.device("cuda:0"))
        train_step_threaded(
            replicas=threaded_replicas_1,
            optimizer=threaded_opt_1,
            batches=local_batches,
            devices=[torch.device("cuda:0")],
            check_sync=True,
        )

    assert_models_close(ref_model_1, threaded_replicas_1[0], "Test 1 failed, models weights: "+str(list(ref_model_1.parameters())[0].detach().cpu()) + " vs " + str(list(threaded_replicas_1[0].parameters())[0].detach().cpu()))
    print("Test 1 passed")

    # ------------------------------------------------------------------
    # Test 2: single-GPU concatenated reference vs multi-GPU split batches
    # ------------------------------------------------------------------
    print("Running test 2: single-GPU concatenated reference vs 2-GPU split")

    seed = 67890
    set_seed(seed)
    ref_model_2 = ToyModel(2, 2).to("cuda:0")
    ref_opt_2 = torch.optim.Adam(ref_model_2.parameters(), lr=1e-2)

    set_seed(seed)
    threaded_model_2 = ToyModel(2, 2).to("cuda:0")
    devices_2 = [torch.device("cuda:0"), torch.device("cuda:1")]
    threaded_replicas_2 = make_replicas(threaded_model_2, devices_2)
    threaded_opt_2 = torch.optim.Adam(threaded_replicas_2[0].parameters(), lr=1e-2)

    set_seed(seed + 1)
    split_batches_per_step = [make_split_batches(step) for step in range(num_steps)]
    concat_batches_per_step = [make_concat_batch(split_batches) for split_batches in split_batches_per_step]

    print(concat_batches_per_step)

    for batch in concat_batches_per_step:
        single_gpu_reference_step(ref_model_2, ref_opt_2, batch, torch.device("cuda:0"))

    for split_batches in split_batches_per_step:
        local_batches = [
            (split_batches[0][0].clone().to("cuda:0"), split_batches[0][1].clone().to("cuda:0")),
            (split_batches[1][0].clone().to("cuda:1"), split_batches[1][1].clone().to("cuda:1")),
        ]
        train_step_threaded(
            replicas=threaded_replicas_2,
            optimizer=threaded_opt_2,
            batches=local_batches,
            devices=devices_2,
            check_sync=True,
        )

    assert_models_close(ref_model_2, threaded_replicas_2[0], "Test 2 failed")
    print("Test 2 passed")

    print("All equivalence tests passed")




if __name__ == "__main__":
    run_equivalence_tests()
    exit()
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

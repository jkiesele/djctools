
import torch
import threading
from contextlib import contextmanager
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

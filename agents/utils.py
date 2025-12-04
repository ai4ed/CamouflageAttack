import functools
import gc
import inspect
import torch
from torch import Tensor


def mellowmax(t: Tensor, alpha=1.0, dim=-1):
    return 1.0 / alpha * (torch.logsumexp(alpha * t, dim=dim) - torch.log(torch.tensor(t.shape[-1], dtype=t.dtype, device=t.device)))


def should_reduce_batch_size(exception: Exception) -> bool:
    _statements = [
        "CUDA out of memory.",
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",
        "DefaultCPUAllocator: can't allocate memory",
    ]
    if isinstance(exception, RuntimeError) and len(exception.args) == 1:
        return any(err in exception.args[0] for err in _statements)
    return False

def find_executable_batch_size(function: callable = None, starting_batch_size: int = 128):
    if function is None:
        return functools.partial(find_executable_batch_size, starting_batch_size=starting_batch_size)

    batch_size = starting_batch_size

    def decorator(*args, **kwargs):
        nonlocal batch_size
        gc.collect()
        torch.cuda.empty_cache()
        params = list(inspect.signature(function).parameters.keys())
        
        if len(params) < (len(args) + 1):
            arg_str = ", ".join([f"{arg}={value}" for arg, value in zip(params[1:], args[1:])])
            raise TypeError(
                f"Batch size was passed into `{function.__name__}` as the first argument when called."
                f"Remove this as the decorator already does so: `{function.__name__}({arg_str})`"
            )
        while True:
            if batch_size == 0:
                raise RuntimeError("No executable batch size found, reached zero.")
            try:
                return function(batch_size, *args, **kwargs)
            except Exception as e:
                if should_reduce_batch_size(e):
                    gc.collect()
                    torch.cuda.empty_cache()
                    batch_size //= 2
                    print(f"Decreasing batch size to: {batch_size}")
                else:
                    raise

    return decorator

import torch


def _ensure_float32(func):
    """Ensure diffusion math runs in float32 regardless of autocast context."""
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        args = tuple(a.float() if isinstance(a, torch.Tensor) else a for a in args)
        return func(*args, **kwargs)
    return wrapper


@_ensure_float32
def predict_start_from_noise( z_t, noise, alpha2):
    x0 = (z_t - (1-alpha2).sqrt() * noise) / alpha2.sqrt().clamp(min = 1e-8)
    return x0

@_ensure_float32
def predict_noise_from_start(z_t, x0, alpha2):
    eps = (z_t - alpha2.sqrt() * x0) / (1-alpha2).sqrt().clamp(min = 1e-8)
    return eps

@_ensure_float32
def predict_start_from_v(z_t, v, alpha2):
    x0 = alpha2.sqrt() * z_t - (1-alpha2).sqrt() * v
    return x0

@_ensure_float32
def predict_noise_from_v(z_t, v, alpha2):
    eps = (1-alpha2).sqrt() * z_t + alpha2.sqrt() * v
    return eps

@_ensure_float32
def predict_v_from_start_and_eps(x, noise, alpha2):
    v = alpha2.sqrt() * noise - x* (1-alpha2).sqrt()
    return v

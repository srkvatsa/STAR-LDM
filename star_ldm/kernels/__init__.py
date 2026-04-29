"""Metal compute shader loader for STAR-LDM.

Loads compiled Metal kernels via torch.utils.cpp_extension.load().
Falls back to JIT-scripted versions if Metal compilation fails or
MPS is not available.

Usage:
    from star_ldm.kernels import metal_ddpm_step, metal_rmsnorm_film, ...
    # Returns None if Metal is not available; caller checks and falls back.
"""

import os
import logging
import torch

logger = logging.getLogger(__name__)

_KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))

# Cached loaded modules
_ddpm_step_mod = None
_rmsnorm_film_mod = None
_tiny_attention_mod = None
_spec_verify_mod = None
_decode_attn_mod = None
_fused_ffn_mod = None

# Flag: have we already tried to load?
_load_attempted = {
    'ddpm_step': False,
    'rmsnorm_film': False,
    'tiny_attention': False,
    'spec_verify': False,
    'decode_attn': False,
    'fused_ffn': False,
}


def _is_mps_available():
    """Check if MPS backend is available."""
    return (
        hasattr(torch.backends, 'mps')
        and torch.backends.mps.is_available()
    )


def _load_metal_extension(name, sources, extra_cflags=None):
    """Load a Metal kernel extension via torch.utils.cpp_extension.

    Args:
        name: Extension module name.
        sources: List of source file paths (.mm files).
        extra_cflags: Additional compiler flags.

    Returns:
        The loaded module, or None if compilation fails.
    """
    if not _is_mps_available():
        logger.info("MPS not available, skipping Metal kernel: %s", name)
        return None

    try:
        from torch.utils.cpp_extension import load

        if extra_cflags is None:
            extra_cflags = []

        # Obj-C++ flags for Metal dispatch
        extra_cflags += [
            '-std=c++17',
            '-ObjC++',
            '-O2',
        ]

        extra_ldflags = [
            '-framework', 'Metal',
            '-framework', 'Foundation',
        ]

        module = load(
            name=name,
            sources=sources,
            extra_cflags=extra_cflags,
            extra_ldflags=extra_ldflags,
            verbose=False,
        )
        logger.info("Successfully loaded Metal kernel: %s", name)
        return module

    except Exception as e:
        logger.warning("Failed to load Metal kernel '%s': %s", name, e)
        return None


def get_ddpm_step_kernel():
    """Get the Metal DDPM step kernel, or None if unavailable."""
    global _ddpm_step_mod
    if not _load_attempted['ddpm_step']:
        _load_attempted['ddpm_step'] = True
        _ddpm_step_mod = _load_metal_extension(
            'metal_ddpm_step',
            [os.path.join(_KERNEL_DIR, 'ddpm_step.mm')],
        )
    return _ddpm_step_mod


def get_rmsnorm_film_kernel():
    """Get the Metal RMSNorm+FiLM kernel, or None if unavailable."""
    global _rmsnorm_film_mod
    if not _load_attempted['rmsnorm_film']:
        _load_attempted['rmsnorm_film'] = True
        _rmsnorm_film_mod = _load_metal_extension(
            'metal_rmsnorm_film',
            [os.path.join(_KERNEL_DIR, 'rmsnorm_film.mm')],
        )
    return _rmsnorm_film_mod


def get_tiny_attention_kernel():
    """Get the Metal tiny attention kernel, or None if unavailable."""
    global _tiny_attention_mod
    if not _load_attempted['tiny_attention']:
        _load_attempted['tiny_attention'] = True
        _tiny_attention_mod = _load_metal_extension(
            'metal_tiny_attention',
            [os.path.join(_KERNEL_DIR, 'tiny_attention.mm')],
        )
    return _tiny_attention_mod


def get_spec_verify_kernel():
    """Get the Metal speculative verification kernel, or None if unavailable."""
    global _spec_verify_mod
    if not _load_attempted['spec_verify']:
        _load_attempted['spec_verify'] = True
        _spec_verify_mod = _load_metal_extension(
            'metal_spec_verify',
            [os.path.join(_KERNEL_DIR, 'spec_verify.mm')],
        )
    return _spec_verify_mod


def get_decode_attention_kernel():
    """Get the Metal decode-N attention kernel, or None if unavailable."""
    global _decode_attn_mod
    if not _load_attempted['decode_attn']:
        _load_attempted['decode_attn'] = True
        _decode_attn_mod = _load_metal_extension(
            'metal_decode_attn',
            [os.path.join(_KERNEL_DIR, 'decode_attention.mm')],
        )
    return _decode_attn_mod


def get_fused_ffn_kernel():
    """Get the Metal fused FFN kernel, or None if unavailable."""
    global _fused_ffn_mod
    if not _load_attempted['fused_ffn']:
        _load_attempted['fused_ffn'] = True
        _fused_ffn_mod = _load_metal_extension(
            'metal_fused_ffn',
            [os.path.join(_KERNEL_DIR, 'fused_ffn.mm')],
        )
    return _fused_ffn_mod


def preload_all_kernels():
    """Eagerly load all Metal kernels. Call at startup for faster first inference."""
    get_ddpm_step_kernel()
    get_rmsnorm_film_kernel()
    get_tiny_attention_kernel()
    get_spec_verify_kernel()
    get_decode_attention_kernel()

"""CPU/GPU heterogeneous overlap for STAR-LDM diffusion sampling.

Exploits Apple Silicon's unified memory architecture to overlap CPU
computation with GPU execution. MPS operations are asynchronous — GPU
work is enqueued and executes independently. We generate the next
step's noise and precompute schedule values on CPU while the GPU runs
the diffusion model forward pass, then do a zero-copy transfer to MPS.

Usage:
    scheduler = AsyncDiffusionScheduler(noise_schedule_fn, device='mps')
    for time, time_next in time_pairs:
        # Get precomputed noise and schedule from previous step
        noise_mps, alpha2, alpha2_next = scheduler.get_precomputed(time, time_next)

        # Start precomputing NEXT step's noise while GPU runs model forward
        scheduler.precompute_next(time_next_next, z_shape)

        # GPU model forward (async — overlaps with CPU noise generation)
        v_pred = model(z_t, alpha2, ...)

        # Use precomputed noise (already on MPS via zero-copy)
        z_t = fused_ddpm_step(z_t, eps, noise_mps, alpha2, alpha2_next, var_lambda)
"""

import torch
from typing import Callable, Optional, Tuple


class AsyncDiffusionScheduler:
    """Manages CPU/GPU overlap for diffusion step precomputation.

    Precomputes noise and schedule values on CPU while GPU runs the
    diffusion model forward pass. On Apple Silicon unified memory,
    the CPU→MPS transfer is zero-copy (shared physical pages).

    Args:
        noise_schedule_fn: Function that maps time → alpha2 (the noise schedule).
        device: Target device ('mps', 'cuda', or 'cpu').
        z_shape: Shape of the latent tensor (B, D) for noise precomputation.
    """

    def __init__(
        self,
        noise_schedule_fn: Callable,
        device: str = 'mps',
        z_shape: Optional[Tuple[int, ...]] = None,
    ):
        self.noise_schedule_fn = noise_schedule_fn
        self.device = torch.device(device)
        self.z_shape = z_shape

        # Precomputed values from CPU
        self._cpu_noise: Optional[torch.Tensor] = None
        self._precomputed_valid = False

    def precompute_noise(self, z_shape: Tuple[int, ...]) -> None:
        """Generate Gaussian noise on CPU (runs while GPU is busy).

        On unified memory (MPS), this uses CPU cores to generate noise
        in parallel with GPU compute. The tensor stays in shared memory
        and the .to('mps') is effectively free (no DMA copy needed).

        Args:
            z_shape: Shape for noise tensor (B, D).
        """
        self.z_shape = z_shape
        # Generate on CPU — uses CPU PRNG which doesn't contend with GPU
        self._cpu_noise = torch.randn(z_shape, device='cpu', dtype=torch.float32)
        self._precomputed_valid = True

    def get_noise(self) -> torch.Tensor:
        """Get precomputed noise, transferred to the target device.

        On MPS with unified memory, the .to('mps') transfer is zero-copy:
        the CPU and GPU share the same physical memory pages, so this
        just updates the tensor's device metadata.

        Returns:
            (B, D) noise tensor on the target device.
        """
        if self._precomputed_valid and self._cpu_noise is not None:
            noise = self._cpu_noise.to(self.device)
            self._precomputed_valid = False
            return noise
        else:
            # Fallback: generate directly on device
            if self.z_shape is not None:
                return torch.randn(self.z_shape, device=self.device)
            raise RuntimeError("No precomputed noise available and z_shape not set")

    def compute_schedule_values(
        self, time: torch.Tensor, time_next: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute alpha2 schedule values on CPU, transfer to device.

        This is fast but we compute it on CPU to demonstrate the
        heterogeneous scheduling pattern and keep GPU free for
        the model forward pass.

        Args:
            time: Current time values (B,).
            time_next: Next time values (B,).

        Returns:
            (alpha2, alpha2_next) on target device, each (B, 1).
        """
        # Compute on CPU
        with torch.no_grad():
            time_cpu = time.cpu() if time.device.type != 'cpu' else time
            time_next_cpu = time_next.cpu() if time_next.device.type != 'cpu' else time_next

            alpha2_cpu = self.noise_schedule_fn(time_cpu).unsqueeze(-1)
            alpha2_next_cpu = self.noise_schedule_fn(time_next_cpu).unsqueeze(-1)

        # Transfer to device (zero-copy on unified memory)
        alpha2 = alpha2_cpu.to(self.device)
        alpha2_next = alpha2_next_cpu.to(self.device)

        return alpha2, alpha2_next


def create_async_diffusion_loop(
    model_forward_fn: Callable,
    noise_schedule_fn: Callable,
    ddpm_step_fn: Callable,
    time_pairs: list,
    z_t: torch.Tensor,
    device: str = 'mps',
    var_lambda: float = 0.2,
    **model_kwargs,
) -> torch.Tensor:
    """Run the diffusion sampling loop with CPU/GPU overlap.

    Orchestrates the async scheduling pattern:
    1. Start noise precomputation on CPU
    2. Run model forward on GPU (async)
    3. Sync: use both GPU output and CPU noise for DDPM step

    Args:
        model_forward_fn: Function(z_t, alpha2, **kwargs) → ModelPrediction.
        noise_schedule_fn: Time → alpha2 schedule function.
        ddpm_step_fn: Fused DDPM step function.
        time_pairs: List of (time, time_next) tuples.
        z_t: Initial noisy latent (B, D).
        device: Device string.
        var_lambda: DDPM variance interpolation.
        **model_kwargs: Additional arguments for model_forward_fn.

    Returns:
        x_start: Final denoised latent (B, D).
    """
    scheduler = AsyncDiffusionScheduler(noise_schedule_fn, device, z_t.shape)
    x_start = None

    # Precompute first step's noise
    scheduler.precompute_noise(z_t.shape)

    for i, (time, time_next) in enumerate(time_pairs):
        # Get schedule values (computed on CPU, transferred to device)
        alpha2, alpha2_next = scheduler.compute_schedule_values(time, time_next)

        # Start precomputing NEXT step's noise on CPU
        # (overlaps with GPU model forward below)
        if i + 1 < len(time_pairs):
            # This runs on CPU while GPU executes the model forward
            import threading
            noise_thread = threading.Thread(
                target=scheduler.precompute_noise,
                args=(z_t.shape,)
            )
            noise_thread.start()

        # GPU: model forward pass (async on MPS)
        model_output = model_forward_fn(z_t, alpha2, **model_kwargs)
        x_start = model_output.pred_x
        eps = model_output.pred_eps

        if time_next[0] <= 0:
            z_t = x_start
            # Wait for any running noise thread
            if i + 1 < len(time_pairs) and noise_thread is not None:
                noise_thread.join()
            continue

        # Wait for noise precomputation to finish
        if i + 1 < len(time_pairs):
            noise_thread.join()

        # Get precomputed noise (zero-copy MPS transfer)
        noise = scheduler.get_noise()

        # DDPM step using both GPU output and CPU-precomputed noise
        z_t = ddpm_step_fn(z_t, eps, noise, alpha2, alpha2_next, var_lambda)

        # Precompute noise for next-next step (if this wasn't the last)
        if i + 2 < len(time_pairs):
            pass  # Already started in the next iteration

    return x_start

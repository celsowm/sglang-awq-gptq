import torch

class _CurrentPlatform:
    def is_cuda_alike(self) -> bool:
        return torch.cuda.is_available()

    def is_tpu(self) -> bool:
        # Assuming no TPU support in sglang for this specific vendoring context
        return False

    def is_xpu(self) -> bool:
        # Check if torch.xpu is available and there's at least one XPU device
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            try:
                return torch.xpu.device_count() > 0
            except Exception: # pylint: disable=broad-except
                return False
        return False

    def is_neuron(self) -> bool:
        return False # Assuming no Neuron support

    def is_cpu(self) -> bool:
        # True if no other specialized hardware is detected/available
        return not any([
            self.is_cuda_alike(),
            self.is_xpu(),
            self.is_tpu(),
            self.is_neuron(),
        ])

    def is_rocm(self) -> bool:
        # ROCm is a type of CUDA-alike platform from PyTorch's perspective
        return self.is_cuda_alike() and \
               hasattr(torch.version, 'hip') and \
               torch.version.hip is not None

    def get_device_capability(self, device: int = 0) -> tuple[int, int]:
        if self.is_cuda_alike() and not self.is_rocm():
            try:
                return torch.cuda.get_device_capability(device)
            except Exception: # pylint: disable=broad-except
                # Fallback if specific device query fails
                return (8, 0)
        # Return a common default for other cases (ROCm, XPU, etc.) or if CUDA not available
        return (8, 0) # Default to SM 8.0 or equivalent MI arch for ROCm if not queryable simply

    def seed_everything(self, seed: int) -> None:
        torch.manual_seed(seed)
        if self.is_cuda_alike():
            torch.cuda.manual_seed_all(seed)
        if self.is_xpu():
            torch.xpu.manual_seed_all(seed) # type: ignore

current_platform = _CurrentPlatform()

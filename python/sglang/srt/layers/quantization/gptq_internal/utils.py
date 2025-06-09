from typing import Any, Dict, Optional, Callable
import torch
from .platforms_placeholder import current_platform

# Adapted from vllm/model_executor/utils.py
def _make_synced_weight_loader(original_weight_loader: Callable):
    def _synced_weight_loader(param, *args, **kwargs):
        original_weight_loader(param, *args, **kwargs)
        # torch._sync(param) is a vLLM-specific TPU utility.
        # For non-TPU environments or if sglang has a different TPU sync mechanism,
        # this can be omitted or replaced. For now, it's a no-op.
        pass
    return _synced_weight_loader

def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: Optional[Dict[str, Any]],
):
    """
    Set attributes on a weight tensor.
    Adapted from vLLM, with overwrite allowed for simplicity during vendoring.
    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        # Original vLLM had an assert to prevent overwriting tensor attributes.
        # This version allows it, which might be needed if attributes are set multiple times
        # or from different sources during sglang's model/weight loading.

        processed_value = value
        # Relies on placeholder current_platform.is_tpu()
        if current_platform.is_tpu() and key == "weight_loader":
            processed_value = _make_synced_weight_loader(value)
        setattr(weight, key, processed_value)

# Placeholder for replace_parameter, which might be found in
# vllm.model_executor.layers.quantization.utils in vLLM.
def replace_parameter(layer: torch.nn.Module, param_name: str, new_param_data: torch.Tensor) -> None:
    """
    Helper to replace a module's parameter with new data.
    The new parameter will have requires_grad=False.
    This is a simplified version for vendoring.
    """
    # Ensure the old parameter is removed if it exists to avoid issues with name conflicts
    # or if the original was not a Parameter but a buffer or plain tensor.
    if hasattr(layer, param_name):
        delattr(layer, param_name)

    layer.register_parameter(param_name, torch.nn.Parameter(new_param_data, requires_grad=False))

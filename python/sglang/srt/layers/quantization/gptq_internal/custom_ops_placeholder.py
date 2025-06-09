import torch
import logging

logger = logging.getLogger(__name__)

_VLLM_OPS_AVAILABLE = False
_vllm_custom_ops_real = None
_torch_ops_vllm_real = None

try:
    import vllm._custom_ops as vllm_custom_ops_real_import
    _vllm_custom_ops_real = vllm_custom_ops_real_import

    # Check for torch.ops.vllm after successfully importing vllm._custom_ops
    if hasattr(torch.ops, "vllm"):
        _torch_ops_vllm_real = torch.ops.vllm
    else:
        # This case might occur if vllm._custom_ops exists but torch.ops.vllm was not registered,
        # or if specific ops are in vllm._custom_ops but not torch.ops.vllm.
        # For ops expected under torch.ops.vllm, this means they won't be found if this path is taken.
        logger.debug("vllm._custom_ops imported but torch.ops.vllm not found.")

    _VLLM_OPS_AVAILABLE = True
    logger.info("Successfully imported vLLM custom ops.")

except ImportError:
    logger.warning(
        "Failed to import vLLM custom ops. "
        "Sglang will use placeholder kernels for GPTQ/Marlin/AWQ operations. "
        "Ensure vLLM is installed with CUDA support if GPU execution is desired."
    )
except Exception as e:
    logger.error(f"An unexpected error occurred while importing vLLM custom ops: {e}")


# Define specific op functions that conditionally call real or placeholder implementations

def _generic_op_call(op_name: str, source_module, is_torch_op: bool, *args, **kwargs):
    """Helper to call an op or raise NotImplementedError."""
    if _VLLM_OPS_AVAILABLE and source_module is not None and hasattr(source_module, op_name):
        return getattr(source_module, op_name)(*args, **kwargs)

    source_name = "torch.ops.vllm" if is_torch_op else "vllm._custom_ops"

    # Fallback to dummy implementations for each op
    if op_name == "get_max_shared_memory_per_block_device_attribute":
        return 49152 # Default value
    elif op_name == "gptq_shuffle":
        if len(args) > 0 and hasattr(args[0], 'device'): return
        raise NotImplementedError(f"Placeholder for {source_name}.{op_name}")
    elif op_name in ["gptq_gemm", "marlin_gemm"]: # Assuming similar dummy output logic
        if len(args) > 0 and hasattr(args[0], 'shape') and hasattr(args[1], 'shape'):
            reshaped_x, qweight = args[0], args[1]
            output_features = qweight.shape[-1]
            if op_name == "marlin_gemm" and len(args) >=6 : # marlin_gemm has size_n as 5th arg (index 4)
                output_features = args[5] # size_n

            output_shape = reshaped_x.shape[:-1] + (output_features,)
            return torch.zeros(output_shape, dtype=reshaped_x.dtype, device=reshaped_x.device)
        raise NotImplementedError(f"Placeholder for {source_name}.{op_name}")
    elif op_name == "fused_marlin_moe": # torch.ops.vllm.fused_marlin_moe
        if len(args) > 0 and hasattr(args[0], 'shape'):
            return torch.zeros_like(args[0]) # x
        raise NotImplementedError(f"Placeholder for {source_name}.{op_name}")
    elif op_name == "gptq_marlin_moe_repack":
        if len(args) > 0 and hasattr(args[0], 'shape'):
            return torch.zeros_like(args[0]) # qweight
        raise NotImplementedError(f"Placeholder for {source_name}.{op_name}")
    elif op_name == "scaled_int8_quant":
        if len(args) > 0 and hasattr(args[0], 'shape'):
            x_2d = args[0]
            return torch.zeros_like(x_2d, dtype=torch.int8), \
                   torch.zeros(x_2d.shape[0], dtype=x_2d.dtype, device=x_2d.device), \
                   None
        raise NotImplementedError(f"Placeholder for {source_name}.{op_name}")
    elif op_name == "marlin_qqq_gemm":
        if len(args) > 0 and hasattr(args[0], 'shape'):
            x_int8 = args[0]
            size_m, size_n = args[6], args[7]
            return torch.zeros((size_m, size_n), dtype=torch.float16, device=x_int8.device)
        raise NotImplementedError(f"Placeholder for {source_name}.{op_name}")
    elif op_name == "fused_experts": # Used by Unquantized MoE
        if len(args) > 0 and hasattr(args[0], 'shape'): # hidden_states
             return torch.zeros_like(args[0])
        raise NotImplementedError(f"Placeholder for {source_name}.{op_name}")
    elif op_name == "fused_topk": # Used by MoE select_experts
        if len(args) > 1 and hasattr(args[0], 'shape') and hasattr(args[1], 'shape'): # hidden_states, gating_output
            hidden_states, gating_output = args[0], args[1]
            top_k = args[2] if len(args) > 2 else 1
            num_tokens = gating_output.shape[0]
            # topk_weights, topk_ids
            return torch.zeros((num_tokens, top_k), dtype=hidden_states.dtype, device=hidden_states.device), \
                   torch.zeros((num_tokens, top_k), dtype=torch.long, device=hidden_states.device)
        raise NotImplementedError(f"Placeholder for {source_name}.{op_name}")

    # General fallback if no specific dummy logic is hit
    raise NotImplementedError(
        f"vLLM custom op '{op_name}' from {source_name} is not available or has no specific placeholder logic. "
        "Please ensure vLLM with compiled kernels is installed for this functionality."
    )

# Ops typically found in vllm._custom_ops
def get_max_shared_memory_per_block_device_attribute(*args, **kwargs):
    return _generic_op_call("get_max_shared_memory_per_block_device_attribute", _vllm_custom_ops_real, False, *args, **kwargs)

def gptq_shuffle(*args, **kwargs):
    return _generic_op_call("gptq_shuffle", _vllm_custom_ops_real, False, *args, **kwargs)

def gptq_gemm(*args, **kwargs):
    return _generic_op_call("gptq_gemm", _vllm_custom_ops_real, False, *args, **kwargs)

def marlin_gemm(*args, **kwargs):
    return _generic_op_call("marlin_gemm", _vllm_custom_ops_real, False, *args, **kwargs)

def gptq_marlin_moe_repack(*args, **kwargs):
    return _generic_op_call("gptq_marlin_moe_repack", _vllm_custom_ops_real, False, *args, **kwargs)

def scaled_int8_quant(*args, **kwargs):
    return _generic_op_call("scaled_int8_quant", _vllm_custom_ops_real, False, *args, **kwargs)

def marlin_qqq_gemm(*args, **kwargs):
    return _generic_op_call("marlin_qqq_gemm", _vllm_custom_ops_real, False, *args, **kwargs)

def fused_experts(*args, **kwargs): # For MoE
    return _generic_op_call("fused_experts", _vllm_custom_ops_real, False, *args, **kwargs)

def fused_topk(*args, **kwargs): # For MoE
    return _generic_op_call("fused_topk", _vllm_custom_ops_real, False, *args, **kwargs)

def grouped_topk(*args, **kwargs): # For MoE
    # This op might not exist in older vllm._custom_ops, provide dummy logic
    if _VLLM_OPS_AVAILABLE and _vllm_custom_ops_real is not None and hasattr(_vllm_custom_ops_real, "grouped_topk"):
        return _vllm_custom_ops_real.grouped_topk(*args, **kwargs)
    # Dummy for grouped_topk
    if len(args) > 1 and hasattr(args[0], 'shape') and hasattr(args[1], 'shape'): # hidden_states, gating_output
        hidden_states, gating_output = args[0], args[1]
        topk = args[2] if len(args) > 2 else 1 # typically 'topk' not 'top_k' in op args
        num_tokens = gating_output.shape[0]
        return torch.zeros((num_tokens, topk), dtype=hidden_states.dtype, device=hidden_states.device), \
               torch.zeros((num_tokens, topk), dtype=torch.long, device=hidden_states.device)
    raise NotImplementedError("Placeholder for vllm._custom_ops.grouped_topk")


# Ops typically found in torch.ops.vllm
def fused_marlin_moe(*args, **kwargs):
    return _generic_op_call("fused_marlin_moe", _torch_ops_vllm_real, True, *args, **kwargs)


# General ops wrapper for any other ops that might be called via `ops.op_name`
class _OpsForwarder:
    def __getattr__(self, name):
        # Prioritize torch.ops.vllm if the op exists there, then vllm._custom_ops
        if _VLLM_OPS_AVAILABLE and _torch_ops_vllm_real is not None and hasattr(_torch_ops_vllm_real, name):
            return getattr(_torch_ops_vllm_real, name)
        if _VLLM_OPS_AVAILABLE and _vllm_custom_ops_real is not None and hasattr(_vllm_custom_ops_real, name):
            return getattr(_vllm_custom_ops_real, name)

        # Fallback to the generic placeholder if op is not found in either real module.
        # This will call _generic_op_call which then tries to find a specific dummy implementation
        # or raises a general NotImplementedError.
        # Note: This assumes op names are unique across _vllm_custom_ops_real and
        # _torch_ops_vllm_real or that the prioritization is acceptable.
        def call_generic_placeholder(*args, **kwargs):
            # Assume not a torch_op for error message if source is unknown at this point.
            return _generic_op_call(name, None, False, *args, **kwargs)
        return call_generic_placeholder

ops = _OpsForwarder()

# This special variable is used by sglang to check kernel availability.
VLLM_KERNELS_AVAILABLE = _VLLM_OPS_AVAILABLE

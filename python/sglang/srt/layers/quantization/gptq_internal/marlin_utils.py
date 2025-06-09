from typing import List, Optional, Tuple

import numpy
import torch

from .custom_ops_placeholder import ops
from .platforms_placeholder import current_platform
from .scalar_type import SglangScalarType, sglang_scalar_types
from .quant_utils import pack_cols, unpack_cols

GPTQ_MARLIN_TILE = 16
GPTQ_MARLIN_MIN_THREAD_N = 64
GPTQ_MARLIN_MIN_THREAD_K = 128
GPTQ_MARLIN_MAX_PARALLEL = 16

MARLIN_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]

USE_FP32_REDUCE_DEFAULT = True


def query_marlin_supported_quant_types(has_zp: bool,
                                       device_capability: Optional[int] = None
                                       ) -> List[SglangScalarType]:
    if device_capability is None:
        capability_tuple = current_platform.get_device_capability()
        # capability_tuple can be None if platform is e.g. CPU
        device_capability = (-1 if capability_tuple is None else
                             capability_tuple[0] * 10 + capability_tuple[1])

    if device_capability < 80: # Requires SM 8.0+
        return []

    if has_zp:
        return [sglang_scalar_types.uint4, sglang_scalar_types.uint8]
    else:
        return [sglang_scalar_types.uint4b8, sglang_scalar_types.uint8b128]


def _check_marlin_supported(
        quant_type: SglangScalarType,
        group_size: Optional[int],
        has_zp: bool,
        device_capability: Optional[int] = None) -> Tuple[bool, Optional[str]]:

    if device_capability is None:
        capability_tuple = current_platform.get_device_capability()
        device_capability = (-1 if capability_tuple is None else
                             capability_tuple[0] * 10 + capability_tuple[1])

    supported_types = query_marlin_supported_quant_types(
        has_zp, device_capability)

    if quant_type not in supported_types:
        return (False, f"Marlin does not support weight_bits = {quant_type}. "
                f"Only types = {supported_types} "
                f"are supported (for group_size = {group_size}, "
                f"device_capability = {device_capability}, zp = {has_zp}).")
    if (group_size is None or group_size not in MARLIN_SUPPORTED_GROUP_SIZES):
        return (False, f"Marlin does not support group_size = {group_size}. "
                f"Only group_sizes = {MARLIN_SUPPORTED_GROUP_SIZES} "
                "are supported.")

    return True, None


def check_marlin_supported(quant_type: SglangScalarType,
                           group_size: int,
                           has_zp: bool = False,
                           device_capability: Optional[int] = None) -> bool:
    cond, _ = _check_marlin_supported(quant_type, group_size, has_zp,
                                      device_capability)
    return cond


def verify_marlin_supported(quant_type: SglangScalarType,
                            group_size: int,
                            has_zp: bool = False) -> None:
    cond, err_msg = _check_marlin_supported(quant_type, group_size, has_zp)
    if not cond:
        assert err_msg is not None
        raise ValueError(err_msg)


def verify_marlin_supports_shape(output_size_per_partition: int,
                                 input_size_per_partition: int,
                                 input_size: int, group_size: int) -> None:

    if output_size_per_partition % GPTQ_MARLIN_MIN_THREAD_N != 0:
        raise ValueError(f"Weight output_size_per_partition = "
                         f"{output_size_per_partition} is not divisible by "
                         f" min_thread_n = {GPTQ_MARLIN_MIN_THREAD_N}. "
                         "Consider reducing tensor_parallel_size or running "
                         "with --quantization gptq.")

    if input_size_per_partition % GPTQ_MARLIN_MIN_THREAD_K != 0:
        raise ValueError(f"Weight input_size_per_partition = "
                         f"{input_size_per_partition} is not divisible "
                         f"by min_thread_k = {GPTQ_MARLIN_MIN_THREAD_K}. "
                         "Consider reducing tensor_parallel_size or running "
                         "with --quantization gptq.")

    if (group_size != -1 and
            group_size < input_size and
            input_size_per_partition % group_size != 0):
        raise ValueError(
            f"Weight input_size_per_partition = {input_size_per_partition}"
            f" is not divisible by group_size = {group_size}."
            "Consider reducing tensor_parallel_size or running "
            "with --quantization gptq.")


def check_marlin_supports_shape(output_size_per_partition: int,
                                input_size_per_partition: int,
                                input_size: int, group_size: int) \
                                    -> Tuple[bool, Optional[str]]:
    try:
        verify_marlin_supports_shape(output_size_per_partition,
                                     input_size_per_partition, input_size,
                                     group_size)
    except ValueError as e:
        return False, e.__str__()
    return True, None


def marlin_make_workspace(output_size_per_partition: int,
                          device: torch.device) -> torch.Tensor:
    max_workspace_size = (output_size_per_partition //
                          GPTQ_MARLIN_MIN_THREAD_N) * GPTQ_MARLIN_MAX_PARALLEL

    return torch.zeros(max_workspace_size,
                       dtype=torch.int32,
                       device=device,
                       requires_grad=False)


def marlin_is_k_full(act_order: bool, is_row_parallel: bool) -> bool:
    return (not act_order) or (act_order and not is_row_parallel)


def marlin_repeat_scales_on_all_ranks(act_order: bool, group_size: int,
                                      is_row_parallel: bool) -> bool:
    is_channelwise = group_size == -1
    return act_order or (is_channelwise and is_row_parallel)


def marlin_make_empty_g_idx(device: torch.device) -> torch.Tensor:
    return torch.nn.Parameter(torch.empty(0, dtype=torch.int32, device=device),
                              requires_grad=False)


def marlin_make_empty_zp(device: torch.device) -> torch.Tensor:
    return torch.nn.Parameter(torch.empty(0, dtype=torch.int32, device=device),
                              requires_grad=False)


def marlin_sort_g_idx(
        g_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    g_idx_sort_indices = torch.argsort(g_idx).to(torch.int32)
    return g_idx[g_idx_sort_indices], g_idx_sort_indices


def get_scale_perms():
    scale_perm: List[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: List[int] = []
    for i in range(4):
        scale_perm_single.extend(
            [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def marlin_permute_scales(s: torch.Tensor, size_k: int, size_n: int,
                          group_size: int) -> torch.Tensor:
    scale_perm, scale_perm_single = get_scale_perms()
    if group_size != -1 and group_size < size_k :
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else: # Channelwise
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()
    return s


def marlin_moe_permute_scales(
    s: torch.Tensor,
    size_k: int,
    size_n: int,
    group_size: int,
):
    num_experts = s.shape[0]
    output = torch.empty_like(s)

    for e in range(num_experts):
        output[e] = marlin_permute_scales(s[e], size_k, size_n, group_size)
    return output


def marlin_zero_points(zp: torch.Tensor, size_k: int, size_n: int,
                       num_bits: int) -> torch.Tensor:
    scale_perm, _ = get_scale_perms()
    zp = zp.reshape((-1, len(scale_perm)))[:, scale_perm]

    if num_bits == 4:
        interleave = numpy.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = numpy.array([0, 2, 1, 3])
    else:
        raise Exception(f"num_bits must be 4 or 8, got {num_bits}")

    zp = zp.reshape((-1, len(interleave)))[:, interleave].ravel()
    zp = zp.reshape((-1, size_n)).contiguous()
    # pack_cols expects torch tensor
    zp = pack_cols(zp if isinstance(zp, torch.Tensor) else torch.from_numpy(zp).to(torch.int32),
                   num_bits, size_k, size_n)
    return zp


def awq_to_marlin_zero_points(q_zp_packed: torch.Tensor, size_k: int,
                              size_n: int, num_bits: int) -> torch.Tensor:
    q_zp = unpack_cols(q_zp_packed, num_bits, size_k, size_n)

    if num_bits == 4:
        undo_interleave = numpy.argsort(numpy.array([0, 2, 4, 6, 1, 3, 5, 7]))
    elif num_bits == 8:
        undo_interleave = numpy.argsort(numpy.array([0, 2, 1, 3]))
    else:
        raise Exception(f"num_bits must be 4 or 8, got {num_bits}")

    # Ensure q_zp is on CPU for numpy operations if it's not already
    q_zp_cpu = q_zp.cpu() if q_zp.device.type != 'cpu' else q_zp
    q_zp_numpy = q_zp_cpu.numpy()

    q_zp_numpy = q_zp_numpy.reshape((-1, len(undo_interleave)))[:, undo_interleave].ravel()
    q_zp_numpy = q_zp_numpy.reshape((-1, size_n))

    # marlin_zero_points expects a tensor
    q_zp_tensor = torch.from_numpy(q_zp_numpy).to(q_zp_packed.device)

    marlin_zp = marlin_zero_points(q_zp_tensor, size_k, size_n, num_bits)
    return marlin_zp


def moe_awq_to_marlin_zero_points(q_zp_packed: torch.Tensor, size_k: int,
                                  size_n: int, num_bits: int):
    num_experts = q_zp_packed.shape[0]
    # Assuming q_zp_packed is (num_experts, groups, packed_out_features)
    output = torch.empty_like(q_zp_packed)
    for e in range(num_experts):
        output[e] = awq_to_marlin_zero_points(q_zp_packed[e], size_k, size_n, num_bits)
    return output


def apply_gptq_marlin_linear(
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zp: torch.Tensor,
        g_idx: torch.Tensor,
        g_idx_sort_indices: torch.Tensor,
        workspace: torch.Tensor,
        wtype: SglangScalarType,
        output_size_per_partition: int,
        input_size_per_partition: int,
        is_k_full: bool,
        bias: Optional[torch.Tensor] = None,
        use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT) -> torch.Tensor:
    reshaped_x = input_tensor.reshape(-1, input_tensor.shape[-1])
    out_shape = input_tensor.shape[:-1] + (output_size_per_partition, )

    output = ops.gptq_marlin_gemm(reshaped_x,
                                  weight,
                                  weight_scale,
                                  weight_zp,
                                  g_idx,
                                  g_idx_sort_indices,
                                  workspace,
                                  wtype,
                                  size_m=reshaped_x.shape[0],
                                  size_n=output_size_per_partition,
                                  size_k=input_size_per_partition,
                                  is_k_full=is_k_full,
                                  has_zp=False, # GPTQ Marlin does not use runtime ZP for gemm
                                  use_fp32_reduce=use_fp32_reduce)

    if bias is not None:
        output.add_(bias)

    return output.reshape(out_shape)


def apply_awq_marlin_linear(
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zp: torch.Tensor,
        g_idx: torch.Tensor,
        g_idx_sort_indices: torch.Tensor,
        workspace: torch.Tensor,
        quant_type: SglangScalarType,
        output_size_per_partition: int,
        input_size_per_partition: int,
        bias: Optional[torch.Tensor] = None,
        use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT) -> torch.Tensor:
    reshaped_x = input_tensor.reshape(-1, input_tensor.shape[-1])
    out_shape = input_tensor.shape[:-1] + (output_size_per_partition, )

    output = ops.gptq_marlin_gemm(reshaped_x,
                                  weight,
                                  weight_scale,
                                  weight_zp,
                                  g_idx,
                                  g_idx_sort_indices,
                                  workspace,
                                  quant_type,
                                  size_m=reshaped_x.shape[0],
                                  size_n=output_size_per_partition,
                                  size_k=input_size_per_partition,
                                  is_k_full=True, # AWQ Marlin implies is_k_full=True
                                  has_zp=True,    # AWQ Marlin uses runtime ZP for gemm
                                  use_fp32_reduce=use_fp32_reduce)

    if bias is not None:
        output.add_(bias)

    return output.reshape(out_shape)

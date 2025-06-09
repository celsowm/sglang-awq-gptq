from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter
import logging

from .custom_ops_placeholder import ops
from sglang.srt.layers.linear import LinearBase, LinearMethodBase
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.logits_processor import ParallelLMHead
from .parameter import (BasevLLMParameter,
                        ChannelQuantScaleParameter,
                        GroupQuantScaleParameter,
                        PackedvLLMParameter)

logger = logging.getLogger(__name__)


class SglangMarlinConfig(QuantizationConfig):
    """Config class for Marlin.
    Reference: https://github.com/IST-DASLab/marlin/tree/master
    """

    def __init__(
        self,
        group_size: int,
        lm_head_quantized: bool,
    ) -> None:
        self.group_size = group_size
        self.lm_head_quantized = lm_head_quantized
        if self.group_size != 128 and self.group_size != -1:
            raise ValueError(
                "Currently, only group size 128 and -1 (channelwise) "
                "is supported for Marlin, but got group_size of "
                f"{self.group_size}")

        self.pack_factor = 32 // 4 # Marlin is typically 4-bit
        self.tile_size = 16
        self.min_n_threads = 64
        self.min_k_threads = 128
        self.max_parallel = 16
        self.perm_len = 1024

    def __repr__(self) -> str:
        return (f"SglangMarlinConfig(group_size={self.group_size}, "
                f"lm_head_quantized={self.lm_head_quantized})")

    @classmethod
    def get_name(cls) -> str:
        return "marlin"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    def get_min_capability(cls) -> int: # Marlin requires SM 8.0+
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SglangMarlinConfig":
        group_size = cls.get_from_keys(config, ["group_size"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        return cls(group_size, lm_head_quantized)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg,
                                     user_quant) -> Optional[str]:
        is_marlin_format = (hf_quant_cfg.get("checkpoint_format") == "marlin"
                            or hf_quant_cfg.get("is_marlin_format", False))
        is_valid_user_quant = (user_quant is None or user_quant == "gptq"
                               or user_quant == "marlin")
        if is_marlin_format and is_valid_user_quant:
            msg = ("The model is serialized in {} format. Using {} kernel.".
                   format(cls.get_name(), cls.get_name()))
            logger.info(msg)
            return cls.get_name()
        return None

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["SglangMarlinLinearMethod"]:
        if (isinstance(layer, LinearBase) or
            (isinstance(layer, ParallelLMHead) and self.lm_head_quantized)):
            return SglangMarlinLinearMethod(self)
        return None


class SglangMarlinLinearMethod(LinearMethodBase):
    """Linear method for Marlin.
    Args:
        quant_config: The SglangMarlinConfig.
    """

    def __init__(self, quant_config: SglangMarlinConfig):
        self.quant_config = quant_config
        # Platform checks for Marlin would typically go here (rely on current_platform).
        # For structural vendoring, assume supported for now if using placeholders.

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int, # Full model input size
        output_size: int, # Full model output size (unused by this method)
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del output_size
        weight_loader = extra_weight_attrs["weight_loader"]

        if params_dtype != torch.float16: # Marlin kernels are typically fp16
            raise ValueError(
                f"The params dtype must be float16, but got {params_dtype}")

        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.quant_config.min_n_threads != 0:
            raise ValueError(
                f"Weight output_size_per_partition = "
                f"{output_size_per_partition} is not divisible by "
                f"min_n_threads = {self.quant_config.min_n_threads}.")
        if output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                f"Weight output_size_per_partition = "
                f"{output_size_per_partition} is not divisible by "
                f"pack_factor = {self.quant_config.pack_factor}.")

        if input_size_per_partition % self.quant_config.min_k_threads != 0:
            raise ValueError(
                f"Weight input_size_per_partition = "
                f"{input_size_per_partition} is not divisible by "
                f"min_k_threads = {self.quant_config.min_k_threads}.")
        if (self.quant_config.group_size != -1 and
            input_size_per_partition % self.quant_config.group_size != 0):
            raise ValueError(f"Weight input_size_per_partition = "
                             f"{input_size_per_partition} is not divisible by "
                             f"group_size = {self.quant_config.group_size}.")

        num_tiles_per_perm = self.quant_config.perm_len // (
            self.quant_config.tile_size**2)
        if output_size_per_partition % num_tiles_per_perm != 0:
            raise ValueError(
                "Each permutation group must reside on the same gpu.")

        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition // self.quant_config.tile_size,
                output_size_per_partition * self.quant_config.tile_size //
                self.quant_config.pack_factor,
                device="cuda", # Marlin weights are CUDA only
                dtype=torch.int32,
            ),
            input_dim=0, output_dim=1, packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            marlin_tile_size=self.quant_config.tile_size,
            weight_loader=weight_loader)

        input_groups = (1 if self.quant_config.group_size == -1 else
                        input_size_per_partition // self.quant_config.group_size)

        scales_args = {
            "data": torch.empty(
                input_groups,
                output_size_per_partition,
                device="cuda", # Scales also on CUDA
                dtype=params_dtype, # Should be fp16
            ),
            "weight_loader": weight_loader
        }
        if input_groups == 1: # Channelwise
            scales = ChannelQuantScaleParameter(output_dim=1, **scales_args)
        else: # Groupwise
            scales = GroupQuantScaleParameter(output_dim=1, input_dim=0, **scales_args)

        max_workspace_size = (
            output_size_per_partition //
            self.quant_config.min_n_threads) * self.quant_config.max_parallel

        workspace = BasevLLMParameter(data=torch.zeros(max_workspace_size,
                                                       device="cuda", # Workspace on CUDA
                                                       dtype=torch.int32), # Original vLLM used torch.int
                                      weight_loader=weight_loader)

        layer.register_parameter("B", qweight) # qweight for Marlin
        layer.register_parameter("s", scales)  # scales for Marlin
        layer.register_parameter("workspace", workspace)


    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.B = Parameter(layer.B.data, requires_grad=False)
        layer.s = Parameter(layer.s.data, requires_grad=False)
        layer.workspace = Parameter(layer.workspace.data, requires_grad=False)
        # Marlin kernels handle weight permutation internally if needed.

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qweight = layer.B
        scales = layer.s
        workspace = layer.workspace

        x_2d = x.reshape(-1, x.shape[-1])

        size_m = x_2d.shape[0]
        size_k = x_2d.shape[1]
        size_n = scales.shape[1] # output_size_per_partition

        output_2d = ops.marlin_gemm(x_2d, qweight, scales, workspace, size_m,
                                    size_n, size_k)

        output = output_2d.reshape(x.shape[:-1] + (output_2d.shape[1], ))

        if bias is not None:
            output = output + bias # Out-of-place add

        return output

__all__ = ["SglangMarlinConfig", "SglangMarlinLinearMethod"]

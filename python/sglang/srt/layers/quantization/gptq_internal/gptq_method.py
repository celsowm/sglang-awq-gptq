import enum
from fractions import Fraction
from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from .custom_ops_placeholder import ops
from sglang.srt.layers.linear import LinearBase, LinearMethodBase
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.logits_processor import ParallelLMHead
from .parameter import (ChannelQuantScaleParameter,
                        GroupQuantScaleParameter,
                        PackedColumnParameter,
                        PackedvLLMParameter,
                        RowvLLMParameter)


class SglangGPTQConfig(QuantizationConfig):
    """Config class for GPTQ.
    Reference: https://arxiv.org/abs/2210.17323
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        desc_act: bool,
        lm_head_quantized: bool,
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.lm_head_quantized = lm_head_quantized
        if self.weight_bits == 32:
            self.pack_factor = 1
        else:
            self.pack_factor = Fraction(32, self.weight_bits)

        if self.weight_bits not in [2, 3, 4, 8, 32]:
            raise ValueError(
                "Currently, only 2/3/4/8/32-bit weight quantization is "
                f"supported for GPTQ, but got {self.weight_bits} bits.")

    def __repr__(self) -> str:
        return (f"SglangGPTQConfig(weight_bits={self.weight_bits}, "
                f"group_size={self.group_size}, "
                f"desc_act={self.desc_act},"
                f"lm_head_quantized={self.lm_head_quantized})")

    @classmethod
    def get_name(cls) -> str:
        return "gptq"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        # Exllama kernels (used by GPTQ) typically require SM 60+
        return 60

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SglangGPTQConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = cls.get_from_keys(config, ["desc_act"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        return cls(weight_bits, group_size, desc_act, lm_head_quantized)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["SglangGPTQLinearMethod"]:
        if (isinstance(layer, LinearBase) or
            (isinstance(layer, ParallelLMHead) and self.lm_head_quantized)):
            return SglangGPTQLinearMethod(self)
        return None


class SglangExllamaState(enum.Enum):
    UNUSED = enum.auto()
    UNINITIALIZED = enum.auto()
    READY = enum.auto()


class SglangGPTQLinearMethod(LinearMethodBase):
    """Linear method for GPTQ.
    Args:
        quant_config: The SglangGPTQConfig.
    """

    def __init__(self, quant_config: SglangGPTQConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del output_size # Unused by this method
        weight_loader = extra_weight_attrs.get("weight_loader")
        if self.quant_config.group_size != -1 and input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size per partition is not aligned with the group size.")

        output_size_per_partition = sum(output_partition_sizes)
        if (output_size_per_partition % self.quant_config.pack_factor != 0
            if isinstance(self.quant_config.pack_factor, int) else
            output_size_per_partition % self.quant_config.pack_factor.numerator != 0):
            raise ValueError(
                "The output size per partition is not aligned with the pack factor.")

        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else: # Channelwise quantization
            group_size = input_size_per_partition # Adjusted for tensor parallelism

        exllama_state = SglangExllamaState.UNINITIALIZED
        scale_and_zero_size = input_size_per_partition // group_size

        scale_and_zero_input_dim = None
        is_row_parallel = input_size != input_size_per_partition

        if is_row_parallel and self.quant_config.group_size != -1 :
            if self.quant_config.desc_act: # act-order
                exllama_state = SglangExllamaState.UNUSED
            scale_and_zero_input_dim = 0 # Shard scales/zeros for row-parallel

        qweight_packed_factor = self.quant_config.pack_factor
        if isinstance(qweight_packed_factor, Fraction):
             qweight_data_shape_in = input_size_per_partition * qweight_packed_factor.denominator // qweight_packed_factor.numerator
        else:
             qweight_data_shape_in = input_size_per_partition // qweight_packed_factor

        qweight = PackedvLLMParameter(
            data=torch.empty(
                qweight_data_shape_in,
                output_size_per_partition,
                dtype=torch.int32,
            ),
            input_dim=0, output_dim=1, packed_dim=0,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader
        )

        g_idx_data = torch.tensor(
            [i // self.quant_config.group_size for i in range(input_size_per_partition)],
            dtype=torch.int32,
        ) if self.quant_config.group_size != -1 else torch.empty(0, dtype=torch.int32)

        g_idx = RowvLLMParameter(data=g_idx_data, input_dim=0, weight_loader=weight_loader)

        qzeros_packed_factor = self.quant_config.pack_factor
        if isinstance(qzeros_packed_factor, Fraction):
            qzeros_data_shape_out = output_size_per_partition * qzeros_packed_factor.denominator // qzeros_packed_factor.numerator
        else:
            qzeros_data_shape_out = output_size_per_partition // qzeros_packed_factor

        qzeros_args = {
            "data": torch.empty(
                scale_and_zero_size,
                qzeros_data_shape_out,
                dtype=torch.int32),
            "weight_loader": weight_loader
        }

        scales_args = {
            "data": torch.empty(
                scale_and_zero_size,
                output_size_per_partition,
                dtype=params_dtype),
            "weight_loader": weight_loader
        }

        if scale_and_zero_input_dim is None: # ColumnParallel or Unsharded RowParallel scales
            scales = ChannelQuantScaleParameter(output_dim=1, **scales_args)
            qzeros = PackedColumnParameter(
                output_dim=1, packed_dim=1,
                packed_factor=self.quant_config.pack_factor, **qzeros_args)
        else: # Sharded RowParallel scales/zeros
            scales = GroupQuantScaleParameter(output_dim=1, input_dim=0, **scales_args)
            qzeros = PackedvLLMParameter(
                input_dim=0, output_dim=1, packed_dim=1,
                packed_factor=self.quant_config.pack_factor, **qzeros_args)

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("g_idx", g_idx)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)
        layer.exllama_state = exllama_state


    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Ensure parameters are standard torch.nn.Parameter instances for potential torch.compile compatibility
        layer.qweight = Parameter(layer.qweight.data, requires_grad=False)
        layer.qzeros = Parameter(layer.qzeros.data, requires_grad=False)
        layer.g_idx = Parameter(layer.g_idx.data, requires_grad=False)
        layer.scales = Parameter(layer.scales.data, requires_grad=False)

        if layer.exllama_state == SglangExllamaState.UNINITIALIZED:
            if self.quant_config.desc_act and self.quant_config.group_size != -1:
                # g_idx is assumed to be in the permuted order if desc_act is true.
                # The actual shuffle op (placeholder or real) would use this.
                pass
            else: # Not desc_act or channelwise, g_idx is effectively unused by exllama shuffle
                layer.g_idx.data = torch.empty((0,), dtype=torch.int32, device=layer.g_idx.device)

            # Call shuffle op (placeholder or real)
            # Exllama is not used for row-parallel with act_order in vLLM
            if layer.exllama_state == SglangExllamaState.UNINITIALIZED and \
               self.quant_config.group_size != -1 and \
               not (isinstance(layer.g_idx, RowvLLMParameter) and self.quant_config.desc_act):
                ops.gptq_shuffle(layer.qweight, layer.g_idx, self.quant_config.weight_bits)

            layer.exllama_state = SglangExllamaState.READY

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        if layer.g_idx.device != layer.qweight.device: # Ensure g_idx is on the same device
            layer.g_idx = Parameter(layer.g_idx.data.to(layer.qweight.device), requires_grad=False)

        out_shape = x.shape[:-1] + (layer.qweight.shape[-1], )
        reshaped_x = x.reshape(-1, x.shape[-1])

        use_exllama = layer.exllama_state == SglangExllamaState.READY

        output = ops.gptq_gemm(reshaped_x, layer.qweight, layer.qzeros,
                               layer.scales, layer.g_idx,
                               use_exllama,
                               self.quant_config.weight_bits)
        if bias is not None:
            output = output + bias
        return output.reshape(out_shape)

__all__ = ["SglangGPTQConfig", "SglangGPTQLinearMethod", "SglangExllamaState"]

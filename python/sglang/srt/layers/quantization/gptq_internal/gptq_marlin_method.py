from typing import Any, Callable, Dict, List, Optional, Set, Union # Added Set, Union
import torch
import logging

# Imports from this package
from .custom_ops_placeholder import ops, fused_marlin_moe as fused_marlin_moe_placeholder
from .fused_moe_layer import (SglangFusedMoE, SglangFusedMoEMethodBase,
                              SglangFusedMoeWeightScaleSupported)
from .kernel_placeholders import MPLinearLayerConfig, choose_mp_linear_kernel
from .marlin_utils import (check_marlin_supported, marlin_moe_permute_scales,
                           marlin_repeat_scales_on_all_ranks, verify_marlin_supported)
from .parameter import (ChannelQuantScaleParameter, GroupQuantScaleParameter,
                        PackedColumnParameter, PackedvLLMParameter, RowvLLMParameter)
from .scalar_type import sglang_scalar_types
from .utils import replace_parameter, set_weight_attrs

# Imports from sglang base
from sglang.srt.layers.linear import LinearBase, LinearMethodBase
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.logits_processor import ParallelLMHead


logger = logging.getLogger(__name__)


class SglangGPTQMarlinConfig(QuantizationConfig):
    """
    Configuration for GPTQ-Marlin quantization.
    It combines GPTQ for initial quantization and Marlin for efficient kernel execution.
    """
    TYPE_MAP = {
        (4, True): sglang_scalar_types.uint4b8,
        (8, True): sglang_scalar_types.uint8b128,
    }

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        desc_act: bool,
        is_sym: bool,
        lm_head_quantized: bool,
    ) -> None:
        if desc_act and group_size == -1: # Channelwise with desc_act is same as without
            desc_act = False

        self.pack_factor = 32 // weight_bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.is_sym = is_sym
        self.lm_head_quantized = lm_head_quantized

        if (weight_bits, is_sym) not in self.TYPE_MAP:
            raise ValueError("Unsupported quantization config for GPTQ-Marlin: "
                             f"bits={weight_bits}, sym={is_sym}")
        self.quant_type = self.TYPE_MAP[(weight_bits, is_sym)]

    def __repr__(self) -> str:
        return (f"SglangGPTQMarlinConfig(quant_type={self.quant_type}, "
                f"group_size={self.group_size}, "
                f"desc_act={self.desc_act}, "
                f"is_sym={self.is_sym}, "
                f"lm_head_quantized={self.lm_head_quantized})")

    @classmethod
    def get_name(cls) -> str:
        return "gptq_marlin"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int: # Marlin kernels require SM 8.0+
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SglangGPTQMarlinConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = cls.get_from_keys(config, ["desc_act"])
        is_sym = cls.get_from_keys(config, ["sym"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        return cls(weight_bits, group_size, desc_act, is_sym, lm_head_quantized)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg,
                                     user_quant) -> Optional[str]:
        can_convert = cls.is_gptq_marlin_compatible(hf_quant_cfg)
        is_valid_user_quant = (user_quant is None or user_quant == "marlin"
                               or user_quant == "gptq_marlin")
        if can_convert and is_valid_user_quant:
            msg = ("The model is convertible to {} during runtime."
                   " Using {} kernel.".format(cls.get_name(), cls.get_name()))
            logger.info(msg)
            return cls.get_name()
        if can_convert and user_quant == "gptq":
            logger.info("Detected that the model can run with gptq_marlin"
                        ", however you specified quantization=gptq explicitly,"
                        " so forcing gptq. Use quantization=gptq_marlin for"
                        " faster inference.")
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[Union["SglangGPTQMarlinLinearMethod", "SglangGPTQMarlinMoEMethod"]]:
        if isinstance(layer, LinearBase) or \
           (isinstance(layer, ParallelLMHead) and self.lm_head_quantized):
            return SglangGPTQMarlinLinearMethod(self)
        elif isinstance(layer, SglangFusedMoE):
            return SglangGPTQMarlinMoEMethod(self)
        return None

    @classmethod
    def is_gptq_marlin_compatible(cls, quant_config: Dict[str, Any]):
        quant_method = quant_config.get("quant_method", "").lower()
        num_bits = quant_config.get("bits")
        group_size = quant_config.get("group_size")
        sym = quant_config.get("sym")
        desc_act = quant_config.get("desc_act")

        if quant_method != "gptq": return False
        if (num_bits is None or group_size is None or sym is None or desc_act is None): return False
        if (num_bits, sym) not in cls.TYPE_MAP: return False

        return check_marlin_supported(quant_type=cls.TYPE_MAP[(num_bits, sym)],
                                      group_size=group_size)


class SglangGPTQMarlinLinearMethod(LinearMethodBase):
    _kernel_backends_being_used: Set[str] = set()

    def __init__(self, quant_config: SglangGPTQMarlinConfig) -> None:
        self.quant_config = quant_config
        verify_marlin_supported(quant_type=self.quant_config.quant_type,
                                group_size=self.quant_config.group_size)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        is_row_parallel = input_size != input_size_per_partition
        weight_loader = extra_weight_attrs.get("weight_loader")

        mp_linear_kernel_config = MPLinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=(input_size_per_partition, output_size_per_partition),
            weight_type=self.quant_config.quant_type,
            act_type=params_dtype,
            group_size=self.quant_config.group_size,
            zero_points=False,
            has_g_idx=self.quant_config.desc_act
        )
        kernel_type = choose_mp_linear_kernel(mp_linear_kernel_config)

        if kernel_type.__name__ not in self._kernel_backends_being_used:
            logger.info("Using %s for SglangGPTQMarlinLinearMethod", kernel_type.__name__)
            self._kernel_backends_being_used.add(kernel_type.__name__)

        effective_group_size = self.quant_config.group_size
        if effective_group_size == -1: # Channelwise
            effective_group_size = input_size_per_partition

        if marlin_repeat_scales_on_all_ranks(self.quant_config.desc_act,
                                             self.quant_config.group_size,
                                             is_row_parallel):
            scales_and_zp_input_dim = None
            scales_and_zp_size = input_size // effective_group_size
        else:
            scales_and_zp_input_dim = 0
            scales_and_zp_size = input_size_per_partition // effective_group_size

        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition // self.quant_config.pack_factor,
                output_size_per_partition, dtype=torch.int32),
            input_dim=0, output_dim=1, packed_dim=0,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader)

        g_idx = RowvLLMParameter(data=torch.empty(input_size_per_partition, dtype=torch.int32),
                                 input_dim=0, weight_loader=weight_loader)

        qzeros_args = {
            "data": torch.empty(
                scales_and_zp_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32),
            "weight_loader": weight_loader
        }
        scales_args = {
            "data": torch.empty(
                scales_and_zp_size, output_size_per_partition, dtype=params_dtype),
            "weight_loader": weight_loader
        }

        if scales_and_zp_input_dim is None:
            scales = ChannelQuantScaleParameter(output_dim=1, **scales_args)
            qzeros = PackedColumnParameter(
                output_dim=1, packed_dim=1,
                packed_factor=self.quant_config.pack_factor, **qzeros_args)
        else:
            scales = GroupQuantScaleParameter(output_dim=1, input_dim=0, **scales_args)
            qzeros = PackedvLLMParameter(
                input_dim=0, output_dim=1, packed_dim=1,
                packed_factor=self.quant_config.pack_factor, **qzeros_args)

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("g_idx", g_idx)
        layer.register_parameter("scales", scales)
        layer.register_parameter("qzeros", qzeros)

        self.kernel = kernel_type(mp_linear_kernel_config,
                                  w_q_param_name="qweight",
                                  w_s_param_name="scales",
                                  w_zp_param_name="qzeros",
                                  w_gidx_param_name="g_idx")

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)


class SglangGPTQMarlinMoEMethod(SglangFusedMoEMethodBase):
    def __init__(self, quant_config: SglangGPTQMarlinConfig) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if self.quant_config.group_size != -1:
            scales_size13 = hidden_size // self.quant_config.group_size
            scales_size2 = intermediate_size // self.quant_config.group_size
            strategy = SglangFusedMoeWeightScaleSupported.GROUP.value
        else: # Channelwise
            scales_size13 = 1
            scales_size2 = 1
            strategy = SglangFusedMoeWeightScaleSupported.CHANNEL.value

        moe_weight_attrs = extra_weight_attrs.copy()
        moe_weight_attrs.update({
            "quant_method": strategy,
            "is_transposed": True
        })

        w13_qweight = torch.nn.Parameter(torch.empty(
            num_experts, hidden_size // self.quant_config.pack_factor,
            2 * intermediate_size, dtype=torch.int32), requires_grad=False)
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, moe_weight_attrs)

        w2_qweight = torch.nn.Parameter(torch.empty(
            num_experts, intermediate_size // self.quant_config.pack_factor,
            hidden_size, dtype=torch.int32), requires_grad=False)
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, moe_weight_attrs)

        w13_scales = torch.nn.Parameter(torch.empty(
            num_experts, scales_size13, 2 * intermediate_size, dtype=params_dtype), requires_grad=False)
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, moe_weight_attrs)

        w2_scales = torch.nn.Parameter(torch.empty(
            num_experts, scales_size2, hidden_size, dtype=params_dtype), requires_grad=False)
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, moe_weight_attrs)

        # QZeros dtype in vLLM for GPTQMarlin MoE is params_dtype (e.g. fp16).
        # This is kept for structural fidelity, though unusual for typical qzeros.
        w13_qzeros = torch.nn.Parameter(torch.empty(
            num_experts, scales_size13, 2 * intermediate_size // self.quant_config.pack_factor,
            dtype=params_dtype), requires_grad=False)
        layer.register_parameter("w13_qzeros", w13_qzeros)
        set_weight_attrs(w13_qzeros, moe_weight_attrs)

        w2_qzeros = torch.nn.Parameter(torch.empty(
            num_experts, scales_size2, hidden_size // self.quant_config.pack_factor,
            dtype=params_dtype), requires_grad=False)
        layer.register_parameter("w2_qzeros", w2_qzeros)
        set_weight_attrs(w2_qzeros, moe_weight_attrs)

        w13_g_idx = torch.nn.Parameter(torch.empty(
            num_experts, hidden_size, dtype=torch.int32), requires_grad=False)
        layer.register_parameter("w13_g_idx", w13_g_idx)
        set_weight_attrs(w13_g_idx, moe_weight_attrs)

        w2_g_idx = torch.nn.Parameter(torch.empty(
            num_experts, intermediate_size, dtype=torch.int32), requires_grad=False)
        layer.register_parameter("w2_g_idx", w2_g_idx)
        set_weight_attrs(w2_g_idx, moe_weight_attrs)

        w13_g_idx_sort_indices = torch.nn.Parameter(torch.empty(
            num_experts, hidden_size, dtype=torch.int32), requires_grad=False)
        layer.register_parameter("w13_g_idx_sort_indices", w13_g_idx_sort_indices)
        set_weight_attrs(w13_g_idx_sort_indices, moe_weight_attrs)

        w2_g_idx_sort_indices = torch.nn.Parameter(torch.empty(
            num_experts, intermediate_size, dtype=torch.int32), requires_grad=False)
        layer.register_parameter("w2_g_idx_sort_indices", w2_g_idx_sort_indices)
        set_weight_attrs(w2_g_idx_sort_indices, moe_weight_attrs)


    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        num_experts = layer.w13_g_idx.shape[0]
        device = layer.w13_g_idx.device

        if self.quant_config.desc_act:
            w13_g_idx_sort_indices_data = torch.empty_like(layer.w13_g_idx_sort_indices.data)
            w2_g_idx_sort_indices_data = torch.empty_like(layer.w2_g_idx_sort_indices.data)
            w13_sorted_g_idx_data = torch.empty_like(layer.w13_g_idx.data)
            w2_sorted_g_idx_data = torch.empty_like(layer.w2_g_idx.data)

            for e in range(num_experts):
                w13_g_idx_sort_indices_data[e] = torch.argsort(layer.w13_g_idx.data[e]).to(torch.int32)
                w2_g_idx_sort_indices_data[e] = torch.argsort(layer.w2_g_idx.data[e]).to(torch.int32)
                w13_sorted_g_idx_data[e] = layer.w13_g_idx.data[e][w13_g_idx_sort_indices_data[e]]
                w2_sorted_g_idx_data[e] = layer.w2_g_idx.data[e][w2_g_idx_sort_indices_data[e]]

            replace_parameter(layer, "w13_g_idx", w13_sorted_g_idx_data)
            replace_parameter(layer, "w2_g_idx", w2_sorted_g_idx_data)
            replace_parameter(layer, "w13_g_idx_sort_indices", w13_g_idx_sort_indices_data)
            replace_parameter(layer, "w2_g_idx_sort_indices", w2_g_idx_sort_indices_data)
        else:
            empty_g_idx = torch.empty((num_experts, 0), dtype=torch.int32, device=device)
            replace_parameter(layer, "w13_g_idx", empty_g_idx.clone())
            replace_parameter(layer, "w2_g_idx", empty_g_idx.clone())
            replace_parameter(layer, "w13_g_idx_sort_indices", empty_g_idx.clone())
            replace_parameter(layer, "w2_g_idx_sort_indices", empty_g_idx.clone())

        marlin_w13_qweight = ops.gptq_marlin_moe_repack(
            layer.w13_qweight, layer.w13_g_idx_sort_indices,
            layer.w13_qweight.shape[1] * self.quant_config.pack_factor,
            layer.w13_qweight.shape[2],
            self.quant_config.quant_type.size_bits)
        replace_parameter(layer, "w13_qweight", marlin_w13_qweight)

        marlin_w2_qweight = ops.gptq_marlin_moe_repack(
            layer.w2_qweight, layer.w2_g_idx_sort_indices,
            layer.w2_qweight.shape[1] * self.quant_config.pack_factor,
            layer.w2_qweight.shape[2],
            self.quant_config.quant_type.size_bits)
        replace_parameter(layer, "w2_qweight", marlin_w2_qweight)

        marlin_w13_scales = marlin_moe_permute_scales(
            s=layer.w13_scales.data,
            size_k=layer.intermediate_size_per_partition,
            size_n=layer.w13_scales.shape[2],
            group_size=self.quant_config.group_size)
        replace_parameter(layer, "w13_scales", marlin_w13_scales)

        marlin_w2_scales = marlin_moe_permute_scales(
            s=layer.w2_scales.data,
            size_k=layer.w2_scales.shape[1] * self.quant_config.pack_factor,
            size_n=layer.w2_scales.shape[2],
            group_size=self.quant_config.group_size)
        replace_parameter(layer, "w2_scales", marlin_w2_scales)


    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.half() # Marlin MoE kernel expects fp16 input

        topk_weights, topk_ids = SglangFusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function)

        # The placeholder op is imported as fused_marlin_moe_placeholder
        output = fused_marlin_moe_placeholder(
            x,
            layer.w13_qweight, layer.w2_qweight,
            layer.w13_scales, layer.w2_scales,
            # Router logits not passed to vLLM kernel; topk_weights & topk_ids are.
            topk_weights, topk_ids,
            g_idx1=layer.w13_g_idx, g_idx2=layer.w2_g_idx,
            sort_indices1=layer.w13_g_idx_sort_indices,
            sort_indices2=layer.w2_g_idx_sort_indices,
            num_bits=self.quant_config.quant_type.size_bits,
        )
        return output.to(orig_dtype)

__all__ = ["SglangGPTQMarlinConfig", "SglangGPTQMarlinLinearMethod", "SglangGPTQMarlinMoEMethod"]

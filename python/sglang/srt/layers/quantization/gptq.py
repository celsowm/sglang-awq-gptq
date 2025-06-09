import logging
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Union

import torch

from sglang.srt.layers.linear import LinearBase, set_weight_attrs
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.utils import replace_parameter
from sglang.srt.utils import is_cuda

# Imports from .gptq_internal
from .gptq_internal.custom_ops_placeholder import VLLM_KERNELS_AVAILABLE
from .gptq_internal import (
    SglangGPTQLinearMethod,
    SglangMarlinLinearMethod,
    SglangGPTQMarlinLinearMethod,
    SglangGPTQMarlinMoEMethod as SglangInternalGPTQMarlinMoEMethod, # Renamed to avoid conflict if local def is primary
    SglangFusedMoEMethodBase,
    SglangFusedMoeWeightScaleSupported,
    marlin_moe_permute_scales,
    check_marlin_supported,
    sglang_scalar_types,
)
from .gptq_internal.custom_ops_placeholder import ops as gptq_internal_ops
from .gptq_internal.custom_ops_placeholder import fused_marlin_moe as fused_marlin_moe_placeholder_op


_is_cuda = is_cuda()

logger = logging.getLogger(__name__)


def check_marlin_format(hf_quant_cfg: Dict[str, Any]) -> bool:
    return hf_quant_cfg.get("checkpoint_format") == "marlin" or hf_quant_cfg.get(
        "is_marlin_format", False
    )


class GPTQConfig(QuantizationConfig):
    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        desc_act: bool,
        lm_head_quantized: bool,
        dynamic: Dict[str, Dict[str, Union[int, bool]]],
    ) -> None:
        # GPTQModel use `dynamic` config property to allow per module
        # quantization config so each module can be individually optimized.
        # Format is Dict[str, Dict] where key is a regex string that can
        # perform both positive ("+:" prefixed) or negative ("-:" prefixed)
        # matching of a module.
        # Default to positive match, override base quant config mode, if no
        # prefix is used. Value is in dict format of field key and override
        # value.
        # Negative matching will skip quantization init for this module
        # entirely:
        # non-quantized inference. More details and quantization examples can be
        # found at: https://github.com/ModelCloud/GPTQModel
        # Example:
        #  # last 1/2 of the layers 10-21 has 8bit vs 4bit for 0-9
        #  # last 1/4 of the layers 16-21 has 8bit and group_size 64
        # dynamic = {
        #  #`.*\.` matches the layers_node prefix
        #  # positive match layer 10-15
        #  r"+:.*\.(?:1[0-5])\..*": {"bits": 8,},
        #  # positive match layer 16-21
        #  r"+:.*\.(?:1[6-9]|20|21)\..*": {"bits": 8, "group_size": 64,},
        #  r"-:.*\.moe\..*": {}, # negative match (skip) all `moe` layers
        # }
        super().__init__()
        self.dynamic = dynamic
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
                f"supported for GPTQ, but got {self.weight_bits} bits."
            )

    def __repr__(self) -> str:
        return (
            f"GPTQConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, "
            f"desc_act={self.desc_act}, "
            f"lm_head_quantized={self.lm_head_quantized}, "
            f"dynamic={self.dynamic})"
        )

    def get_scaled_act_names(self) -> List[str]:
        raise NotImplementedError

    @classmethod
    def get_name(cls) -> str:
        return "gptq"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GPTQConfig":
        dynamic = cls.get_from_keys_or(config, ["dynamic"], default={})
        dynamic = {} if dynamic is None else dynamic
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = cls.get_from_keys(config, ["desc_act"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        return cls(weight_bits, group_size, desc_act, lm_head_quantized, dynamic)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[SglangGPTQLinearMethod]:
        from sglang.srt.layers.quantization import get_linear_quant_method
        return get_linear_quant_method(self, layer, prefix, SglangGPTQLinearMethod)


class GPTQMarlinConfig(QuantizationConfig):
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
        dynamic: Dict[str, Dict[str, Union[int, bool]]],
        full_config: Dict[str, Any],
    ) -> None:
        # (Same detailed comment block for `dynamic` as in GPTQConfig)
        # GPTQModel use `dynamic` config property to allow per module
        # quantization config so each module can be individually optimized.
        # ... (rest of the comment block) ...
        super().__init__()
        if desc_act and group_size == -1:
            desc_act = False
        self.dynamic = dynamic
        self.weight_bits = weight_bits
        self.is_sym = is_sym
        self.pack_factor = 32 // weight_bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.lm_head_quantized = lm_head_quantized
        self.full_config = full_config

        if (weight_bits, is_sym) not in self.TYPE_MAP:
            raise ValueError(
                f"Unsupported quantization config: bits={weight_bits}, sym={is_sym}"
            )
        self.quant_type = self.TYPE_MAP[(weight_bits, is_sym)]

    def __repr__(self) -> str:
        return (
            f"GPTQMarlinConfig(quant_type={self.quant_type}, "
            f"group_size={self.group_size}, "
            f"desc_act={self.desc_act}, "
            f"lm_head_quantized={self.lm_head_quantized}, "
            f"dynamic={self.dynamic})"
        )

    def get_scaled_act_names(self) -> List[str]:
        raise NotImplementedError

    @classmethod
    def get_name(cls) -> str:
        return "gptq_marlin"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GPTQMarlinConfig":
        dynamic = cls.get_from_keys_or(config, ["dynamic"], default={})
        dynamic = {} if dynamic is None else dynamic
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = cls.get_from_keys(config, ["desc_act"])
        is_sym = cls.get_from_keys(config, ["sym"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        return cls(
            weight_bits, group_size, desc_act, is_sym,
            lm_head_quantized, dynamic, config,
        )

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        is_marlin_format = check_marlin_format(hf_quant_cfg)
        can_convert = cls.is_gptq_marlin_compatible(hf_quant_cfg)
        is_valid_user_quant = (
            user_quant is None or user_quant == "marlin" or user_quant == "gptq_marlin"
        )
        if not is_marlin_format and can_convert and is_valid_user_quant:
            msg = (
                "The model is convertible to {} during runtime."
                " Using {} kernel.".format(cls.get_name(), cls.get_name())
            )
            logger.info(msg)
            return cls.get_name()
        if not is_marlin_format and can_convert and user_quant == "gptq":
            logger.info(
                "Detected that the model can run with gptq_marlin"
                ", however you specified quantization=gptq explicitly,"
                " so forcing gptq. Use quantization=gptq_marlin for"
                " faster inference"
            )
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[Union[SglangGPTQMarlinLinearMethod, SglangInternalGPTQMarlinMoEMethod]]: # Use aliased MoE method
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.quantization import get_linear_quant_method

        if isinstance(layer, FusedMoE):
            # If current file defines GPTQMarlinMoEMethod, use that. Otherwise, use SglangInternal...
            return GPTQMarlinMoEMethod(self) if "GPTQMarlinMoEMethod" in globals() else SglangInternalGPTQMarlinMoEMethod(self)
        return get_linear_quant_method(self, layer, prefix, SglangGPTQMarlinLinearMethod)

    @classmethod
    def is_gptq_marlin_compatible(cls, quant_config: Dict[str, Any]):
        if not VLLM_KERNELS_AVAILABLE:
            return False
        if not _is_cuda: return False

        quant_method = quant_config.get("quant_method", "").lower()
        num_bits = quant_config.get("bits")
        group_size = quant_config.get("group_size")
        sym = quant_config.get("sym")
        desc_act = quant_config.get("desc_act")

        if quant_method != "gptq": return False
        if num_bits is None or group_size is None or sym is None or desc_act is None: return False

        if (num_bits, sym) not in cls.TYPE_MAP: return False

        return check_marlin_supported(
            quant_type=cls.TYPE_MAP[(num_bits, sym)], group_size=group_size
        )


class MarlinConfig(QuantizationConfig):
    def __init__(
        self,
        group_size: int,
        lm_head_quantized: bool,
    ) -> None:
        super().__init__()
        self.group_size = group_size
        self.lm_head_quantized = lm_head_quantized
        if self.group_size != 128 and self.group_size != -1:
            raise ValueError(
                "Currently, only group size 128 and -1 (channelwise) "
                "is supported for Marlin, but got group_size of "
                f"{self.group_size}"
            )
        # 4 Bits packed into 32 bit datatype.
        self.pack_factor = 32 // 4
        # Tile size used by marlin kernels.
        self.tile_size = 16
        # Min out_features dim
        self.min_n_threads = 64
        # Min in_features dim
        self.min_k_threads = 128
        # Max parallel problems to solve at once
        self.max_parallel = 16
        # Permutation length used by the marlin kernels.
        self.perm_len = 1024

    def __repr__(self) -> str:
        return (
            f"MarlinConfig(group_size={self.group_size}, "
            f"lm_head_quantized={self.lm_head_quantized})"
        )

    def get_scaled_act_names(self) -> List[str]:
        raise NotImplementedError

    @classmethod
    def get_name(cls) -> str:
        return "marlin"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MarlinConfig":
        group_size = cls.get_from_keys(config, ["group_size"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        return cls(group_size, lm_head_quantized)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        is_marlin_format = check_marlin_format(hf_quant_cfg)
        is_valid_user_quant = (
            user_quant is None or user_quant == "gptq" or user_quant == "marlin"
        )
        if is_marlin_format and is_valid_user_quant:
            msg = "The model is serialized in {} format. Using {} kernel.".format(
                cls.get_name(), cls.get_name()
            )
            logger.info(msg)
            return cls.get_name()
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[SglangMarlinLinearMethod]:
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
        if isinstance(layer, LinearBase) or (
            isinstance(layer, ParallelLMHead) and self.lm_head_quantized
        ):
            return SglangMarlinLinearMethod(self)
        return None


class GPTQMarlinMoEMethod(SglangFusedMoEMethodBase):
    def __init__(self, quant_config: GPTQMarlinConfig) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # intermediate_size is the full intermediate size
        intermediate_size = extra_weight_attrs.pop("intermediate_size", intermediate_size_per_partition * layer.tp_size) # type: ignore

        self.is_k_full = (not self.quant_config.desc_act) or (
            intermediate_size_per_partition == intermediate_size
        )

        if self.quant_config.group_size != -1:
            scales_size13 = hidden_size // self.quant_config.group_size
            w2_scales_input_dim = intermediate_size if self.quant_config.desc_act else intermediate_size_per_partition
            scales_size2 = w2_scales_input_dim // self.quant_config.group_size
            strategy = SglangFusedMoeWeightScaleSupported.GROUP.value
        else: # Channelwise
            scales_size13 = 1
            scales_size2 = 1
            strategy = SglangFusedMoeWeightScaleSupported.CHANNEL.value

        moe_attrs = extra_weight_attrs.copy()
        moe_attrs.update({"quant_method": strategy, "is_transposed": True})

        layer.w13_qweight = torch.nn.Parameter(torch.empty(num_experts, hidden_size // self.quant_config.pack_factor, 2 * intermediate_size_per_partition, dtype=torch.int32), requires_grad=False)
        set_weight_attrs(layer.w13_qweight, moe_attrs)
        layer.w2_qweight = torch.nn.Parameter(torch.empty(num_experts, intermediate_size_per_partition // self.quant_config.pack_factor, hidden_size, dtype=torch.int32), requires_grad=False)
        set_weight_attrs(layer.w2_qweight, moe_attrs)
        layer.w13_scales = torch.nn.Parameter(torch.empty(num_experts, scales_size13, 2 * intermediate_size_per_partition, dtype=params_dtype), requires_grad=False)
        set_weight_attrs(layer.w13_scales, moe_attrs)
        layer.w2_scales = torch.nn.Parameter(torch.empty(num_experts, scales_size2, hidden_size, dtype=params_dtype), requires_grad=False)
        set_weight_attrs(layer.w2_scales, moe_attrs)
        set_weight_attrs(layer.w2_scales, {"load_full_w2": self.quant_config.desc_act})

        # Qzeros dtype was params_dtype in vLLM, maintained for structural fidelity.
        layer.w13_qzeros = torch.nn.Parameter(torch.empty(num_experts, scales_size13, 2 * intermediate_size_per_partition // self.quant_config.pack_factor, dtype=params_dtype), requires_grad=False)
        set_weight_attrs(layer.w13_qzeros, moe_attrs)
        layer.w2_qzeros = torch.nn.Parameter(torch.empty(num_experts, scales_size2, hidden_size // self.quant_config.pack_factor, dtype=params_dtype), requires_grad=False)
        set_weight_attrs(layer.w2_qzeros, moe_attrs)
        set_weight_attrs(layer.w2_qzeros, {"load_full_w2": self.quant_config.desc_act})

        layer.w13_g_idx = torch.nn.Parameter(torch.empty(num_experts, hidden_size, dtype=torch.int32), requires_grad=False)
        set_weight_attrs(layer.w13_g_idx, moe_attrs)
        layer.w2_g_idx = torch.nn.Parameter(torch.empty(num_experts, intermediate_size_per_partition, dtype=torch.int32), requires_grad=False)
        set_weight_attrs(layer.w2_g_idx, moe_attrs)
        layer.w13_g_idx_sort_indices = torch.nn.Parameter(torch.empty(num_experts, hidden_size, dtype=torch.int32), requires_grad=False)
        set_weight_attrs(layer.w13_g_idx_sort_indices, moe_attrs)
        layer.w2_g_idx_sort_indices = torch.nn.Parameter(torch.empty(num_experts, intermediate_size_per_partition, dtype=torch.int32), requires_grad=False)
        set_weight_attrs(layer.w2_g_idx_sort_indices, moe_attrs)


    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        num_experts = layer.w13_g_idx.shape[0]
        device = layer.w13_g_idx.device
        if self.quant_config.desc_act:
            w13_g_idx_sort_indices = torch.empty_like(layer.w13_g_idx)
            w2_g_idx_sort_indices = torch.empty_like(layer.w2_g_idx)
            w13_sorted_g_idx = torch.empty_like(layer.w13_g_idx)
            w2_sorted_g_idx = torch.empty_like(layer.w2_g_idx)
            for e in range(num_experts):
                w13_g_idx_sort_indices[e] = torch.argsort(layer.w13_g_idx.data[e]).to(torch.int32)
                w2_g_idx_sort_indices[e] = torch.argsort(layer.w2_g_idx.data[e]).to(torch.int32)
                w13_sorted_g_idx[e] = layer.w13_g_idx.data[e][w13_g_idx_sort_indices[e]]
                w2_sorted_g_idx[e] = layer.w2_g_idx.data[e][w2_g_idx_sort_indices[e]]
            replace_parameter(layer, "w13_g_idx", w13_sorted_g_idx)
            replace_parameter(layer, "w2_g_idx", w2_sorted_g_idx)
            replace_parameter(layer, "w13_g_idx_sort_indices", w13_g_idx_sort_indices)
            replace_parameter(layer, "w2_g_idx_sort_indices", w2_g_idx_sort_indices)
        else:
            empty_g_idx = torch.empty((num_experts, 0), dtype=torch.int32, device=device)
            replace_parameter(layer, "w13_g_idx", empty_g_idx.clone())
            replace_parameter(layer, "w2_g_idx", empty_g_idx.clone())
            replace_parameter(layer, "w13_g_idx_sort_indices", empty_g_idx.clone())
            replace_parameter(layer, "w2_g_idx_sort_indices", empty_g_idx.clone())

        marlin_w13_qweight = gptq_internal_ops.gptq_marlin_moe_repack(
            layer.w13_qweight, layer.w13_g_idx_sort_indices,
            layer.w13_qweight.shape[1] * self.quant_config.pack_factor,
            layer.w13_qweight.shape[2], self.quant_config.quant_type.size_bits)
        replace_parameter(layer, "w13_qweight", marlin_w13_qweight)
        marlin_w2_qweight = gptq_internal_ops.gptq_marlin_moe_repack(
            layer.w2_qweight, layer.w2_g_idx_sort_indices,
            layer.w2_qweight.shape[1] * self.quant_config.pack_factor,
            layer.w2_qweight.shape[2], self.quant_config.quant_type.size_bits)
        replace_parameter(layer, "w2_qweight", marlin_w2_qweight)

        marlin_w13_scales = marlin_moe_permute_scales(
            s=layer.w13_scales.data, size_k=layer.intermediate_size_per_partition,
            size_n=layer.w13_scales.shape[2], group_size=self.quant_config.group_size)
        replace_parameter(layer, "w13_scales", marlin_w13_scales)

        size_k_w2_scales = layer.w2_scales.shape[1] * \
            (self.quant_config.group_size if self.quant_config.group_size != -1 else self.quant_config.pack_factor)

        marlin_w2_scales = marlin_moe_permute_scales(
            s=layer.w2_scales.data, size_k=size_k_w2_scales,
            size_n=layer.w2_scales.shape[2], group_size=self.quant_config.group_size)
        replace_parameter(layer, "w2_scales", marlin_w2_scales)


    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
    ) -> torch.Tensor:

        orig_dtype = x.dtype
        x = x.half()

        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE as SglangFMoE

        topk_weights, topk_ids = SglangFMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
        )

        output = fused_marlin_moe_placeholder_op(
            x,
            layer.w13_qweight, layer.w2_qweight,
            layer.w13_scales, layer.w2_scales,
            topk_weights, topk_ids,
            g_idx1=layer.w13_g_idx, g_idx2=layer.w2_g_idx,
            sort_indices1=layer.w13_g_idx_sort_indices,
            sort_indices2=layer.w2_g_idx_sort_indices,
            num_bits=self.quant_config.quant_type.size_bits,
            is_k_full=self.is_k_full,
        )
        return output.to(orig_dtype)

__all__ = ["GPTQConfig", "GPTQMarlinConfig", "MarlinConfig", "GPTQMarlinMoEMethod"]

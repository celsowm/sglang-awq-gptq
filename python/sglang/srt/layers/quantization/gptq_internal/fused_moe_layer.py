from abc import abstractmethod
from enum import Enum
from typing import Callable, List, Optional, Tuple

import torch
import logging # Added

# from vllm.distributed import (get_tensor_model_parallel_rank,
#                               get_tensor_model_parallel_world_size,
#                               tensor_model_parallel_all_reduce)
# Using placeholder for distributed utilities.
# Sglang should replace these with its own distributed management if different.
from torch.distributed import get_rank as get_tensor_model_parallel_rank
from torch.distributed import get_world_size as get_tensor_model_parallel_world_size
def tensor_model_parallel_all_reduce(x):
    # print("Warning: tensor_model_parallel_all_reduce placeholder returns input without reduction.")
    return x


logger = logging.getLogger(__name__)

# sglang.srt.layers.quantization.base_config.QuantizationConfig is the expected base.
# MinimalQuantizeMethodBase is a placeholder if sglang's QuantizeMethodBase isn't suitable.
# For MoE, SglangFusedMoEMethodBase defines its own abstract methods, so direct inheritance
# from object or a very basic base class is also an option.
from sglang.srt.layers.quantization.base_config import QuantizationConfig
class MinimalQuantizeMethodBase:
    pass


from .utils import set_weight_attrs
from .platforms_placeholder import current_platform
from .custom_ops_placeholder import ops as custom_ops


class SglangFusedMoeWeightScaleSupported(Enum):
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"


class SglangFusedMoEMethodBase(MinimalQuantizeMethodBase):
    @abstractmethod
    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        raise NotImplementedError

    @abstractmethod
    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              router_logits: torch.Tensor, top_k: int, renormalize: bool,
              use_grouped_topk: bool,
              topk_group: Optional[int] = None,
              num_expert_group: Optional[int] = None,
              custom_routing_function: Optional[Callable] = None
             ) -> torch.Tensor:
        raise NotImplementedError


class SglangUnquantizedFusedMoEMethod(SglangFusedMoEMethodBase):
    """MoE method without quantization."""

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        w13_weight = torch.nn.Parameter(torch.empty(num_experts,
                                                    2 * intermediate_size,
                                                    hidden_size,
                                                    dtype=params_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, {"weight_loader": extra_weight_attrs.get("weight_loader")})

        w2_weight = torch.nn.Parameter(torch.empty(num_experts,
                                                   hidden_size,
                                                   intermediate_size,
                                                   dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, {"weight_loader": extra_weight_attrs.get("weight_loader")})

    def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            router_logits: torch.Tensor,
            top_k: int,
            renormalize: bool,
            use_grouped_topk: bool,
            topk_group: Optional[int] = None,
            num_expert_group: Optional[int] = None,
            custom_routing_function: Optional[Callable] = None
    ) -> torch.Tensor:
        # Adapted from vLLM UnquantizedFusedMoEMethod.forward_cuda
        # Assumes CUDA context for sglang.

        topk_weights, topk_ids = SglangFusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function)

        return custom_ops.fused_experts(hidden_states=x, # type: ignore
                                        w1=layer.w13_weight,
                                        w2=layer.w2_weight,
                                        topk_weights=topk_weights,
                                        topk_ids=topk_ids,
                                        inplace=True)


class SglangFusedMoE(torch.nn.Module):
    """
    Vendored FusedMoE layer for MoE models.
    Adapted from vLLM's FusedMoE.
    """
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        prefix: str = "",
        custom_routing_function: Optional[Callable] = None,
    ):
        super().__init__()

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.tp_size = (tp_size if tp_size is not None else
                        get_tensor_model_parallel_world_size())
        self.top_k = top_k
        self.num_experts = num_experts
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function

        if quant_config is None:
            self.quant_method: Optional[SglangFusedMoEMethodBase] = (
                SglangUnquantizedFusedMoEMethod())
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix) # type: ignore
        assert self.quant_method is not None

        self.quant_method.create_weights(
            layer=self,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=self.intermediate_size_per_partition,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader # Pass instance method as loader
        )

    def weight_loader(self, param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor, weight_name: str,
                      shard_id: str, expert_id: int) -> None:
        # This is a dummy weight_loader.
        # Sglang will use its own mechanisms for loading weights into the parameters
        # created by `self.quant_method.create_weights`.
        # This method exists to satisfy the `weight_loader` argument in `create_weights`
        # if the vendored `set_weight_attrs` or underlying parameter classes expect it.
        pass


    @staticmethod
    def select_experts(hidden_states: torch.Tensor,
                       router_logits: torch.Tensor,
                       top_k: int,
                       use_grouped_topk: bool,
                       renormalize: bool,
                       topk_group: Optional[int] = None,
                       num_expert_group: Optional[int] = None,
                       custom_routing_function: Optional[Callable] = None):
        if use_grouped_topk:
            assert topk_group is not None
            assert num_expert_group is not None
            # Assumes custom_ops.grouped_topk is available via placeholder
            topk_weights, topk_ids = custom_ops.grouped_topk(hidden_states=hidden_states, # type: ignore
                                                           gating_output=router_logits,
                                                           topk=top_k,
                                                           renormalize=renormalize,
                                                           num_expert_group=num_expert_group,
                                                           topk_group=topk_group)
        elif custom_routing_function is None:
            topk_weights, topk_ids = custom_ops.fused_topk(hidden_states=hidden_states, # type: ignore
                                                           gating_output=router_logits,
                                                           topk=top_k,
                                                           renormalize=renormalize)
        else:
            topk_weights, topk_ids = custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize)
        return topk_weights, topk_ids

    def forward(self, hidden_states: torch.Tensor,
                router_logits: torch.Tensor):
        assert self.quant_method is not None

        final_hidden_states = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function)

        if self.reduce_results and self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)

        return final_hidden_states

    @classmethod
    def make_expert_params_mapping(
            cls, ckpt_gate_proj_name: str, ckpt_down_proj_name: str,
            ckpt_up_proj_name: str,
            num_experts: int) -> List[Tuple[str, str, int, str]]:
        # This mapping is for vLLM's weight loading.
        # May not be directly applicable to sglang's weight loading.
        return [
            ("experts.w13_" if weight_name
             in [ckpt_gate_proj_name, ckpt_up_proj_name] else "experts.w2_",
             f"experts.{expert_id}.{weight_name}.", expert_id, shard_id)
            for expert_id in range(num_experts) for shard_id, weight_name in [
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            ]
        ]

__all__ = [
    "SglangFusedMoeWeightScaleSupported",
    "SglangFusedMoEMethodBase",
    "SglangUnquantizedFusedMoEMethod",
    "SglangFusedMoE",
]

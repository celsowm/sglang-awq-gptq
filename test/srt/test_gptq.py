import unittest
import torch

from sglang.srt.layers.quantization.gptq import (
    GPTQConfig,
    GPTQMarlinConfig,
    MarlinConfig,
)
# Assuming VLLM_KERNELS_AVAILABLE is now correctly exposed by custom_ops_placeholder
from sglang.srt.layers.quantization.gptq_internal.custom_ops_placeholder import (
    VLLM_KERNELS_AVAILABLE,
)
from sglang.srt.layers.quantization.gptq_internal import (
    SglangGPTQLinearMethod,
    SglangMarlinLinearMethod,
    SglangGPTQMarlinLinearMethod,
    sglang_scalar_types,
)
from sglang.srt.layers.linear import LinearBase
from sglang.srt.utils import is_cuda as _is_cuda

class TestGPTQMethods(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = "cuda" if torch.cuda.is_available() and _is_cuda() else "cpu"
        if self.device == "cpu":
            self.skipTest("Quantization tests require CUDA.")

        # Dimensions need to be compatible with Marlin constraints if testing Marlin
        # Marlin: K >= 128, N >= 64 (prefer >=256 for some kernels)
        # GPTQ group_size typically 128. Input_size should be multiple of group_size.
        self.input_size = 256  # Multiple of 128 for group size, >=128 for Marlin K
        self.output_size = 512 # Multiple of 64/256 for Marlin N
        self.batch_size = 2

        self.input_size_per_partition = self.input_size
        self.output_partition_sizes = [self.output_size]
        self.params_dtype = torch.float16

    def _populate_gptq_params(self, layer, config: GPTQConfig):
        # qweight: (input_size_per_partition // pack_factor, output_size_per_partition)
        # qzeros: (input_size_per_partition // group_size, output_size_per_partition // pack_factor)
        # scales: (input_size_per_partition // group_size, output_size_per_partition)
        # g_idx: (input_size_per_partition)
        pack_factor = int(config.pack_factor) if isinstance(config.pack_factor, int) else \
                      config.pack_factor.numerator // config.pack_factor.denominator

        qweight_shape = (self.input_size_per_partition // pack_factor, self.output_size)
        layer.qweight.data.uniform_(-1, 1).round_().to(torch.int32) # Dummy int32 data

        if config.group_size != -1 :
            qzeros_shape = (self.input_size_per_partition // config.group_size, self.output_size // pack_factor)
            scales_shape = (self.input_size_per_partition // config.group_size, self.output_size)
            g_idx_shape = (self.input_size_per_partition,)
            layer.qzeros.data.uniform_(-1,1).round_().to(torch.int32)
            layer.scales.data.uniform_(-0.1, 0.1).to(self.params_dtype)
            layer.g_idx.data = torch.arange(g_idx_shape[0], device=self.device) // config.group_size

        else: # Channelwise, different shapes for qzeros/scales, g_idx might be empty
            qzeros_shape = (1, self.output_size // pack_factor) # Simplified for channelwise
            scales_shape = (1, self.output_size) # Simplified for channelwise
            layer.qzeros.data.uniform_(-1,1).round_().to(torch.int32)
            layer.scales.data.uniform_(-0.1, 0.1).to(self.params_dtype)
            # g_idx is often empty or unused for channelwise if not desc_act
            if hasattr(layer, 'g_idx') and layer.g_idx is not None:
                 layer.g_idx.data = torch.empty((0,), dtype=torch.int32, device=self.device)


    def _populate_marlin_params(self, layer, config: MarlinConfig):
        # B (qweight): (input_size // tile_size, output_size * tile_size // pack_factor)
        # s (scales): ( (input_size // group_size) or 1, output_size )
        # workspace: (output_size // min_n_threads) * max_parallel
        tile_size = config.tile_size
        pack_factor = config.pack_factor

        B_shape = (self.input_size // tile_size, self.output_size * tile_size // pack_factor)
        layer.B.data.uniform_(-1, 1).round_().to(torch.int32)

        num_groups = 1 if config.group_size == -1 else self.input_size // config.group_size
        s_shape = (num_groups, self.output_size)
        layer.s.data.uniform_(-0.01, 0.01).to(self.params_dtype)
        # Workspace is initialized with zeros, which is fine.

    def _populate_gptq_marlin_params(self, layer, config: GPTQMarlinConfig):
        # Similar to GPTQ for qweight, qzeros, scales, g_idx initially
        # These are then processed by process_weights_after_loading
        pack_factor = config.pack_factor

        qweight_shape = (self.input_size_per_partition // pack_factor, self.output_size)
        layer.qweight.data.uniform_(-1, 1).round_().to(torch.int32)

        effective_group_size = config.group_size if config.group_size != -1 else self.input_size_per_partition

        # scales_and_zp_size logic from GPTQMarlinLinearMethod
        is_row_parallel = self.input_size != self.input_size_per_partition
        from sglang.srt.layers.quantization.gptq_internal.marlin_utils import marlin_repeat_scales_on_all_ranks
        if marlin_repeat_scales_on_all_ranks(config.desc_act, config.group_size, is_row_parallel):
            scales_and_zp_size = self.input_size // effective_group_size
        else:
            scales_and_zp_size = self.input_size_per_partition // effective_group_size

        qzeros_shape = (scales_and_zp_size, self.output_size // pack_factor)
        scales_shape = (scales_and_zp_size, self.output_size)
        g_idx_shape = (self.input_size_per_partition,)

        layer.qzeros.data.uniform_(-1,1).round_().to(torch.int32)
        layer.scales.data.uniform_(-0.1, 0.1).to(self.params_dtype)
        if config.desc_act:
            # Dummy g_idx, sort order will be based on this if process_weights_after_loading were real
            layer.g_idx.data = torch.randperm(g_idx_shape[0], device=self.device).to(torch.int32)
        else:
            # Monotonic g_idx if not desc_act
            layer.g_idx.data = torch.arange(g_idx_shape[0], device=self.device) // \
                               (config.group_size if config.group_size != -1 else g_idx_shape[0])


    def _setup_linear_layer(self, config_class, method_class, populate_params_func, **config_kwargs):
        config = config_class(**config_kwargs)
        layer = LinearBase(self.input_size, self.output_size, bias=False, params_dtype=self.params_dtype).to(self.device)
        method = method_class(config)

        # create_weights will initialize parameters with empty tensors
        method.create_weights(
            layer, self.input_size_per_partition, self.output_partition_sizes,
            self.input_size, self.output_size, self.params_dtype,
            # Provide dummy weight_loader if methods expect it (they do via BasevLLMParameter)
            weight_loader = lambda p, l, *a, **kw: None
        )

        # Populate with some data
        populate_params_func(layer, config)

        # process_weights_after_loading is crucial for some methods (e.g., GPTQ, GPTQ-Marlin)
        # It might call placeholder ops which could raise NotImplementedError
        # If VLLM_KERNELS_AVAILABLE is false, this will likely hit a placeholder.
        if VLLM_KERNELS_AVAILABLE:
            try:
                method.process_weights_after_loading(layer)
            except NotImplementedError:
                # This is expected if a specific op (e.g., gptq_shuffle) inside process_weights is a placeholder
                # For tests that check 'apply', this means weights might not be fully processed for real kernels.
                pass # Allow test to proceed to apply() to check its specific op call
        else:
            # If kernels are not available, process_weights_after_loading will definitely hit placeholders.
            # We want to test if apply() then correctly raises or is skipped.
            # So, we catch the expected NotImplementedError here.
            with self.assertRaises(NotImplementedError, msg=f"process_weights_after_loading for {method_class.__name__} should raise with no kernels"):
                 method.process_weights_after_loading(layer)


        input_x = torch.randn(self.batch_size, self.input_size, dtype=self.params_dtype, device=self.device)
        return layer, method, input_x

    # --- GPTQ Tests ---
    def test_gptq_method_instantiation(self):
        config = GPTQConfig(weight_bits=4, group_size=128, desc_act=False, lm_head_quantized=False, dynamic={})
        method = config.get_quant_method(LinearBase(128, 128), "test")
        self.assertIsInstance(method, SglangGPTQLinearMethod)

    def _run_apply_test(self, config_class, method_class, populate_func, config_kwargs):
        layer, method, x = self._setup_linear_layer(config_class, method_class, populate_func, **config_kwargs)
        output = method.apply(layer, x)
        self.assertEqual(output.shape, (self.batch_size, self.output_size))
        self.assertEqual(output.dtype, self.params_dtype)

    def _run_apply_raises_test(self, config_class, method_class, populate_func, config_kwargs):
        if VLLM_KERNELS_AVAILABLE:
            self.skipTest("Skipping this test because vLLM kernels/ops ARE available.")

        # Setup will call process_weights_after_loading, which itself might raise.
        # The goal is to check if apply() raises when kernels are not available.
        # _setup_linear_layer already handles expected errors from process_weights_after_loading.
        layer, method, x = self._setup_linear_layer(config_class, method_class, populate_func, **config_kwargs)

        with self.assertRaises(NotImplementedError):
            method.apply(layer, x)

    @unittest.skipUnless(VLLM_KERNELS_AVAILABLE, "vLLM kernels not available for GPTQ apply test")
    def test_gptq_method_apply_with_kernels(self):
        self._run_apply_test(GPTQConfig, SglangGPTQLinearMethod, self._populate_gptq_params,
                             weight_bits=4, group_size=128, desc_act=False, lm_head_quantized=False, dynamic={})

    def test_gptq_method_apply_raises_if_kernels_unavailable(self):
        self._run_apply_raises_test(GPTQConfig, SglangGPTQLinearMethod, self._populate_gptq_params,
                                   weight_bits=4, group_size=128, desc_act=False, lm_head_quantized=False, dynamic={})

    # --- Marlin Tests ---
    def test_marlin_method_instantiation(self):
        config = MarlinConfig(group_size=128, lm_head_quantized=False)
        method = config.get_quant_method(LinearBase(128, 256), "test") # N=256 for Marlin
        self.assertIsInstance(method, SglangMarlinLinearMethod)

    @unittest.skipUnless(VLLM_KERNELS_AVAILABLE, "vLLM kernels not available for Marlin apply test")
    def test_marlin_method_apply_with_kernels(self):
        self._run_apply_test(MarlinConfig, SglangMarlinLinearMethod, self._populate_marlin_params,
                             group_size=128, lm_head_quantized=False)

    def test_marlin_method_apply_raises_if_kernels_unavailable(self):
        self._run_apply_raises_test(MarlinConfig, SglangMarlinLinearMethod, self._populate_marlin_params,
                                   group_size=128, lm_head_quantized=False)

    # --- GPTQ-Marlin Tests ---
    def test_gptq_marlin_method_instantiation(self):
        # Need full_config for GPTQMarlinConfig.from_config
        base_config_dict = {"bits": 4, "group_size": 128, "desc_act": False, "sym": True, "lm_head": False}
        config = GPTQMarlinConfig.from_config(base_config_dict)
        method = config.get_quant_method(LinearBase(128, 256), "test")
        self.assertIsInstance(method, SglangGPTQMarlinLinearMethod)

    @unittest.skipUnless(VLLM_KERNELS_AVAILABLE, "vLLM kernels not available for GPTQ-Marlin apply test")
    def test_gptq_marlin_method_apply_with_kernels(self):
        base_config_dict = {"bits": 4, "group_size": 128, "desc_act": False, "sym": True, "lm_head": False, "dynamic": {}}
        # from_config is used by sglang, so test with it.
        # Directly instantiating GPTQMarlinConfig needs all args.
        # config_kwargs for _setup_linear_layer should match from_config's needs or direct init.
        # Let's use direct init here for clarity on what config_kwargs _setup_linear_layer gets.
        self._run_apply_test(GPTQMarlinConfig, SglangGPTQMarlinLinearMethod, self._populate_gptq_marlin_params,
                             weight_bits=4, group_size=128, desc_act=False, is_sym=True,
                             lm_head_quantized=False, dynamic={}, full_config=base_config_dict)

    def test_gptq_marlin_method_apply_raises_if_kernels_unavailable(self):
        base_config_dict = {"bits": 4, "group_size": 128, "desc_act": False, "sym": True, "lm_head": False, "dynamic": {}}
        self._run_apply_raises_test(GPTQMarlinConfig, SglangGPTQMarlinLinearMethod, self._populate_gptq_marlin_params,
                                   weight_bits=4, group_size=128, desc_act=False, is_sym=True,
                                   lm_head_quantized=False, dynamic={}, full_config=base_config_dict)

if __name__ == "__main__":
    unittest.main()

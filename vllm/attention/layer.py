# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer."""
import copy
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import vllm.envs as envs
from vllm.attention import AttentionType
from vllm.attention.selector import backend_name_to_enum, get_attn_backend
from vllm.config import CacheConfig, get_current_vllm_config
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group,
                                          is_v1_kv_transfer_group)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.platforms import _Backend, current_platform
from vllm.utils import direct_register_custom_op
from vllm.v1.attention.backends.utils import validate_kv_sharing_target


class Attention(nn.Module):
    """Attention layer.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:

    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Return the output tensor.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        per_layer_sliding_window: Optional[int] = None,
        use_mla: bool = False,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
        **extra_impl_args,
    ) -> None:
        """
        The KV cache is stored inside this class and is accessed via
        `self.kv_cache`.
        """
        super().__init__()
        if per_layer_sliding_window is not None:
            # per-layer sliding window
            sliding_window = per_layer_sliding_window
        elif cache_config is not None:
            # model-level sliding window
            sliding_window = cache_config.sliding_window
        else:
            sliding_window = None

        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            block_size = cache_config.block_size
            is_attention_free = cache_config.is_attention_free
            calculate_kv_scales = cache_config.calculate_kv_scales
        else:
            kv_cache_dtype = "auto"
            block_size = 16
            is_attention_free = False
            calculate_kv_scales = False
        if num_kv_heads is None:
            num_kv_heads = num_heads

        # The default k/v_scale is set to 1.0. This is ignored
        # when kv-cache is not fp8, and should be used with
        # kv-cache in fp8_e5m2. For kv-cache in fp8_e4m3, we
        # expect the pre-quantized k/v_scale to be loaded along
        # with the model weights.
        self.kv_cache_dtype = kv_cache_dtype
        self.calculate_kv_scales = calculate_kv_scales
        self._k_scale = torch.tensor(1.0, dtype=torch.float32)
        self._v_scale = torch.tensor(1.0, dtype=torch.float32)
        # FlashAttn doesn't support quantizing the kv-cache only
        # but requires q to be quantized as well.
        self._q_scale = torch.tensor(1.0, dtype=torch.float32)
        self._prob_scale = torch.tensor(1.0, dtype=torch.float32)

        # We also keep the float32 versions of k/v_scale for attention
        # backends that don't support tensors (Flashinfer)
        self._k_scale_float = 1.0
        self._v_scale_float = 1.0

        self.use_mla = use_mla
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window

        quant_method = quant_config.get_quant_method(
            self, prefix=prefix) if quant_config else None
        if quant_method is not None and not isinstance(
                quant_method, UnquantizedLinearMethod):
            assert isinstance(quant_method, BaseKVCacheMethod)
            # TODO (mgoin): kv cache dtype should be specified in the FP8
            # checkpoint config and become the "auto" behavior
            if self.kv_cache_dtype == "fp8_e5m2":
                raise ValueError("fp8_e5m2 kv-cache is not supported with "
                                 "fp8 checkpoints.")
            # If quantization is enabled, we make "k_scale" and "v_scale"
            # parameters so that it can be loaded from the model checkpoint.
            # The k/v_scale will then be converted back to native float32
            # values after weight loading.
            self.quant_method = quant_method
            self.quant_method.create_weights(self)

        # During model initialization, the default dtype is set as the model
        # weight and activation dtype.
        dtype = torch.get_default_dtype()
        attn_backend = get_attn_backend(head_size,
                                        dtype,
                                        kv_cache_dtype,
                                        block_size,
                                        is_attention_free,
                                        blocksparse_params is not None,
                                        use_mla=use_mla)
        impl_cls = attn_backend.get_impl_cls()
        self.impl = impl_cls(num_heads, head_size, scale, num_kv_heads,
                             alibi_slopes, sliding_window, kv_cache_dtype,
                             blocksparse_params, logits_soft_cap, attn_type,
                             kv_sharing_target_layer_name, **extra_impl_args)
        self.backend = backend_name_to_enum(attn_backend.get_name())
        self.dtype = dtype

        # For cuda-alike (CUDA and ROCM) and cpu platforms, we control how
        # torch.compile works by registering the attention as one giant
        # opaque custom op. For other platforms, we directly call them
        # and let torch.compile handle them.
        self.use_direct_call = not current_platform.is_cuda_alike(
        ) and not current_platform.is_cpu()

        self.use_output = attn_backend.accept_output_buffer
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.layer_name = prefix
        self.attn_type = attn_type

        if kv_sharing_target_layer_name is not None:
            if not envs.VLLM_USE_V1:
                raise NotImplementedError(
                    "Cross-layer KV sharing is not supported in V0.")

            validate_kv_sharing_target(
                prefix,
                kv_sharing_target_layer_name,
                compilation_config.static_forward_context,
            )
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        # use a placeholder kv cache tensor during init, which will be replaced
        # by bind_kv_cache
        # this variable will not be accessed if use_direct_call is True
        self.kv_cache = [
            torch.tensor([]) for _ in range(get_current_vllm_config(
            ).parallel_config.pipeline_parallel_size)
        ]

        self.q_range = torch.tensor(envs.Q_SCALE_CONSTANT, dtype=torch.float32)
        self.k_range = torch.tensor(envs.K_SCALE_CONSTANT, dtype=torch.float32)
        self.v_range = torch.tensor(envs.V_SCALE_CONSTANT, dtype=torch.float32)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        # For some alternate attention backends like MLA the attention output
        # shape does not match the query shape, so we optionally let the model
        # definition specify the output tensor shape.
        output_shape: Optional[torch.Size] = None,
    ) -> torch.Tensor:
        """
        The KV cache is stored inside this class and is accessed via
        `self.kv_cache`.

        Attention metadata (`attn_metadata`) is set using a context manager in
        the model runner's `execute_model` method. It is accessed via forward
        context using
        `vllm.forward_context.get_forward_context().attn_metadata`.
        """
        if self.calculate_kv_scales:
            attn_metadata = get_forward_context().attn_metadata
            if attn_metadata.enable_kv_scales_calculation:
                self.calc_kv_scales(query, key, value)
        if self.use_output:
            output_shape = (output_shape
                            if output_shape is not None else query.shape)
            output = torch.empty(output_shape,
                                 dtype=query.dtype,
                                 device=query.device)
            hidden_size = output_shape[-1]
            # We skip reshaping query, key and value tensors for the MLA
            # backend since these tensors have different semantics and are
            # processed differently.
            if not self.use_mla:
                # Reshape the query, key, and value tensors.
                # NOTE(woosuk): We do this outside the custom op to minimize the
                # CPU overheads from the non-CUDA-graph regions.
                query = query.view(-1, self.num_heads, self.head_size)
                output = output.view(-1, self.num_heads, self.head_size)
                if key is not None:
                    key = key.view(-1, self.num_kv_heads, self.head_size)
                if value is not None:
                    value = value.view(-1, self.num_kv_heads, self.head_size)
            if self.use_direct_call:
                forward_context: ForwardContext = get_forward_context()
                attn_metadata = forward_context.attn_metadata
                if isinstance(attn_metadata, dict):
                    attn_metadata = attn_metadata[self.layer_name]
                self_kv_cache = self.kv_cache[forward_context.virtual_engine]
                self.impl.forward(self,
                                  query,
                                  key,
                                  value,
                                  self_kv_cache,
                                  attn_metadata,
                                  output=output)
            else:
                torch.ops.vllm.unified_attention_with_output(
                    query, key, value, output, self.layer_name)
            return output.view(-1, hidden_size)
        else:
            if self.use_direct_call:
                forward_context = get_forward_context()
                attn_metadata = forward_context.attn_metadata
                if isinstance(attn_metadata, dict):
                    attn_metadata = attn_metadata[self.layer_name]
                self_kv_cache = self.kv_cache[forward_context.virtual_engine]
                return self.impl.forward(self, query, key, value,
                                         self_kv_cache, attn_metadata)
            else:
                return torch.ops.vllm.unified_attention(
                    query, key, value, self.layer_name)

    def calc_kv_scales(self, query, key, value):
        self._q_scale.copy_(torch.abs(query).max() / self.q_range)
        self._k_scale.copy_(torch.abs(key).max() / self.k_range)
        self._v_scale.copy_(torch.abs(value).max() / self.v_range)
        self._k_scale_float = self._k_scale.item()
        self._v_scale_float = self._v_scale.item()
        # We only calculate the scales once
        self.calculate_kv_scales = False

    def extra_repr(self) -> str:
        s = f"head_size={self.impl.head_size}"  # type: ignore
        s += f", num_heads={self.impl.num_heads}"  # type: ignore
        s += f", num_kv_heads={self.impl.num_kv_heads}"  # type: ignore
        s += f", scale={self.impl.scale}"  # type: ignore
        s += f", backend={self.impl.__class__.__name__}"
        return s

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        if hasattr(self.impl, "process_weights_after_loading"):
            self.impl.process_weights_after_loading(act_dtype)


class MultiHeadAttention(nn.Module):
    """Multi-headed attention without any cache, used for ViT."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        dtype = torch.get_default_dtype()
        attn_backend = get_attn_backend(head_size,
                                        dtype,
                                        kv_cache_dtype=None,
                                        block_size=16,
                                        is_attention_free=False)
        backend = backend_name_to_enum(attn_backend.get_name())
        if backend in {_Backend.FLASH_ATTN, _Backend.FLASH_ATTN_VLLM_V1}:
            backend = _Backend.XFORMERS

        self.attn_backend = backend if backend in {
            _Backend.TORCH_SDPA, _Backend.XFORMERS, _Backend.PALLAS_VLLM_V1
        } else _Backend.TORCH_SDPA

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Input shape: batch_size x seq_len x hidden_size"""
        # TODO(Isotr0py): Use existing backend implementations and support FA3
        bsz, q_len, _ = query.size()
        kv_len = key.size(1)

        query = query.view(bsz, q_len, self.num_heads, self.head_size)
        key = key.view(bsz, kv_len, self.num_kv_heads, self.head_size)
        value = value.view(bsz, kv_len, self.num_kv_heads, self.head_size)

        if (num_repeat := self.num_queries_per_kv) > 1:
            # Handle MQA and GQA
            key = torch.repeat_interleave(key, num_repeat, dim=2)
            value = torch.repeat_interleave(value, num_repeat, dim=2)

        if self.attn_backend == _Backend.XFORMERS:
            from xformers import ops as xops

            out = xops.memory_efficient_attention_forward(query,
                                                          key,
                                                          value,
                                                          scale=self.scale)
        elif self.attn_backend == _Backend.TORCH_SDPA:
            query, key, value = (x.transpose(1, 2)
                                 for x in (query, key, value))
            out = F.scaled_dot_product_attention(query,
                                                 key,
                                                 value,
                                                 scale=self.scale)
            out = out.transpose(1, 2)
        elif self.attn_backend == _Backend.PALLAS_VLLM_V1:
            query, key, value = (x.transpose(1, 2)
                                 for x in (query, key, value))
            from torch_xla.experimental.custom_kernel import flash_attention
            out = flash_attention(query, key, value, sm_scale=self.scale)
            out = out.transpose(1, 2)

        return out.reshape(bsz, q_len, -1)


def wait_for_kv_layer_from_connector(layer_name: str):
    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return

    connector = get_kv_transfer_group()

    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None:
        return
    assert isinstance(attn_metadata, dict)
    connector.wait_for_layer_load(layer_name)


def maybe_save_kv_layer_to_connector(
    layer_name: str,
    kv_cache_layer: List[torch.Tensor],
):
    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return

    connector = get_kv_transfer_group()

    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None:
        return
    assert isinstance(attn_metadata, dict)
    connector.save_kv_layer(layer_name, kv_cache_layer,
                            attn_metadata[layer_name])


def unified_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    wait_for_kv_layer_from_connector(layer_name)

    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[layer_name]
    self = forward_context.no_compile_layers[layer_name]
    kv_cache = self.kv_cache[forward_context.virtual_engine]
    output = self.impl.forward(self, query, key, value, kv_cache,
                               attn_metadata)

    maybe_save_kv_layer_to_connector(layer_name, kv_cache)
    return output


def unified_attention_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(query).contiguous()


direct_register_custom_op(
    op_name="unified_attention",
    op_func=unified_attention,
    mutates_args=[],
    fake_impl=unified_attention_fake,
    dispatch_key=current_platform.dispatch_key,
)


def unified_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    wait_for_kv_layer_from_connector(layer_name)
    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[layer_name]
    self = forward_context.no_compile_layers[layer_name]
    kv_cache = self.kv_cache[forward_context.virtual_engine]

    # If there's no metadata, just forward everything
    if attn_metadata is None:
        self.impl.forward(self,
                          query,
                          key,
                          value,
                          kv_cache,
                          None,
                          output=output)
        maybe_save_kv_layer_to_connector(layer_name, kv_cache)
        return

    # How many slices?
    qsl = attn_metadata.query_start_loc  # e.g. [0,256,512,768,1024]
    num_slices = qsl.numel() - 1
    if num_slices <= 1:
        self.impl.forward(self,
                          query,
                          key,
                          value,
                          kv_cache,
                          attn_metadata,
                          output=output)
        maybe_save_kv_layer_to_connector(layer_name, kv_cache)
        return

    # Multi‐query/GQA: loop each slice (sequence)
    for i in range(num_slices):
        # print("Processing slice", i, "of", num_slices)
        start = int(qsl[i].item())
        end = int(qsl[i + 1].item())
        length = end - start  # Length of the current query slice tokens

        # Slice Q/K/V and the output buffer for the current sequence tokens
        # Note: k_i, v_i might contain only the *new* tokens to added to cache,
        # or they might not be used directly if K/V read from
        # cache via block_table.
        # The shape [length, num_heads|num_kv_heads, head_dim] is important.
        q_i = query[start:end]
        k_i = key[start:
                  end]  # Slicing K/V this way assumes K/V only current tokens
        v_i = value[start:end]
        out_i = output[start:end]  # View for the output of this slice

        # --- Create metadata 'md' for the single sequence slice 'i' ---
        md = copy.copy(attn_metadata)  # Start with a shallow copy

        # 1. Information about the Query slice (q_i)
        md.num_actual_tokens = length  # Total tokens call (just q_i)
        md.num_input_tokens = length  # Number of tokens being processed now
        md.max_query_len = length  # Max query length *single-sequence batch*
        md.query_start_loc = torch.tensor(
            [0, length],  # Cumulative lengths for q_i (batch_size=1)
            dtype=qsl.dtype,
            device=qsl.device)

        # 2. Information about the Key/Value context (KV Cache via
        # PagedAttention)
        # The kernel needs the actual length of sequence 'i' currently
        # stored in the cache
        # This information should be in the original metadata's seq_lens
        # for sequence 'i'
        if attn_metadata.seq_lens is None:
            # This should not happen if using paged attention context, error or
            raise ValueError("md_orig.seq_lens is None, cannot determine "
                             "cache sequence length")
        # Get the total length of sequence 'i' currently in the cache
        seq_len_in_cache_i = int(attn_metadata.seq_lens[i].item())

        # Set seq_lens for the single-sequence batch context. This tells
        # the kernel the history length for sequence 'i'.
        md.seq_lens = torch.tensor([seq_len_in_cache_i],
                                   dtype=attn_metadata.seq_lens.dtype,
                                   device=attn_metadata.seq_lens.device)

        # Set the max sequence length for K for this single sequence context.
        # This should be the length of sequence 'i' in the cache.
        # md_orig.max_seq_len might be the max across the original batch,
        # setting it specifically for sequence 'i' seems more accurate here.
        md.max_seq_len = seq_len_in_cache_i  # Max length for K/V this sequence

        # Slice the block_table to provide only the blocks for sequence 'i'
        if attn_metadata.block_table is not None:
            md.block_table = attn_metadata.block_table[
                i:i + 1, :]  # Shape becomes [1, num_blocks]
        else:
            # If not using block_table (e.g., pure prefill with cu_seqlens_k),
            # ensure md.block_table is None or handle accordingly.
            md.block_table = None
            # If NOT using block_table, you might need cu_seqlens_k
            # corresponding to k_i and v_i, which would be [0, length]
            # if k_i/v_i are sliced like q_i.
            # md.cu_seqlens_k = md.query_start_loc # If need and no block_table

        # Slice the slot_mapping. This maps the *current* query tokens (length)
        # to their slots in the KV cache (for writing new K/V).
        # This slicing seems correct.
        md.slot_mapping = attn_metadata.slot_mapping[start:
                                                     end]  # Shape [length]

        # 3. Slice other per-sequence metadata if present and used by the kernel
        #    (Ensure these expect a shape starting with batch_size=1)
        if getattr(attn_metadata, "prefix_kv_lens", None) is not None:
            md.prefix_kv_lens = attn_metadata.prefix_kv_lens[i:i +
                                                             1]  # Shape [1]
        if getattr(attn_metadata, "suffix_kv_lens", None) is not None:
            md.suffix_kv_lens = attn_metadata.suffix_kv_lens[i:i +
                                                             1]  # Shape [1]
        if getattr(attn_metadata, "cu_prefix_query_lens", None) is not None:
            print(
                "Warning: cu_prefix_query_lens is not used in this context, ")
            pass  # Placeholder - needs careful implementation if used

        # --- Call the attention implementation ---
        # The kernel should now have:
        # - q_i: [length, num_heads, head_dim]
        # - k_i, v_i: [length, num_kv_heads, head_dim]
        # (representing *new* tokens, if any)
        # - kv_cache: The full cache object (kernel uses metadata to index)
        # - md: Metadata adjusted for a single sequence ('i') containing:
        #   - Query info: cu_seqlens_q=[0, length], max_query_len=length
        #   - KV Cache Context: block_table=[1, num_blocks],
        # seq_lens=[seq_len_in_cache_i], max_seq_len=seq_len_in_cache_i
        #   - Correct slot_mapping for q_i
        # GQA is handled implicitly by the kernel based on
        # head dims of q_i, k_i, v_i.

        self.impl.forward(self, q_i, k_i, v_i, kv_cache, md, output=out_i)

    maybe_save_kv_layer_to_connector(layer_name, kv_cache)


def unified_attention_with_output_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="unified_attention_with_output",
    op_func=unified_attention_with_output,
    mutates_args=["output"],
    fake_impl=unified_attention_with_output_fake,
    dispatch_key=current_platform.dispatch_key,
)

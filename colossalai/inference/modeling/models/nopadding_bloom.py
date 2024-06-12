from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import ProcessGroup
from transformers.models.bloom.modeling_bloom import BloomAttention, BloomBlock, BloomForCausalLM, BloomMLP, BloomModel

from colossalai.inference.config import InputMetaData, ModelShardInferenceConfig
from colossalai.inference.flash_decoding_utils import FDIntermTensors
from colossalai.inference.modeling.backends.attention_backend import AttentionMetaData, get_attention_backend
from colossalai.inference.modeling.backends.pre_attention_backend import get_pre_attention_backend
from colossalai.inference.utils import can_use_flash_attn2, get_alibi_slopes
from colossalai.logging import get_dist_logger
from colossalai.shardformer.layer.parallel_module import ParallelModule
from colossalai.tensor.d_tensor import Layout

logger = get_dist_logger(__name__)


def bloom_causal_lm_forward(
    self: BloomForCausalLM,
    input_tokens_ids: torch.Tensor,
    output_tensor: torch.Tensor,
    inputmetadata: InputMetaData,
    k_caches: List[torch.Tensor] = None,
    v_caches: List[torch.Tensor] = None,
) -> torch.Tensor:
    """Replacement of forward function in BloomForCausalLM.

    Args:
        input_tokens_ids (torch.Tensor): Input token Ids with no paddings.
        output_tensor (torch.Tensor): Intermediate tensor to hold attention output.
        inputmetadata (InputMetaData): Ths input metadata for a single step.
        k_caches (List[torch.Tensor], optional): List of key caches. Defaults to None.
        v_caches (List[torch.Tensor], optional): List of value caches. Defaults to None.
    """

    hidden_states = bloom_model_forward(
        self.transformer,
        input_tokens_ids=input_tokens_ids,
        output_tensor=output_tensor,
        inputmetadata=inputmetadata,
        k_caches=k_caches,
        v_caches=v_caches,
        use_cuda_kernel=inputmetadata.use_cuda_kernel,
        high_precision=inputmetadata.high_precision,
    )

    logits = self.lm_head(hidden_states)
    return logits


def bloom_model_forward(
    self: BloomModel,
    input_tokens_ids: torch.Tensor,
    output_tensor: torch.Tensor,
    inputmetadata: InputMetaData,
    k_caches: List[torch.Tensor] = None,
    v_caches: List[torch.Tensor] = None,
    use_cuda_kernel: Optional[bool] = True,
    high_precision: bool = False,
) -> torch.Tensor:
    """Replacement of forward function in BloomModel.

    Args:
        input_tokens_ids (torch.Tensor): Input token IDs with no padding.
        output_tensor (torch.Tensor): Intermediate tensor to hold attention output.
        inputmetadata (InputMetaData): Ths input metadata for a single step.
        k_caches (List[torch.Tensor], optional): List of k caches. Defaults to None.
        v_caches (List[torch.Tensor], optional): List of v caches. Defaults to None.
        use_cuda_kernel (Optional[bool], optional): Whether to use CUDA kernel. Defaults to True.
        high_precision (bool, optional): Whether to use high precision. Defaults to False.
    """
    block_tables = inputmetadata.block_tables
    sequence_lengths = inputmetadata.sequence_lengths
    batch_size = inputmetadata.batch_size
    kv_seq_len = inputmetadata.kv_seq_len

    # NOTE: current our cuda kernel not supports batch_size >= 32 and kv_seq_len > 512
    if batch_size >= 32 and kv_seq_len > 512:
        use_cuda_kernel = False

    cu_seqlens = None
    if use_cuda_kernel:
        if can_use_flash_attn2(inputmetadata.dtype):
            cu_seqlens = F.pad(torch.cumsum(sequence_lengths, dim=0, dtype=torch.int32), (1, 0))

    input_embeds = self.word_embeddings(input_tokens_ids)
    hidden_states = self.word_embeddings_layernorm(input_embeds)

    sm_scale = 1.0 / (inputmetadata.head_dim**0.5)
    norm_output = torch.empty_like(hidden_states)
    tokens_to_verify = inputmetadata.num_tokens_to_verify if inputmetadata.use_spec_dec else None
    residual = None

    for layer_id, layer in enumerate(self.h):
        # print(f"[DEBUG] layer id {layer_id} residual {residual}")
        # print(f"[DEBUG] number of layers {len(self.h)}")
        hidden_states = layer(
            hidden_states=hidden_states,
            residual=residual,
            block_tables=block_tables,
            k_cache=k_caches[layer_id],
            v_cache=v_caches[layer_id],
            is_prompts=inputmetadata.is_prompts,
            is_verifier=inputmetadata.use_spec_dec,
            tokens_to_verify=tokens_to_verify,
            sequence_lengths=sequence_lengths,
            fd_inter_tensor=inputmetadata.fd_inter_tensor,
            kv_seq_len=kv_seq_len,
            output_tensor=output_tensor,
            norm_output=norm_output,
            sm_scale=sm_scale,
            use_cuda_kernel=use_cuda_kernel,
            cu_seqlens=cu_seqlens,
            high_precision=high_precision,
        )
        # print(f"[DEBUG] kcaches {k_caches[layer_id]} vcaches {v_caches[layer_id]}")

    if inputmetadata.is_prompts:
        seq_len_cumsum = sequence_lengths.cumsum(dim=0)
        hidden_states = hidden_states[seq_len_cumsum - 1].contiguous()

    hidden_states = self.ln_f(hidden_states)
    return hidden_states


def bloom_block_forward(
    self: BloomBlock,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    block_tables: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    sequence_lengths: torch.Tensor,
    fd_inter_tensor: FDIntermTensors,
    is_prompts: bool = True,
    is_verifier: bool = False,
    tokens_to_verify: int = None,
    kv_seq_len: int = 0,
    output_tensor: torch.Tensor = None,
    norm_output: torch.Tensor = None,
    use_cuda_kernel: bool = True,
    sm_scale: int = None,
    cu_seqlens: torch.Tensor = None,
    high_precision: bool = False,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Replacement of forward function in the BloomBlock module.

    Args:
        hidden_states (torch.Tensor): input to the layer of shape [token_num, embed_dim].
        block_tables (torch.Tensor): A 2D tensor of shape [batch_size, max_blocks_per_sequence],
            storing mapping of token_position_id -> block_id.
        k_cache (torch.Tensor): It holds the GPU memory for the key cache.
        v_cache (torch.Tensor): It holds the GPU memory for the key cache.
        sequence_lengths (torch.Tensor): Holding the sequence length of each sequence.
        fd_inter_tensor (FDIntermTensors): Holding tensors used for
            storing intermediate values in flash-decoding.
        is_prompts (bool, optional): Whether the current inference process is in the context input phase. Defaults to True.
        kv_seq_len (int, optional): The max sequence length of input sequences. Defaults to 0.
        output_tensor (torch.Tensor, optional): The mid tensor holds the output of attention. Defaults to None.
        norm_output (torch.Tensor, optional): The mid tensor holds the output of layernorm. Defaults to None.
        sm_scale (int, optional): Used for flash attention. Defaults to None.
        use_cuda_kernel: (bool, optional): Whether to use cuda kernel. Defaults to True.
        cu_seqlens(torch.Tensor, optional): Holding the cumulative sum of sequence length.
        high_precision(Optional[bool]): Whether to use float32 for underlying calculations of float16 data to achieve higher precision, defaults to False.

    Returns:
        torch.Tensor: The output tensor.
    """
    # LayerNorm before attention
    norm_output = self.input_layernorm(hidden_states)
    # print(f"[DEBUG] check apply_residual_connection_post_layernorm {self.apply_residual_connection_post_layernorm}")
    # Found false from model config, so change the residual to norm_output

    if self.apply_residual_connection_post_layernorm:
        residual = norm_output
    else:
        # residual = hidden_states
        residual = norm_output

    # Self attention
    attention_output = self.self_attention(
        hidden_states=norm_output,
        block_tables=block_tables,
        k_cache=k_cache,
        v_cache=v_cache,
        is_prompts=is_prompts,
        is_verifier=is_verifier,
        tokens_to_verify=tokens_to_verify,
        sequence_lengths=sequence_lengths,
        fd_inter_tensor=fd_inter_tensor,
        kv_seq_len=kv_seq_len,
        output_tensor=output_tensor,
        sm_scale=sm_scale,
        cu_seqlens=cu_seqlens,
        high_precision=high_precision,
    )

    attention_output = attention_output + residual

    # LayerNorm post attention
    norm_output = self.post_attention_layernorm(attention_output)

    if self.apply_residual_connection_post_layernorm:
        residual = norm_output
    else:
        # residual = attention_output
        residual = norm_output

    # MLP (including residuals)
    output = self.mlp(norm_output, residual)
    return output


def bloom_gelu_forward(x: torch.Tensor) -> torch.Tensor:
    """
    Custom bias GELU function. Adapted from Megatron-DeepSpeed code. Here we use a simple implementation (inference) to
    make the model jitable.

    Args:
        x (`torch.tensor`, *required*):
            input hidden states
    """
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


class NopadBloomMLP(BloomMLP, ParallelModule):
    def __init__(
        self,
        dense_h_to_4h_w: torch.Tensor = None,
        dense_4h_to_h_w: torch.Tensor = None,
        process_group: ProcessGroup = None,
    ):
        ParallelModule.__init__(self)
        self.process_group = process_group
        # TODO: tp mlp

        # Native layers from BloomMLP
        self.dense_h_to_4h_w = dense_h_to_4h_w
        # self.gelu_impl = GeLUFunction.apply

        self.dense_4h_to_h_w = dense_4h_to_h_w

    @staticmethod
    def from_native_module(
        module: BloomMLP, process_group: Union[ProcessGroup, List[ProcessGroup]], *args, **kwargs
    ) -> ParallelModule:
        """
        Initialize the weight of NopadBloomMLP from original BloomMLP.

        Args:
            module (nn.Module): The original BloomMLP layer.

        Returns:
            NopadBloomMLP: The initialized NopadBloomMLP layer.
        """
        dense_h_to_4h_w = module.dense_h_to_4h.weight.transpose(0, 1)
        dense_4h_to_h_w = module.dense_4h_to_h.weight.transpose(0, 1)

        mlp_layer = NopadBloomMLP(
            dense_h_to_4h_w=dense_h_to_4h_w,
            dense_4h_to_h_w=dense_4h_to_h_w,
            process_group=process_group,
        )
        return mlp_layer

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Forward function of NopafBloomMLP.

        Args:
            hidden_states (torch.Tensor): The input tensor with shape [token_num, embed_dim].
            residual (torch.Tensor): The residual tensor with shape [token_num, embed_dim].

        Returns:
            torch.Tensor: The output tensor with shape [token_num, embed_dim].
        """
        # hidden_states = torch.mm(self.dense_h_to_4h_w, hidden_states)
        # bias = torch.zeros_like(hidden_states)
        # hidden_states = self.gelu_impl(hidden_states, bias)
        # intermediate_output = torch.mm(self.dense_4h_to_h_w, hidden_states)
        # bias = torch.zeros_like(intermediate_output)
        # output = bias_dropout_add_fused_inference(intermediate_output, bias, residual, 0.0)

        hidden_states = torch.mm(hidden_states, self.dense_h_to_4h_w)
        hidden_states = bloom_gelu_forward(hidden_states)
        hidden_states = torch.mm(hidden_states, self.dense_4h_to_h_w)

        return hidden_states + residual


class NopadBloomAttention(ParallelModule):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attn_qproj_w: torch.Tensor = None,
        attn_kproj_w: torch.Tensor = None,
        attn_vproj_w: torch.Tensor = None,
        attn_oproj: ParallelModule = None,
        # qkv_w: torch.Tensor = None,
        model_shard_infer_config: ModelShardInferenceConfig = None,
        process_group: ProcessGroup = None,
        helper_layout: Layout = None,
    ):
        """This layer replaces the BloomAttention.

        Args:
            hidden_size (int): Imensionality of the embeddings and hidden states.
            n_heads (int): Number of attention heads for each attention layer in the Transformer encoder.
            attn_qproj_w (torch.Tensor, optional): The transposed q_proj weight. Defaults to None.
            attn_kproj_w (torch.Tensor, optional): The transposed k_proj weight. Defaults to None.
            attn_vproj_w (torch.Tensor, optional): The transposed v_proj weight. Defaults to None.
            attn_oproj (torch.Tensor, optional): The transposed o_proj weight. Defaults to None.
        """
        ParallelModule.__init__(self)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        if self.head_dim not in [64, 128, 256]:
            model_shard_infer_config.use_cuda_kernel = False  # TODO: need set Falsh for op.flash_decoding_attention
        self.process_group = process_group

        # Alibi attention configs
        slopes_start = self.process_group.rank() * num_heads
        self.alibi_slopes = get_alibi_slopes(num_heads=num_heads, device=attn_qproj_w.device)[
            slopes_start : slopes_start + num_heads
        ].contiguous()
        self.alibi_slopes = nn.Parameter(self.alibi_slopes)

        # Inference configs
        self.helper_layout = helper_layout
        self.use_cuda_kernel = model_shard_infer_config.use_cuda_kernel

        self.process_group = process_group
        self.attn_backend = get_attention_backend(model_shard_infer_config)
        self.pre_attn_backend = get_pre_attention_backend(model_shard_infer_config)

        # qkv weights and projection layer
        self.o_proj = attn_oproj
        qkv_weight_list = [attn_qproj_w.transpose(0, 1), attn_kproj_w.transpose(0, 1), attn_vproj_w.transpose(0, 1)]
        self.qkv_weight = nn.Parameter(torch.stack(qkv_weight_list, dim=0))

        # self.qkv_w = qkv_w

    @staticmethod
    def from_native_module(
        module: BloomAttention, process_group: Union[ProcessGroup, List[ProcessGroup]], *args, **kwargs
    ) -> "NopadBloomAttention":
        """Interface for shardformer to initialize the NopadBloomAttention weight by
        origin BloomAttention layer.

        Args:
            module (BloomAttention): The original BloomAttention layer.
        """

        hidden_size = module.hidden_size
        num_heads = module.num_heads
        # TODO: the fused qkv layer got bias=True
        q_proj_w, k_proj_w, v_proj_w = module.query_key_value.weight.view((hidden_size, 3, -1)).transpose(0, 1)

        attn_qproj_w = q_proj_w
        attn_kproj_w = k_proj_w
        attn_vproj_w = v_proj_w
        attn_oproj = module.dense
        # qkv_w = module.query_key_value.weight.transpose(0, 1)

        model_shard_infer_config = kwargs.get("model_shard_infer_config", None)
        # helper_layout = (module.query_key_value.weight.dist_layout)

        attn_layer = NopadBloomAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            attn_qproj_w=attn_qproj_w,
            attn_kproj_w=attn_kproj_w,
            attn_vproj_w=attn_vproj_w,
            attn_oproj=attn_oproj,
            # qkv_w=qkv_w,
            model_shard_infer_config=model_shard_infer_config,
            process_group=process_group,
            # helper_layout=helper_layout,
        )
        return attn_layer

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        pass

    def _split_heads(self, fused_qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split the last dimension into (num_heads, head_dim) without making any copies, results share same memory
        storage as `fused_qkv`

        Args:
            fused_qkv (`torch.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]

        Returns:
            query: [batch_size, seq_length, num_heads, head_dim] key: [batch_size, seq_length, num_heads, head_dim]
            value: [batch_size, seq_length, num_heads, head_dim]
        """
        batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
        fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads, 3, self.head_dim)
        return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_tables: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        sequence_lengths: torch.Tensor,
        fd_inter_tensor: FDIntermTensors,
        is_prompts: bool = True,
        is_verifier: bool = False,
        tokens_to_verify: int = None,
        kv_seq_len: int = 0,
        output_tensor: torch.Tensor = None,
        sm_scale: int = None,
        cu_seqlens: torch.Tensor = None,
        high_precision: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward function of the NopadBloomAttention. Current attention does not support speculative decoding.

        Args:
            hidden_states (torch.Tensor): input to the layer of shape [token_num, embed_dim].
            block_tables (torch.Tensor): A 2D tensor of shape [batch_size, max_blocks_per_sequence],
                storing mapping of token_position_id -> block_id.
            k_cache (torch.Tensor): It holds the GPU memory for the key cache.
            v_cache (torch.Tensor): It holds the GPU memory for the key cache.
            sequence_lengths (torch.Tensor, optional): Holding the sequence length of each sequence.
            cos_sin (Tuple[torch.Tensor], optional): Holding cos and sin.
            fd_inter_tensor (FDIntermTensors, optional): Holding tensors used for
                storing intermediate values in flash-decoding.
            is_prompts (bool, optional): Whether the current inference process is in the context input phase. Defaults to True.
            kv_seq_len (int, optional): The max sequence length of input sequences. Defaults to 0.
            output_tensor (torch.Tensor, optional): The mid tensor holds the output of attention. Defaults to None.
            sm_scale (int, optional): Used for flash attention. Defaults to None.
            use_cuda_kernel: (bool, optional): Whether to use cuda kernel. Defaults to True.
            cu_seqlens(torch.Tensor, optional): Holding the cumulative sum of sequence length.
            high_precision(Optional[bool]): Whether to use float32 for underlying calculations of float16 data to achieve higher precision, defaults to False.
        """

        token_nums = hidden_states.size(0)
        hidden_states = hidden_states.expand(3, -1, -1)
        block_size = k_cache.size(-2)

        # fused_qkv = torch.mm(hidden_states, self.qkv_w)
        # (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)
        # batch_size, q_length, _, _ = query_layer.shape

        # query_states = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        # key_states = key_layer.permute(0, 2, 3, 1).reshape(batch_size * self.num_heads, self.head_dim, q_length)
        # value_states = value_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)

        query_states, key_states, value_states = (
            torch.bmm(hidden_states, self.qkv_weight).view(3, token_nums, self.num_heads, self.head_dim).unbind(0)
        )

        attn_metadata = AttentionMetaData(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            block_size=block_size,
            kv_seq_len=kv_seq_len,
            sequence_lengths=sequence_lengths,
            sm_scale=sm_scale,
            alibi_slopes=self.alibi_slopes,
            cu_seqlens=cu_seqlens,
            output_tensor=output_tensor,
            use_spec_dec=is_verifier,
            use_alibi_attn=True,
        )

        if is_prompts:
            self.pre_attn_backend.prefill(attn_metadata, high_precision=high_precision)
            attn_output = self.attn_backend.prefill(attn_metadata, token_nums=token_nums)
        else:
            q_len = tokens_to_verify + 1 if is_verifier else 1
            self.pre_attn_backend.decode(attn_metadata, q_len=q_len)
            attn_output = self.attn_backend.decode(attn_metadata, fd_inter_tensor=fd_inter_tensor, q_len=q_len)

        attn_output = attn_output.view(-1, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output

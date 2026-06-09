from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

import transformers
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5GatedDeltaNet,
    Qwen3_5ModelOutputWithPast,
    Qwen3_5PreTrainedModel,
    Qwen3_5TextModel,
    apply_mask_to_padding_states,
)
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as _stock_qwen3_5


def _fused_rmsnorm_forward(self, x):
    # Fused, bit-faithful replacement for the stock Qwen3_5RMSNorm.forward: same op order
    # (fp32 normalize, (1 + weight) multiply in fp32, downcast last) in ONE kernel instead
    # of ~5 + 3 dtype copies. RMSNorm runs 81x per forward pass (x2 under grad ckpt) and
    # accounted for ~46% of all aten::copy_ calls in the step profile.
    return F.rms_norm(
        x.float(), (x.shape[-1],), 1.0 + self.weight.float(), self.eps
    ).type_as(x)


_stock_qwen3_5.Qwen3_5RMSNorm.forward = _fused_rmsnorm_forward


class A2DQwen3_5TextConfig(Qwen3_5TextConfig):
    model_type = "a2d-qwen3_5"  # <- NEW model_type
    bidirectional_linear: bool = False  # <- NEW: enable bidirectional linear-attn


class A2DQwen3_5GatedDeltaNet(Qwen3_5GatedDeltaNet):
    """Bidirectional gated delta-net for masked diffusion (no KV/conv cache).

    Approach A: run the causal delta-rule scan left->right and right->left with
    SHARED weights, sum the outputs; use a non-causal (centered) depthwise conv.
    Zero new parameters; gated by ``config.bidirectional_linear`` so the disabled
    path exactly reproduces the stock causal layer.
    """

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.config = config  # stock __init__ does not retain config

    def _causal_conv(self, mixed_qkv):
        # Stock causal short conv: pad k-1 on the left, crop to T.
        # mixed_qkv: [b, conv_dim, T]
        T = mixed_qkv.shape[-1]
        return F.silu(self.conv1d(mixed_qkv)[..., :T])

    def _noncausal_conv(self, mixed_qkv):
        # Non-causal (centered) depthwise conv so each position sees both sides.
        # mixed_qkv: [b, conv_dim, T]
        T = mixed_qkv.shape[-1]
        k = self.conv_kernel_size
        out = F.conv1d(
            mixed_qkv,
            self.conv1d.weight,
            self.conv1d.bias,
            padding=k // 2,
            groups=self.conv_dim,
        )[..., :T]
        return F.silu(out)

    def _scan(self, query, key, value, g, beta):
        core, _ = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        return core  # [b, T, heads, head_v_dim]

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        bidirectional = getattr(self.config, "bidirectional_linear", False)

        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        b, seq_len, _ = hidden_states.shape

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)  # [b, conv_dim, T]
        z = self.in_proj_z(hidden_states).reshape(b, seq_len, -1, self.head_v_dim)
        beta = self.in_proj_b(hidden_states).sigmoid()  # [b, T, num_v]
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(
            self.in_proj_a(hidden_states).float() + self.dt_bias
        )  # [b, T, num_v]

        # Conv: centered (non-causal) when bidirectional, else stock causal so the
        # disabled path reproduces the stock layer exactly.
        if bidirectional:
            mixed_qkv = self._noncausal_conv(mixed_qkv)
        else:
            mixed_qkv = self._causal_conv(mixed_qkv)
        mixed_qkv = mixed_qkv.transpose(1, 2)  # [b, T, conv_dim]

        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(b, seq_len, -1, self.head_k_dim)
        key = key.reshape(b, seq_len, -1, self.head_k_dim)
        value = value.reshape(b, seq_len, -1, self.head_v_dim)
        if self.num_v_heads // self.num_k_heads > 1:
            rep = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)

        out_fwd = self._scan(query, key, value, g, beta)
        if bidirectional:
            def flip(t):
                return torch.flip(t, dims=[1])
            out_bwd = self._scan(flip(query), flip(key), flip(value), flip(g), flip(beta))
            core_attn_out = out_fwd + torch.flip(out_bwd, dims=[1])
        else:
            core_attn_out = out_fwd

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(b, seq_len, -1)
        return self.out_proj(core_attn_out)


class A2DQwen3_5TextModel(Qwen3_5TextModel):
    config_class = A2DQwen3_5TextConfig
    _supports_flex_attn = True

    def __init__(self, config):
        super().__init__(config)
        if getattr(config, "bidirectional_linear", False):
            for i, layer in enumerate(self.layers):
                if config.layer_types[i] == "linear_attention":
                    layer.linear_attn = A2DQwen3_5GatedDeltaNet(config, layer_idx=i)
            self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Qwen3_5ModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Cache support is needed for BD3LM's KV-cached block decoding (sampler).
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        # mRoPE position ids (text only): replicate the stock 4-way expand.
        if position_ids is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(
                past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device
            )
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        # ---- Mask handling for full-attention vs linear (delta-net) layers ----
        # 2D tensor / None  -> padding-only bidirectional mask for full-attn (MDLM-style).
        # 4D tensor         -> BD3LM block-causal mask for sdpa (passed by trainer/sampler).
        # BlockMask object  -> BD3LM block-causal mask for flex_attention (block-sparse, fast).
        # In the block-mask cases the recurrent delta-net can't consume the mask, so it runs
        # causally with no padding mask (BD3LM conditioning reaches the noised half via full-attn).
        if attention_mask is not None and not (
            isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2
        ):
            full_attn_mask = attention_mask
            linear_attn_mask = None
        else:
            if attention_mask is None:
                attention_mask = torch.ones(
                    inputs_embeds.shape[:2], device=inputs_embeds.device, dtype=torch.long
                )
            full_attn_mask = _prepare_4d_attention_mask(attention_mask, self.dtype)
            # linear-attention layers: 2D padding mask only; their bidirectionality (when
            # config.bidirectional_linear) comes from the dual-scan in A2DQwen3_5GatedDeltaNet.
            linear_attn_mask = self._update_linear_attn_mask(attention_mask, past_key_values)
        # -----------------------------------------------------------------------

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_mask = (
                linear_attn_mask
                if self.config.layer_types[i] == "linear_attention"
                else full_attn_mask
            )
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return Qwen3_5ModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class A2DQwen3_5LMHeadModel(Qwen3_5ForCausalLM):
    config_class = A2DQwen3_5TextConfig
    config: A2DQwen3_5TextConfig
    # qwen3_5 doesn't declare flex support, but the attention interface resolves it and BD3LM's
    # block-causal mask is far cheaper as a block-sparse flex_attention than a dense sdpa mask.
    _supports_flex_attn = True

    def __init__(self, config):
        Qwen3_5PreTrainedModel.__init__(self, config)
        self.model = A2DQwen3_5TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # belt-and-suspenders: ensure full-attn modules never self-select causal
        for module in self.modules():
            if hasattr(module, "is_causal"):
                module.is_causal = False
        self.post_init()


transformers.AutoConfig.register("a2d-qwen3_5", A2DQwen3_5TextConfig)
transformers.AutoModel.register(A2DQwen3_5TextConfig, A2DQwen3_5LMHeadModel)
transformers.AutoModelForMaskedLM.register(A2DQwen3_5TextConfig, A2DQwen3_5LMHeadModel)

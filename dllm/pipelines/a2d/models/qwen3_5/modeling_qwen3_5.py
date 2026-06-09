from typing import Optional

import torch
from torch import nn

import transformers
from transformers.cache_utils import Cache
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ModelOutputWithPast,
    Qwen3_5PreTrainedModel,
    Qwen3_5TextModel,
)
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig


class A2DQwen3_5TextConfig(Qwen3_5TextConfig):
    model_type = "a2d-qwen3_5"  # <- NEW model_type


class A2DQwen3_5TextModel(Qwen3_5TextModel):
    config_class = A2DQwen3_5TextConfig

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

        # mRoPE position ids (text only): replicate the stock 4-way expand.
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        # ---- NEW CODE: bidirectional, padding-only mask for FULL-ATTENTION layers ----
        if attention_mask is None:
            attention_mask = torch.ones(
                inputs_embeds.shape[:2], device=inputs_embeds.device, dtype=torch.long
            )
        full_attn_mask = _prepare_4d_attention_mask(attention_mask, self.dtype)
        # linear-attention layers: keep stock 2D padding-mask handling (causal in M1)
        linear_attn_mask = self._update_linear_attn_mask(attention_mask, past_key_values)
        # -----------------------------------------------------------------------------

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
                past_key_values=None,        # diffusion: no incremental cache
                use_cache=False,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return Qwen3_5ModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=None)


class A2DQwen3_5LMHeadModel(Qwen3_5ForCausalLM):
    config_class = A2DQwen3_5TextConfig
    config: A2DQwen3_5TextConfig

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

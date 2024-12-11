import torch
from torch import nn
from transformers import MistralConfig
from transformers.models.mistral.modeling_mistral import MistralPreTrainedModel, MistralDecoderLayer, MistralRMSNorm, MistralForCausalLM, MistralModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging
from transformers.cache_utils import Cache, DynamicCache
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union, List


class CustomModelOutput:
    def __init__(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union['Cache', List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_legacy_cache: Optional[bool] = None,
        causal_mask: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        all_hidden_states: Optional[List[torch.Tensor]] = None,
        all_self_attns: Optional[List[torch.Tensor]] = None,
        next_decoder_cache: Optional[Union['Cache', List[torch.FloatTensor]]] = None,
    ):
        # Initialize inputs
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.position_ids = position_ids
        self.past_key_values = past_key_values
        self.inputs_embeds = inputs_embeds
        self.labels = labels
        self.use_cache = use_cache
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.return_dict = return_dict
        self.cache_position = cache_position
        
        # Initialize outputs
        self.return_legacy_cache = return_legacy_cache
        self.causal_mask = causal_mask
        self.hidden_states = hidden_states
        self.all_hidden_states = all_hidden_states
        self.all_self_attns = all_self_attns
        self.next_decoder_cache = next_decoder_cache
        

class CustomModelOutputWithPast:
    def __init__(
        self,
        base_model_output: Union[Tuple, BaseModelOutputWithPast],
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ):
        self.base_model_output = base_model_output
        self.labels = labels
        self.return_dict = return_dict


class MistralModel_custom(MistralModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MistralDecoderLayer`]

    Args:
        config: MistralConfig
    """

    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MistralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward_embed_tokens(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> CustomModelOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            # logger.warning_once(
            #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            # )
            use_cache = False

        # 1. embedding
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # 2. retrieve kv cache (shape: [num_layers, ...])
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            return_legacy_cache = True

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, use_cache, output_attentions
        )

        hidden_states = inputs_embeds
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        
        return CustomModelOutput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            
            return_legacy_cache=return_legacy_cache,
            causal_mask=causal_mask,
            hidden_states=hidden_states,
            all_hidden_states=all_hidden_states,
            all_self_attns=all_self_attns,
            next_decoder_cache=next_decoder_cache,
        )
        
    
    def forward_layers(
        self,
        idx: int, # custom argument - layer index
        previous_outputs: CustomModelOutput, # custom argument - previous outputs
        past_key_values_layer: Optional[Union[Cache, List[torch.FloatTensor]]] = None, # custom argument - past key values
    ) -> CustomModelOutput:
        # extract previous outputs
        input_ids = previous_outputs.input_ids
        attention_mask = previous_outputs.attention_mask
        position_ids = previous_outputs.position_ids
        past_key_values = previous_outputs.past_key_values
        # update past key values
        if idx != 0:
            key_states, value_states = past_key_values_layer[0] # shape: [1, ...]
            past_key_values.update(key_states, value_states, idx)
        inputs_embeds = previous_outputs.inputs_embeds
        labels = previous_outputs.labels
        use_cache = previous_outputs.use_cache
        output_attentions = previous_outputs.output_attentions
        output_hidden_states = previous_outputs.output_hidden_states
        return_dict = previous_outputs.return_dict
        cache_position = previous_outputs.cache_position
        
        return_legacy_cache = previous_outputs.return_legacy_cache
        causal_mask = previous_outputs.causal_mask
        hidden_states = previous_outputs.hidden_states
        all_hidden_states = previous_outputs.all_hidden_states
        all_self_attns = previous_outputs.all_self_attns
        next_decoder_cache = previous_outputs.next_decoder_cache
        
        # decoder layers
        # originally, the loop is for layer in self.layers
        decoder_layer = self.layers[idx]
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)
            
        return CustomModelOutput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            
            return_legacy_cache=return_legacy_cache,
            causal_mask=causal_mask,
            hidden_states=hidden_states,
            all_hidden_states=all_hidden_states,
            all_self_attns=all_self_attns,
            next_decoder_cache=next_decoder_cache,
        )
            

    def forward_norm(
        self,
        previous_outputs: CustomModelOutput, # custom argument - previous outputs
    ) -> CustomModelOutputWithPast:
        # load previous outputs
        input_ids = previous_outputs.input_ids
        attention_mask = previous_outputs.attention_mask
        position_ids = previous_outputs.position_ids
        past_key_values = previous_outputs.past_key_values
        inputs_embeds = previous_outputs.inputs_embeds
        labels = previous_outputs.labels
        use_cache = previous_outputs.use_cache
        output_attentions = previous_outputs.output_attentions
        output_hidden_states = previous_outputs.output_hidden_states
        return_dict = previous_outputs.return_dict
        cache_position = previous_outputs.cache_position
        
        return_legacy_cache = previous_outputs.return_legacy_cache
        causal_mask = previous_outputs.causal_mask
        hidden_states = previous_outputs.hidden_states
        all_hidden_states = previous_outputs.all_hidden_states
        all_self_attns = previous_outputs.all_self_attns
        next_decoder_cache = previous_outputs.next_decoder_cache
        
        # do normalization
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return CustomModelOutputWithPast(
            base_model_output=BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            ),
            labels=labels,
            return_dict=return_dict,
        )
        

class MistralForCausalLM_custom(MistralForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MistralModel_custom(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward_1(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> CustomModelOutput:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 1. embedding
        outputs = self.model.forward_embed_tokens(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels, # added for forward_4
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        
        return outputs
        
    def forward_2(
        self,
        idx: int, # custom argument - layer index
        previous_outputs: CustomModelOutput, # custom argument - previous outputs
        past_key_values_layer: Optional[Union[Cache, List[torch.FloatTensor]]] = None, # custom argument - past key values
    ) -> CustomModelOutput:
        
        # 2. decoder layers
        outputs = self.model.forward_layers(
            idx=idx,
            previous_outputs=previous_outputs,
            past_key_values_layer=past_key_values_layer,
        )
        
        return outputs
    
    def forward_3(
        self,
        previous_outputs: CustomModelOutput, # custom argument - previous outputs
    ) -> CustomModelOutputWithPast:

        # 3. normalization
        outputs = self.model.forward_norm(
            previous_outputs=previous_outputs,
        )
        
        return outputs
    
    def forward_4(
        self,
        previous_outputs: CustomModelOutputWithPast, # custom argument - previous outputs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # extract previous outputs
        outputs = previous_outputs.base_model_output
        labels = previous_outputs.labels
        return_dict = previous_outputs.return_dict
        
        # 4. logits

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Ensure tensors are on the same device
            shift_labels = shift_labels.to(shift_logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

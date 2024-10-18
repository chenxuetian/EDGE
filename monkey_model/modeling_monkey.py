from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging
from .modeling_qwen import QWenModel, QWenLMHeadModel


SUPPORT_CUDA = torch.cuda.is_available()
SUPPORT_BF16 = SUPPORT_CUDA and torch.cuda.is_bf16_supported()
SUPPORT_FP16 = SUPPORT_CUDA and torch.cuda.get_device_capability(0)[0] >= 7
logger = logging.get_logger(__name__)
class MonkeyModel(QWenModel):
    def __init__(self, config):
        super().__init__(config)
    
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
    ):
        if past_key_values is None:
            bs, n_patchs, _, _, _ = images.shape  # (bs, 5, C, H, W)
            feats = self.visual(images.flatten(0, 1)).unflatten(0, sizes=(bs, n_patchs))  # (bs, 5, seq_len, d_hidden)
            images = feats.flatten(1, 2) # (bs, 5*seq_len, d_hidden)
        else:
            images = None
        return super().forward(input_ids,
            past_key_values,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,inputs_embeds,
            encoder_hidden_states,
            encoder_attention_mask,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
            images)
    


class MonkeyLMHeadModel(QWenLMHeadModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.rotary_emb\.inv_freq"]
    _keys_to_ignore_on_load_unexpected = [r"h\.\d+\.attn\.masked_bias"]

    def __init__(self, config):
        super().__init__(config)
        assert (
            config.bf16 + config.fp16 + config.fp32 <= 1
        ), "Only one of \"bf16\", \"fp16\", \"fp32\" can be true"

        autoset_precision = config.bf16 + config.fp16 + config.fp32 == 0

        if autoset_precision:
            if SUPPORT_BF16:
                logger.warn(
                    "The model is automatically converting to bf16 for faster inference. "
                    "If you want to disable the automatic precision, please manually add bf16/fp16/fp32=True to \"AutoModelForCausalLM.from_pretrained\"."
                )
                config.bf16 = True
            elif SUPPORT_FP16:
                logger.warn(
                    "The model is automatically converting to fp16 for faster inference. "
                    "If you want to disable the automatic precision, please manually add bf16/fp16/fp32=True to \"AutoModelForCausalLM.from_pretrained\"."
                )
                config.fp16 = True
            else:
                config.fp32 = True

        if config.bf16 and SUPPORT_CUDA and not SUPPORT_BF16:
            logger.warn("Your device does NOT seem to support bf16, you can switch to fp16 or fp32 by by passing fp16/fp32=True in \"AutoModelForCausalLM.from_pretrained\".")
        if config.fp16 and SUPPORT_CUDA and not SUPPORT_FP16:
            logger.warn("Your device does NOT support faster inference with fp16, please switch to fp32 which is likely to be faster")
        if config.fp32:
            if SUPPORT_BF16:
                logger.warn("Your device support faster inference by passing bf16=True in \"AutoModelForCausalLM.from_pretrained\".")
            elif SUPPORT_FP16:
                logger.warn("Your device support faster inference by passing fp16=True in \"AutoModelForCausalLM.from_pretrained\".")

        self.transformer = MonkeyModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.bf16:
            self.transformer.bfloat16()
            self.lm_head.bfloat16()
        if config.fp16:
            self.transformer.half()
            self.lm_head.half()
        
        # self.post_init()
        
    def _reset_parameters(self):
        self.linkin._reset_parameters()
        self.det_neck._reset_parameters()
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
            
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            images=images,
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # shift_logits = lm_logits[..., 1282:-1, :].contiguous()
            # shift_labels = labels[..., 1283:].contiguous()
            # loss = F.cross_entropy(
            #     shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            # )
            lm_logits = lm_logits[..., :-1, :]
            labels = labels[..., 1:]
            lm_logits = lm_logits[labels != -100]
            labels = labels[labels != -100]
            loss = F.cross_entropy(
                lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)
            )
        
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

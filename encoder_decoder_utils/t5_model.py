# From: https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py
import typing
import warnings
from typing import Optional, Tuple
from dataclasses import dataclass

import torch
import transformers
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import ModelOutput
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import (
    T5ForConditionalGeneration,
)
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput


@dataclass
class EncoderOutput(ModelOutput):
    hidden_states: torch.FloatTensor = None
    attention_mask: torch.FloatTensor = None


@dataclass
class HierarchicalSeq2SeqLMOutput(Seq2SeqLMOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        sequence_losses (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*, returned when
            `labels` is provided): Language modeling loss (for each token in the batch).
        encoder_outputs (:obj:`EncoderOutput`, *optional*, returned when `encoder_outputs` is provided): The outputs of
            the encoder.

    """
    sequence_losses: Optional[torch.FloatTensor] = None
    encoder_outputs: Optional[EncoderOutput] = None


class DepthForConditionalGeneration(T5ForConditionalGeneration):

    @classmethod
    def _from_config(cls, config, **kwargs):
        super()._from_config(config, **kwargs)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            target_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            cross_attention_mask: Optional[torch.BoolTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            encoder_outputs=None,
            is_shuffled: Optional[torch.BoolTensor] = False,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> typing.Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        """
        Perform the forward pass of T5 model for generation.

        :param input_ids: A tensor of shape (batch_size, input_sequence_length) with the token ids of the input
            sequence.
        :param encoder_attention_mask: A tensor of shape (batch_size, input_sequence_length, input_sequence_length)
            with the attention mask from the encoder to the encoder (0 for mask, 1 for no mask).
        :param target_ids: A tensor of shape (batch_size, target_sequence_length, target_sequence_length) with
            the token ids of the decoder input sequence.
        :param decoder_attention_mask: A tensor of shape (batch_size, target_sequence_length, input_sequence_length)
            with the attention mask from the decoder to the decoder (0 for mask, 1 for no mask).
        :param cross_attention_mask: A tensor of shape (batch_size, target_sequence_length, input_sequence_length) with
            the attention mask from the decoder to the encoder (0 for mask, 1 for no mask).
        :param labels: A tensor of shape (batch_size, target_sequence_length) with the token ids of the labels
            sequence.
        :param encoder_outputs: A BaseModelOutput with the outputs of the encoder.
        :param is_shuffled: A tensor of shape (batch_size) with a boolean mask indicating whether the input is shuffled
            or not.
        :param head_mask: A tensor of shape (num_layers, num_heads) with boolean mask for the encoder self-attention
            modules.
        :param decoder_head_mask: A tensor of shape (num_layers, num_heads) with boolean mask for the decoder
            self-attention modules.
        :param cross_attn_head_mask: A tensor of shape (num_layers, num_heads) with boolean mask for the decoder
            cross-attention modules.
        :param past_key_values: A tuple of length num_layers with the cached past key values for the decoder.
        :param inputs_embeds: A tensor of shape (batch_size, input_sequence_length, hidden_size) with the embeddings of
            the input sequence.
        :param decoder_inputs_embeds: A tensor of shape (batch_size, target_sequence_length, hidden_size) with the
            embeddings of the decoder input sequence.
        :param use_cache: A boolean indicating whether to use the cached keys and values.
        :param output_attentions: A boolean indicating whether to return the attentions weights.
        :param output_hidden_states: A boolean indicating whether to return the hidden states.
        :param return_dict: A boolean indicating whether to return a ModelOutput instead of a plain tuple. True
            indicates that a ModelOutput should be returned. False indicates that a tuple should be returned.

        :return: A Seq2SeqLMOutput containing the outputs of the forward pass (i.e., the logits, loss, etc...)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(transformers.models.t5.modeling_t5.__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=encoder_attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and target_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            target_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if target_ids is not None:
                target_ids = target_ids.to(self.decoder.first_device)
            if cross_attention_mask is not None:
                cross_attention_mask = cross_attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=target_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=cross_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)  # (bsz, seq_length, vocab_size)

        sequence_losses = None
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            sequence_losses = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)).reshape(labels.shape)

            is_padding = labels.eq(-100)
            loss = sequence_losses[~is_padding].mean()

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return HierarchicalSeq2SeqLMOutput(
            loss=loss,
            sequence_losses=sequence_losses,
            logits=lm_logits,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_attentions=encoder_outputs.attentions,
        )

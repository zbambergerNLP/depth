import numpy as np
from torch import inf
import torch
import unittest
from typing import Optional, Tuple, List, Any, Dict

from torch.nn import Module
from transformers import T5Tokenizer, PreTrainedTokenizer, PreTrainedModel
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from absl.testing import parameterized
from transformers import T5Tokenizer, T5Model

import corruption
from t5_model import DepthForConditionalGeneration
import tokenizer_utils as discourse_tokenizer


def setUp(return_baseline: bool = False) -> tuple[PreTrainedTokenizer, T5Model]:
    # Set up the model and tokenizer. This is done once for all tests.
    base_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    base_model = T5ForConditionalGeneration.from_pretrained("t5-small")

    # tokenizer, _ = discourse_tokenizer.create_discourse_ul2_tokenizer(model_name="t5-small")
    tokenizer = discourse_tokenizer.DepthTokenizer.from_pretrained("t5-small")
    model = DepthForConditionalGeneration.from_pretrained("t5-small")
    model.resize_token_embeddings(len(tokenizer))

    if return_baseline:
        return base_tokenizer, base_model

    return tokenizer, model


tokenizer, model = setUp()


class TestCustomT5Model(parameterized.TestCase):
    """
    Tests for the custom T5 model.

    The tests are parameterized, so that we can test multiple inputs at once.

    Our custom T5 model is a hierarchical encoder-decoder model. This heirarchy is created by adding special tokens
    to the input and target sequences. Namely, we add <eosen> tokens to the end of each sentence, and <sent_i> tokens
    to the beginning of each sentence, where i is the index of the sentence in the sequence. We also add <extra_id_i>
    tokens as part of the traditional span-masking objective (i.e. the objective used in the original T5 paper).
    """

    @parameterized.named_parameters(
        {
            "testcase_name": "TestSingleSentenceNoPadding",
            "input_ids": torch.tensor(
                [tokenizer.encode(
                    ["<eosen>", "<sent_17>", "hello", "<extra_id_82>", "<eosen>"],
                    add_special_tokens=False)
                ]),
            "input_token_type_ids": torch.tensor([[1, 1, 1, 1, 1]]),
            "target_ids": torch.tensor(
                [tokenizer.encode(
                    ["<eosen>", "<sent_17>", "<extra_id_82>", "world", "<eosen>"],
                    add_special_tokens=False)
                ]),
            "target_token_type_ids": torch.tensor([[1, 1, 1, 1, 1]]),
            "expected_encoder_self_attention_mask": torch.tensor([[[1, 1, 1, 1, 1],
                                                                   [1, 1, 1, 1, 1],  # sentence token in the input
                                                                   [1, 1, 1, 1, 1],
                                                                   [1, 1, 1, 1, 1],
                                                                   [1, 1, 1, 1, 1]]]),
            "expected_decoder_self_attention_mask": torch.tensor([[[1, 0, 0, 0, 0],
                                                                   [0, 1, 0, 0, 0],  # sentence token in target
                                                                   [1, 1, 1, 0, 0],
                                                                   [1, 1, 1, 1, 0],
                                                                   [1, 1, 1, 1, 1]]]),
            "expected_cross_attention_mask": torch.tensor([[[1, 1, 1, 1, 1],
                                                            [0, 1, 0, 0, 0],  # sentence token in target
                                                            [1, 1, 1, 1, 1],
                                                            [1, 1, 1, 1, 1],
                                                            [1, 1, 1, 1, 1]]]),
        },
        {
            "testcase_name": "TestMultipleSentencesNoPadding",
            "input_ids": torch.tensor(
                [tokenizer.encode(
                    ["<eosen>", "<sent_17>", "hello", "<extra_id_82>", "<eosen>", "<sent_9>", "<extra_id_13>",
                     "are", "you?", "<eosen>"],
                    add_special_tokens=False)
                ]),
            "input_token_type_ids": torch.tensor([[1, 1, 1, 1, 1, 2, 2, 2, 2, 2]]),
            "target_ids": torch.tensor(
                [tokenizer.encode(
                    ["<eosen>", "<sent_17>", "<extra_id_82>", "world", "<eosen>", "<sent_9>", "<extra_id_13>",
                     "how", "<eosen>"],
                    add_special_tokens=False)
                ]),
            "target_token_type_ids": torch.tensor([[1, 1, 1, 1, 1, 2, 2, 2, 2]]),
            # 10 x 10 attention mask corresponding to the 10 tokens in the input sequence
            "expected_encoder_self_attention_mask": torch.tensor([[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                                   [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                                   [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                                                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]]),
            # 9 x 9 attention mask corresponding to the 9 tokens in the target sequence
            "expected_decoder_self_attention_mask": torch.tensor([[[1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                   [0, 1, 0, 0, 0, 0, 0, 0, 0],  # sentence token #1
                                                                   [1, 1, 1, 0, 0, 0, 0, 0, 0],
                                                                   [1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                                   [1, 1, 1, 1, 1, 0, 0, 0, 0],
                                                                   [0, 1, 0, 0, 0, 1, 0, 0, 0],  # sentence token #2
                                                                   [1, 1, 1, 1, 1, 1, 1, 0, 0],
                                                                   [1, 1, 1, 1, 1, 1, 1, 1, 0],
                                                                   [1, 1, 1, 1, 1, 1, 1, 1, 1]]]),
            # 9 x 10 attention mask corresponding to the 9 tokens in the target sequence attending to 10 tokens in the
            # input sequence
            "expected_cross_attention_mask": torch.tensor([[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # sentence token #1
                                                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # sentence token #2
                                                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]]),

        },
        {
            "testcase_name": "TestSingleSentenceWithPadding",
            "input_ids": torch.tensor(
                [tokenizer.encode(
                    ["<eosen>", "<sent_17>", "hello", "<extra_id_82>", "<eosen>"],
                    add_special_tokens=False) + [tokenizer.pad_token_id] * 5]
            ),
            "input_token_type_ids": torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]]),
            "target_ids": torch.tensor(
                [tokenizer.encode(
                    ["<eosen>", "<sent_17>", "<extra_id_82>", "world", "<eosen>"],
                    add_special_tokens=False) + [tokenizer.pad_token_id] * 4]
            ),
            "target_token_type_ids": torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0]]),
            # 10 x 10 attention mask corresponding to the 10 tokens in the input sequence
            "expected_encoder_self_attention_mask": torch.tensor([[[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                                   [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                                   [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                                   [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                                   [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                                   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                                                   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                                                   [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                                                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]]),
            # 9 x 9 attention mask corresponding to the 9 (of which 4 are padding) tokens in the target sequence
            "expected_decoder_self_attention_mask": torch.tensor([[[1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                   [0, 1, 0, 0, 0, 0, 0, 0, 0],
                                                                   [1, 1, 1, 0, 0, 0, 0, 0, 0],
                                                                   [1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                                   [1, 1, 1, 1, 1, 0, 0, 0, 0],
                                                                   [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                                                   [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                                                   [0, 0, 0, 0, 0, 0, 0, 1, 0],
                                                                   [0, 0, 0, 0, 0, 0, 0, 0, 1]]]),
            # 9 x 10 attention mask corresponding to the 9 tokens in the target sequence attending to 10 tokens in the
            # input sequence
            "expected_cross_attention_mask": torch.tensor([[[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # padding token in decoder
                                                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # padding token in decoder
                                                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # padding token in decoder
                                                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # padding token in decoder
                                                            ]]),
        },
        {
            "testcase_name": "TestMultipleSentencesWithPadding",
            "input_ids": torch.tensor(
                [tokenizer.encode(
                    ["<eosen>", "<sent_17>", "hello", "<extra_id_82>", "<eosen>", "<sent_9>", "<extra_id_13>",
                     "are", "you?", "<eosen>"],
                    add_special_tokens=False) + [tokenizer.pad_token_id] * 5]),
            "input_token_type_ids": torch.tensor([[1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0]]),
            "target_ids": torch.tensor(
                [tokenizer.encode(
                    ["<eosen>", "<sent_17>", "<extra_id_82>", "world", "<eosen>", "<sent_9>", "<extra_id_13>",
                     "hello", "<eosen>"],
                    add_special_tokens=False) + [tokenizer.pad_token_id] * 4]
            ),
            "target_token_type_ids": torch.tensor([[1, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0]]),
            # 15 x 15 attention mask corresponding to the 15 tokens in the input sequence (including padding) attending
            # to 15 tokens in the input sequence (including padding). Padding tokens can attend only themselves.
            "expected_encoder_self_attention_mask": torch.tensor([[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                                   [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                                   [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]]),
            # 13 x 15 attention mask corresponding to 13 tokens in the target sequence attending to 15 tokens in the
            # input sequence.
            # NOTE: Within cross attention, padding tokens from the decoder to all tokens in the encoder.
            # We are able to do this since loss will not be computed over padding tokens anyway.
            "expected_cross_attention_mask": torch.tensor(
                [[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # padding token in decoder
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # padding token in decoder
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # padding token in decoder
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # padding token in decoder
                  ]]),
            # 13 x 13 attention mask corresponding to 13 tokens in the target sequence attending to 13 tokens in the
            # target sequence
            "expected_decoder_self_attention_mask": torch.tensor([[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                   [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                   [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                   [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                   [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                   [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                                                   [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                                                   [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                                                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]]),
        },
    )
    def test_forward(
            self,
            input_ids,
            input_token_type_ids,
            target_ids,
            target_token_type_ids,
            expected_encoder_self_attention_mask,
            expected_cross_attention_mask,
            expected_decoder_self_attention_mask,
            seed=42,
    ):
        """

        :param input_ids: A batch of input_ids. A tensor of shape (batch_size, input_sequence_length)
        :param input_token_type_ids: A tensor of shape (batch_size, input_sequence_length) where the value at
            position (i, j) is the sentence index of the jth token in the ith input sequence.
        :param target_ids: A batch of target_ids. A tensor of shape (batch_size, target_sequence_length)
        :param target_token_type_ids: A tensor of shape (batch_size, target_sequence_length) where the value at
            position (i, j) is the sentence index of the jth token in the ith target sequence.
        :param expected_encoder_self_attention_mask: A tensor of shape
            (batch_size, input_sequence_length, input_sequence_length) where the value at position (i, j, k) is 1 if
            the jth token in the ith input sequence attends to the kth token in the ith input sequence and 0 otherwise.
        :param expected_cross_attention_mask: A tensor of shape
            (batch_size, target_sequence_length, input_sequence_length) where the value at position (i, j, k) is 1 if
            the jth token in the ith target sequence attends to the kth token in the ith input sequence and 0 otherwise.
        :param expected_decoder_self_attention_mask: A tensor of shape
            (batch_size, target_sequence_length, target_sequence_length) where the value at position (i, j, k) is 1 if
            the jth token in the ith target sequence attends to the kth token in the ith target sequence and 0 otherwise.
        """
        # Test the forward method of the model.
        batch_encoder_self_attention_mask, batch_cross_attention_mask, batch_decoder_self_attention_mask = (
            corruption.create_attention_mask(
                input_ids=input_ids,
                target_ids=target_ids,
                input_token_type_ids=input_token_type_ids,
                target_token_type_ids=target_token_type_ids,
                tokenizer=tokenizer,
            )
        )
        decoder_input_ids = torch.tensor(
            corruption.shift_tokens_right(
                target_ids,
                pad_token_id=tokenizer.pad_token_id,
                decoder_start_token_id=tokenizer.convert_tokens_to_ids(discourse_tokenizer.END_OF_SENTENCE_TOKEN),
            ),
            dtype=torch.long,
        )
        outputs = model.forward(
            encoder_input_ids=input_ids,
            encoder_self_attention_mask=batch_encoder_self_attention_mask,
            encoder_cross_attention_mask=batch_cross_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_self_attention_mask=batch_decoder_self_attention_mask,
            labels=target_ids,
            output_attentions=True,
        )

        # Attention outputs consist of tuples of length num_layers where each element is a tensor of shape
        # (batch_size, num_heads, input_sequence_length, input_sequence_length)
        encoder_attentions = outputs.encoder_attentions
        cross_attentions = outputs.cross_attentions
        decoder_attentions = outputs.decoder_attentions

        for layer_index in range(len(encoder_attentions)):
            for head_index in range(encoder_attentions[layer_index].shape[1]):
                is_masked_encoder_self_attention = torch.where(
                    condition=(
                        (encoder_attentions[layer_index][:, head_index, :, :] == 0)
                    ),
                    input=torch.zeros(encoder_attentions[layer_index][:, head_index, :, :].shape),
                    other=torch.ones(encoder_attentions[layer_index][:, head_index, :, :].shape)
                )
                self.assertTrue(torch.all(is_masked_encoder_self_attention == expected_encoder_self_attention_mask))

                is_masked_decoder_self_attention = torch.where(
                    condition=(
                        (decoder_attentions[layer_index][:, head_index, :, :] == 0)
                    ),
                    input=torch.zeros(decoder_attentions[layer_index][:, head_index, :, :].shape),
                    other=torch.ones(decoder_attentions[layer_index][:, head_index, :, :].shape)
                )
                self.assertTrue(torch.all(is_masked_decoder_self_attention == expected_decoder_self_attention_mask))

                is_masked_cross_attention = torch.where(
                    condition=(
                        (cross_attentions[layer_index][:, head_index, :, :] == 0)
                    ),
                    input=torch.zeros(cross_attentions[layer_index][:, head_index, :, :].shape),
                    other=torch.ones(cross_attentions[layer_index][:, head_index, :, :].shape)
                )
                self.assertTrue(torch.all(is_masked_cross_attention == expected_cross_attention_mask))


if __name__ == '__main__':
    unittest.main()

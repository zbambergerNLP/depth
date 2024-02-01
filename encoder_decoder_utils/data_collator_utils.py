import typing
import transformers
import torch
import numpy as np
from dataclasses import dataclass
import datasets

from encoder_decoder_utils.corruption import (
    corrupt_for_vanilla_t5,
    shuffle_inputs,
    create_attention_mask,
    shift_tokens_right,
    create_model_input_for_corrupted_batch,
    create_sentinel_ids,
    random_spans_noise_mask,
)

from encoder_decoder_utils.constants import (
    DEPTHTokenizerConstants,
)


@dataclass
class T5DataCollator:
    """
    Data collator used for T5 span-masked language modeling.

    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed
    length.

    For more information on how T5 span-masked language modeling works, one can take a look
    """

    def __init__(
            self,
            tokenizer: transformers.PreTrainedTokenizer,
            noise_density: float,
            mean_noise_span_length: float,
            input_length: int,
            target_length: int,
            pad_token_id: int,
            decoder_start_token_id: int,
            seed: int = 42,
    ):
        """Initialize a T5DataCollator instance.

        :param tokenizer: The tokenizer to use as part of span corruption in the data collator.
        :param noise_density: The density of noise to be added to the input sequence.
        :param mean_noise_span_length: The mean length of the noise spans.
        :param input_length: The length of the input sequence.
        :param target_length: The length of the target sequence.
        :param pad_token_id: The id of the pad token.
        :param decoder_start_token_id: The id of the decoder start token.
        :param seed: The seed to use for random number generation.
        """
        np.random.seed(seed)
        self.tokenizer = tokenizer
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.input_length = input_length
        self.target_length = target_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(
            self,
            examples: datasets.Dataset,
    ) -> typing.Dict[str, np.ndarray]:
        """Generate a dictionary of input tensors to a Vanilla T5 language model.

        :param examples: A list of examples, each of which is a dictionary of the form
            {
                "content": str,
                "input_ids": List[int],
                "token_type_ids": List[int],
                "num_truncated_tokens": int,
                "special_tokens_mask": List[int],
                "length": int,
            }
        :return: A BatchEncoding instance containing the following fields:
            - "input_ids": The input ids of the model.
            - "encoder_self_attention_mask": The self attention mask of the encoder.
            - "encoder_token_type_ids": The token type ids of the encoder.
            - "cross_attention_mask": The cross attention mask of the decoder.
            - "decoder_input_ids": The input ids of the decoder.
            - "decoder_self_attention_mask": The self attention mask of the decoder.
            - "labels": The labels for the decoder.
        """
        batch_encoding = {k: np.array([examples[i][k] for i in range(len(examples))]) for k, v in examples[0].items()}
        batch = corrupt_for_vanilla_t5(
            batch_encoding,
            self.tokenizer.vocab_size,
            self.input_length,
            self.target_length,
            self.pad_token_id,
            self.tokenizer.eos_token_id,
            self.decoder_start_token_id,
            self.noise_density,
        )
        batch = {k: torch.from_numpy(v) for k, v in batch.items()}
        return batch


@dataclass
class DEPTHDataCollator:
    """
    Data collator used for T5 span-masked language modeling as well as SLM sentence un-shuffling.

    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed
    length.

    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .

    Likewise, for information on sentence un-shuffling and the required tokenization, and attention mask modifications,
    refer to the official paper: <https://arxiv.org/pdf/2010.16249.pdf>.

    :param tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
        The tokenizer used for encoding the data.
    :param noise_density (:obj:`float`): The probability with which to (randomly) mask tokens in the input.
    :param mean_noise_span_length (:obj:`float`): The average span length of the masked tokens.
    :param input_length (:obj:`int`): The expected input length after masking.
    :param target_length (:obj:`int`): The expected target length after masking.
    :param pad_token_id: (:obj:`int`): The pad token id of the model
    :param decoder_start_token_id: (:obj:`int): The decoder start token id of the model
    :param sentence_shuffling_probability: (:obj:`float`): The probability with which to shuffle the sentences.
    """

    def __init__(
            self,
            tokenizer: transformers.PreTrainedTokenizerBase,
            noise_density: float,
            mean_noise_span_length: float,
            input_length: int,
            target_length: int,
            pad_token_id: int,
            decoder_start_token_id,
            seed: int = 42,
            sentence_shuffling_probability: float = 0.0,
    ):
        """Initializes the data collator.

        Apply corruption in the form of span masking and sentence shuffling to the input data.

        :param tokenizer: The tokenizer used for encoding the textual data within examples.
        :param noise_density: The desired ratio of masked tokens to total tokens in the input. Note that this is just
            a target; if the exact value cannot be achieved, the actual noise density will be close to this value. This
            value must be in the range [0, 1].
        :param mean_noise_span_length: The average length of a span of masked tokens. This value must be positive.
        :param input_length: The desired length of the input sequence (including special tokens). This value must be
            positive.
        :param target_length: The desired length of the target sequence (including special tokens). This value must be
            positive.
        :param pad_token_id: The id of the pad token for the model.
        :param decoder_start_token_id: The id of the decoder start token for the model.
        :param seed: A seed for the random number generator.
        :param sentence_shuffling_probability: The probability with which to shuffle the sentences in the input.
        """
        np.random.seed(seed)
        self.tokenizer = tokenizer
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.input_length = input_length
        self.target_length = target_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.sentence_shuffling_probability = sentence_shuffling_probability

    def __call__(
            self,
            examples: datasets.Dataset,
    ) -> transformers.BatchEncoding:
        """Generate a dictionary of input tensors to a Discourse T5 language model.

        :param examples: A list of examples, each of which is a dictionary of the form
            {
                "content": str,
                "input_ids": List[int],
                "token_type_ids": List[int],
                "num_truncated_tokens": int,
                "special_tokens_mask": List[int],
                "length": int,
            }
        :return: A BatchEncoding instance containing the following fields:
            - "encoder_input_ids": The input ids of the model.
            - "encoder_self_attention_mask": The self attention mask of the encoder.
            - "encoder_token_type_ids": The token type ids of the encoder.
            - "cross_attention_mask": The cross attention mask of the decoder.
            - "decoder_input_ids": The input ids of the decoder.
            - "decoder_self_attention_mask": The self attention mask of the decoder.
            - "labels": The labels for the decoder.
        """
        batch = transformers.BatchEncoding(
            {k: np.array([examples[i][k] for i in range(len(examples))]) for k, v in examples[0].items()}
        )

        input_ids = np.array(batch.pop(DEPTHTokenizerConstants.INPUT_IDS))
        token_type_ids = np.array(batch.pop(DEPTHTokenizerConstants.TOKEN_TYPE_IDS))
        batch_size, padded_sequence_length = input_ids.shape
        sequence_lengths = np.sum(np.not_equal(token_type_ids, 0).astype(np.int32), axis=1)
        span_mask = np.reshape(
            np.concatenate(
                [
                    random_spans_noise_mask(
                        sequence_length=example_sequence_length,
                        maximum_length=padded_sequence_length,
                        noise_density=self.noise_density,
                        mean_noise_span_length=self.mean_noise_span_length)
                    for example_sequence_length in sequence_lengths]
            ),
            newshape=[batch_size, padded_sequence_length],
        )

        # Shift the span mask by two in order to account for the initial end of sentence and start of sentence tokens.
        span_mask = torch.concat(
            [torch.zeros([batch_size, 2], dtype=torch.bool),
             torch.tensor(span_mask[:, :-2], dtype=torch.bool)],
            dim=1,
        )

        # Identify special tokens.
        special_tokens = self.tokenizer.all_special_ids
        sentence_tokens = list(
            filter(lambda token: f'<{DEPTHTokenizerConstants.SENT}' in token, self.tokenizer.all_special_tokens))
        sentence_tokens.append(DEPTHTokenizerConstants.END_OF_SENTENCE_TOKEN)
        sentence_token_ids = self.tokenizer.convert_tokens_to_ids(sentence_tokens)

        # Ensure mask is only applied to non-special tokens.
        augmented_input_span_mask = np.where(np.isin(input_ids, special_tokens, invert=True), span_mask, False)

        # Create a sentinel mask, where 0s indicate a lack of mask, positive values indicate the start of a masked span,
        #  and -1 indicates the continuation of a masked span.
        input_ids_sentinel = create_sentinel_ids(self.tokenizer, augmented_input_span_mask.astype(np.int8))

        (modified_encoder_input_ids,
         modified_encoder_token_type_ids,
         modified_label_ids,
         modified_label_token_type_ids) = (
            create_model_input_for_corrupted_batch(
                input_ids=input_ids,
                input_ids_sentinel=input_ids_sentinel,
                token_type_ids=token_type_ids,
                batch_size=batch_size,
                sequence_lengths=sequence_lengths,
                padded_sequence_length=padded_sequence_length,
                sentence_token_ids=sentence_token_ids,
            )
        )

        modified_label_ids = np.array(modified_label_ids)

        # T5 assumes that labels which correspond with pad tokens are replaced with -100. These tokens are ignored
        #  when computing the loss.
        # modified_label_ids = torch.tensor(modified_label_ids, dtype=torch.long)
        modified_label_ids[modified_label_ids == self.pad_token_id] = -100
        modified_decoder_input_ids = shift_tokens_right(
            modified_label_ids,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
        )
        # Prepend a 1 to the token type ids to account for the initial token of the decoder.
        # modified_decoder_token_type_ids = modified_label_ids.new_zeros(modified_label_ids.shape)
        modified_decoder_token_type_ids = np.zeros(modified_label_ids.shape, dtype=np.int64)
        modified_decoder_token_type_ids[..., 1:] = modified_label_ids[..., :-1].copy()
        modified_decoder_token_type_ids[..., 0] = 1

        # As in the original SLM paper, we conditionally shuffle every example in the batch:
        # https://arxiv.org/pdf/2010.16249.pdf
        # See page 3, under the "Sequence Shuffling" in section 2.
        shuffle_batch = np.random.uniform() < self.sentence_shuffling_probability

        if shuffle_batch:

            batch_encoder_input_ids = []
            batch_encoder_token_type_ids = []

            for example_index in range(batch_size):
                # Identify the unique sentence ids, the number of tokens in each sentence, and the start index of each
                #  sentence.
                (example_encoder_sentence_ids,
                 example_encoder_sentence_start_indices,
                 example_encoder_sentence_lengths) = np.unique(
                    modified_encoder_token_type_ids[example_index][1:],
                    return_counts=True,
                    return_index=True)

                # Shuffle the sentences.
                (example_encoder_shuffled_sentence_order,
                 example_encoder_shuffled_sentence_lengths,
                 example_encoder_shuffled_sentence_start_indices,
                 example_encoder_shuffled_token_type_ids) = (
                    shuffle_inputs(
                        sentence_unique_ids=example_encoder_sentence_ids,
                        sentence_start_indices=example_encoder_sentence_start_indices,
                        sentence_lengths=example_encoder_sentence_lengths
                    )
                )

                # Concatenate the shuffled sentences.
                example_encoder_shuffled_end_indices = (
                        example_encoder_shuffled_sentence_start_indices + example_encoder_shuffled_sentence_lengths)
                example_encoder_shuffled_input_ids = np.concatenate(
                    [modified_encoder_input_ids[example_index][start_index:end_index] for start_index, end_index in zip(
                        example_encoder_shuffled_sentence_start_indices, example_encoder_shuffled_end_indices)
                     ]
                )

                # Prepend the start of sentence token.
                # TODO: Get the [EOSEN] token id from the tokenizer.
                batch_encoder_input_ids.append(
                    np.concatenate([[32120], example_encoder_shuffled_input_ids]))
                batch_encoder_token_type_ids.append(
                    np.concatenate(
                        [
                            [example_encoder_shuffled_token_type_ids[0]],
                            example_encoder_shuffled_token_type_ids,
                        ]
                    )
                )
            modified_encoder_input_ids = np.stack(batch_encoder_input_ids)
            modified_encoder_token_type_ids = np.stack(batch_encoder_token_type_ids)
        else:
            modified_label_token_type_ids = np.array(modified_label_token_type_ids)
            modified_encoder_token_type_ids = np.array(modified_encoder_token_type_ids)
            modified_encoder_input_ids = np.array(modified_encoder_input_ids)
            modified_label_ids = np.array(modified_label_ids)

        batch_encoder_self_attention_mask, batch_cross_attention_mask, batch_decoder_self_attention_mask = (
            create_attention_mask(
                input_ids=modified_encoder_input_ids,
                target_ids=modified_decoder_input_ids,
                input_token_type_ids=modified_encoder_token_type_ids,
                target_token_type_ids=modified_decoder_token_type_ids,
                tokenizer=self.tokenizer,
            )
        )

        # TODO: Create static variables for the string keys (model inputs) below.
        batch["input_ids"] = torch.tensor(modified_encoder_input_ids)
        batch["encoder_attention_mask"] = torch.tensor(batch_encoder_self_attention_mask)
        batch["cross_attention_mask"] = torch.tensor(batch_cross_attention_mask)
        batch["target_ids"] = torch.tensor(modified_decoder_input_ids)
        batch["decoder_attention_mask"] = torch.tensor(batch_decoder_self_attention_mask)
        batch["labels"] = torch.tensor(modified_label_ids)
        # All examples in the batch will have the same value for the "is_shuffled" key.
        batch["is_shuffled"] = torch.tensor([shuffle_batch] * batch_size)
        batch["is_shuffled"].resize(batch_size, 1)

        # TODO: Save the below variables so that they can be logged in wandb.

        if 'num_truncated_tokens' in batch:
            batch.__delitem__('num_truncated_tokens')
        if 'special_tokens_mask' in batch:
            batch.__delitem__('special_tokens_mask')
        # if 'input_length' in batch:
        #     batch.__delitem__('input_length')
        if 'attention_mask' in batch:
            batch.__delitem__('attention_mask')

        return batch

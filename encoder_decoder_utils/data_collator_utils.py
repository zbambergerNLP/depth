import typing
import transformers
import torch
import numpy as np
from dataclasses import dataclass
import datasets
from transformers import TrainingArguments, TrainerState, TrainerControl

from encoder_decoder_utils import constants

from encoder_decoder_utils import tokenizer_utils
from encoder_decoder_utils.corruption import (
    corrupt_for_vanilla_t5,
    corrupt_for_depth,
    create_depth_encoder_self_attention_mask,
    shift_tokens_right
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
            tokenizer: tokenizer_utils.DepthTokenizer,
            noise_density: float,
            mean_noise_span_length: float,
            input_length: int,
            target_length: int,
            pad_token_id: int,
            decoder_start_token_id,
            seed: int = 42,
            sentence_shuffling_probability: float = 0.0,
            pmi: bool = False,
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
        self.pmi = pmi

    def __call__(
            self,
            examples: typing.Union[
                typing.Dict[str, np.ndarray],
                typing.List[typing.Dict[str, np.ndarray]],
                transformers.BatchEncoding,
            ],
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
        shuffle_probability = self.sentence_shuffling_probability
        do_shuffle = np.random.uniform(0, 1) < shuffle_probability

        batch = corrupt_for_depth(
            examples=examples,
            tokenizer=self.tokenizer,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
            noise_density=self.noise_density,
            mean_noise_span_length=self.mean_noise_span_length,
            input_length=self.input_length,
            target_length=self.target_length,
            pmi=self.pmi,
            do_shuffle=do_shuffle,
        )
        return transformers.BatchEncoding(
            {
                constants.DepthDataCollatorConstants.INPUT_IDS: torch.tensor(
                    batch[constants.DepthDataCollatorConstants.INPUT_IDS],
                    dtype=torch.long,
                ),
                constants.DepthDataCollatorConstants.LABELS: torch.tensor(
                    batch[constants.DepthDataCollatorConstants.LABELS],
                    dtype=torch.long,
                ),
                constants.DepthDataCollatorConstants.TARGET_IDS: torch.tensor(
                    batch[constants.DepthDataCollatorConstants.TARGET_IDS],
                    dtype=torch.long,
                ),
                constants.DepthDataCollatorConstants.ENCODER_ATTENTION_MASK: torch.tensor(
                    batch[constants.DepthDataCollatorConstants.ENCODER_ATTENTION_MASK],
                    dtype=torch.int8,
                ),
                constants.DepthDataCollatorConstants.DECODER_ATTENTION_MASK: torch.tensor(
                    batch[constants.DepthDataCollatorConstants.DECODER_ATTENTION_MASK],
                    dtype=torch.int8,
                ),
                constants.DepthDataCollatorConstants.CROSS_ATTENTION_MASK: torch.tensor(
                    batch[constants.DepthDataCollatorConstants.CROSS_ATTENTION_MASK],
                    dtype=torch.int8,
                ),
                constants.DepthDataCollatorConstants.IS_SHUFFLED: torch.tensor(
                    batch[constants.DepthDataCollatorConstants.IS_SHUFFLED],
                    dtype=torch.bool,
                ),
                # TODO: Add "length" (number of tokens in input and target) to the batch encoding, and add support for
                #  it within the trainer class (i.e., log statistics about the length of the input and target
                #  sequences).
                # constants.DepthDataCollatorConstants.LENGTH: torch.tensor(
                #     batch[constants.DepthDataCollatorConstants.LENGTH],
                #     dtype=torch.long,
                # ),
            }
        )


@dataclass
class DEPTHDataCollatorFineTuning:

    def __init__(
            self,
            tokenizer: tokenizer_utils.DepthTokenizer,
            input_length: int,
            target_length: int,
            pad_token_id: int,
            decoder_start_token_id,
            seed: int = 42,
    ):
        """Initializes the data collator.

        Prepare the inputs and labels for fine-tuning a DEPTH pre-trained model on a downstream task.

        :param tokenizer: The tokenizer used for encoding the textual data within examples.
        :param input_length: The desired length of the input sequence (including special tokens). This value must be
            positive.
        :param target_length: The desired length of the target sequence (including special tokens). This value must be
            positive.
        :param pad_token_id: The id of the pad token for the model.
        :param decoder_start_token_id: The id of the decoder start token for the model.
        :param seed: A seed for the random number generator.
       """
        np.random.seed(seed)
        self.tokenizer = tokenizer
        self.input_length = input_length
        self.target_length = target_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(
            self,
            examples: typing.List[typing.Dict[str, np.ndarray]],
    ) -> transformers.BatchEncoding:
        """Generate a dictionary of input tensors to a Discourse T5 language model.

        :param examples: A list of examples, each of which is a dictionary of the form
            {
                "input_ids": List[int],
                "token_type_ids": List[int],
                "labels": List[int],
            }
        :return: A BatchEncoding instance containing the following fields:
            - "encoder_input_ids": The input ids of the model.
            - "encoder_self_attention_mask": The self attention mask of the encoder.
            - "cross_attention_mask": The cross attention mask of the decoder.
            - "decoder_input_ids": The input ids of the decoder.
            - "decoder_self_attention_mask": The self attention mask of the decoder.
            - "labels": The labels for the decoder.
        """
        # Extract input_ids, labels, and token_type_ids from the examples
        input_ids = np.stack([example["input_ids"] for example in examples])
        labels = np.stack([example["labels"] for example in examples])
        token_type_ids = np.stack([example["token_type_ids"] for example in examples])

        # Create target ids
        target_ids = torch.tensor(
                shift_tokens_right(
                input_ids=labels,
                pad_token_id=self.pad_token_id,
                decoder_start_token_id=self.decoder_start_token_id
            )
        )

        # Create attention masks
        encoder_attention_mask = torch.tensor(
            create_depth_encoder_self_attention_mask(
                input_ids=input_ids,
                input_token_type_ids=token_type_ids,
                tokenizer=self.tokenizer,
                sentence_token_ids=self.tokenizer.get_sentence_token_and_eosen_ids()
            )
        )

        # Convert input_ids and labels to tensors
        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)

        # Decoder attention mask is autoregressive
        decoder_attention_mask = torch.tril(
            torch.ones(
                (
                    len(target_ids),
                    self.target_length,
                    self.target_length
                 ),
                dtype=torch.int8,
            )
        )

        # Cross attention mask is all ones.
        cross_attention_mask = torch.ones(
            (
                len(target_ids),
                self.target_length,
                self.input_length
            ),
            dtype=torch.int8,
        )

        # Convert to tensors and return
        return transformers.BatchEncoding(
            {
                constants.DepthDataCollatorConstants.INPUT_IDS:input_ids,
                constants.DepthDataCollatorConstants.LABELS: labels,
                constants.DepthDataCollatorConstants.TARGET_IDS: target_ids,
                constants.DepthDataCollatorConstants.ENCODER_ATTENTION_MASK: encoder_attention_mask,
                constants.DepthDataCollatorConstants.DECODER_ATTENTION_MASK: decoder_attention_mask,
                constants.DepthDataCollatorConstants.CROSS_ATTENTION_MASK: cross_attention_mask,
            }
        )

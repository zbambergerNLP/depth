import random
import typing
import transformers
import torch
import numpy as np
from dataclasses import dataclass
import datasets
import string
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
            sentence_loss_coefficient: float = 1.0,
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
        self.sentence_loss_coefficient = sentence_loss_coefficient
        self.vocab_size = 32128

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

        # Create a vector of size [vocab_size], where each float value represents the weight of the corresponding token
        # in the vocabulary when computing the loss.
        sentence_token_ids = self.tokenizer.get_sentence_token_ids()
        loss_weights = torch.zeros(self.vocab_size)
        loss_weights[sentence_token_ids] = 1.0

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
                constants.DepthDataCollatorConstants.LOSS_WEIGHTS: loss_weights,
                constants.DepthDataCollatorConstants.SENTENCE_LOSS_COEFFICIENT: torch.tensor(
                    self.sentence_loss_coefficient,
                    dtype=torch.float32
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
                constants.DepthDataCollatorConstants.INPUT_IDS: input_ids,
                constants.DepthDataCollatorConstants.LABELS: labels,
                constants.DepthDataCollatorConstants.TARGET_IDS: target_ids,
                constants.DepthDataCollatorConstants.ENCODER_ATTENTION_MASK: encoder_attention_mask,
                constants.DepthDataCollatorConstants.DECODER_ATTENTION_MASK: decoder_attention_mask,
                constants.DepthDataCollatorConstants.CROSS_ATTENTION_MASK: cross_attention_mask,
            }
        )


# TODO: Decompose the logic of __call__ into smaller functions.
# TODO: Shift to using the constants in the constants module.
@dataclass
class DataCollatorForNI:
    tokenizer = None
    padding = 'longest'
    max_source_length = 512
    max_target_length = 128
    pad_to_multiple_of = 8
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    add_task_name: bool = False
    add_task_definition: bool = True
    num_pos_examples: int = 0
    num_neg_examples: int = 0
    add_explanation: bool = False
    tk_instruct: bool = False
    text_only: bool = False

    def __init__(self,
                 tokenizer,
                 padding='longest',
                 max_source_length=512,
                 max_target_length=128,
                 pad_to_multiple_of=8,
                 label_pad_token_id=-100,
                 return_tensors="pt",
                 add_task_name=False,
                 add_task_definition=True,
                 num_pos_examples=0,
                 num_neg_examples=0,
                 add_explanation=False,
                 tk_instruct=False,
                 text_only=False,
                 **kwargs):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.return_tensors = return_tensors
        self.add_task_name = add_task_name
        self.add_task_definition = add_task_definition
        self.num_pos_examples = num_pos_examples
        self.num_neg_examples = num_neg_examples
        self.add_explanation = add_explanation
        self.tk_instruct = tk_instruct
        self.text_only = text_only
        self.is_depth = isinstance(self.tokenizer, tokenizer_utils.DepthTokenizer)

    def __call__(self, batch, return_tensors=None):

        if return_tensors is None:
            return_tensors = self.return_tensors

        sources = []
        for instance in batch:
            if self.tk_instruct:
                all_valid_encodings = [
                    # instruction only
                    {
                        "add_task_name": False,
                        "add_task_definition": True,
                        "num_pos_examples": 0,
                        "num_neg_examples": 0,
                        "add_explanation": False,
                    },
                    # example only
                    {
                        "add_task_name": False,
                        "add_task_definition": False,
                        "num_pos_examples": 2,
                        "num_neg_examples": 0,
                        "add_explanation": False,
                    },
                    # instruction + pos examples
                    {
                        "add_task_name": False,
                        "add_task_definition": True,
                        "num_pos_examples": 2,
                        "num_neg_examples": 0,
                        "add_explanation": False,
                    },
                    # instruction + pos examples + neg examples
                    {
                        "add_task_name": False,
                        "add_task_definition": True,
                        "num_pos_examples": 2,
                        "num_neg_examples": 2,
                        "add_explanation": False,
                    },
                    # instruction + pos (w. explanation)
                    {
                        "add_task_name": False,
                        "add_task_definition": True,
                        "num_pos_examples": 2,
                        "num_neg_examples": 0,
                        "add_explanation": True,
                    },
                ]
                encoding_schema = random.choice(all_valid_encodings)
                add_task_name = encoding_schema["add_task_name"]
                add_task_definition = encoding_schema["add_task_definition"]
                num_pos_examples = encoding_schema["num_pos_examples"]
                num_neg_examples = encoding_schema["num_neg_examples"]
                add_explanation = encoding_schema["add_explanation"]
            else:
                add_task_name = self.add_task_name
                add_task_definition = self.add_task_definition
                num_pos_examples = self.num_pos_examples
                num_neg_examples = self.num_neg_examples
                add_explanation = self.add_explanation

            task_input = ""
            # add the input first.
            task_input += "Now complete the following example -\n"
            task_input += f"Input: {instance['Instance']['input'].strip()}"
            if not task_input[-1] in string.punctuation:
                task_input += "."
            task_input += "\n"
            task_input += "Output: "

            task_name = ""
            if add_task_name:
                task_name += instance["Task"] + ". "

            definition = ""
            if add_task_definition:
                if isinstance(instance["Definition"], list):
                    definition = (
                        "Definition: " + instance["Definition"][0].strip()
                    )
                else:
                    definition = "Definition: " + instance["Definition"].strip()
                if not definition[-1] in string.punctuation:
                    definition += "."
                definition += "\n\n"

            # try to add positive examples.
            pos_examples = []
            for idx, pos_example in enumerate(
                instance["Positive Examples"][:num_pos_examples]
            ):
                pos_example_str = f" Positive Example {idx+1} -\n"
                pos_example_str += f"Input: {pos_example['input'].strip()}"
                if not pos_example_str[-1] in string.punctuation:
                    pos_example_str += "."
                pos_example_str += "\n"
                pos_example_str += f" Output: {pos_example['output'].strip()}"
                if not pos_example_str[-1] in string.punctuation:
                    pos_example_str += "."
                pos_example_str += "\n"
                if add_explanation and "explanation" in pos_example:
                    pos_example_str += (
                        f" Explanation: {pos_example['explanation'].strip()}"
                    )
                    if not pos_example_str[-1] in string.punctuation:
                        pos_example_str += "."
                    pos_example_str += "\n"
                pos_example_str += "\n"

                prefix_len_pos = len(
                    self.tokenizer(
                        definition
                        + " ".join(pos_examples)
                        + pos_example_str
                        + task_input,
                        max_length=self.max_source_length,
                    )["input_ids"]
                )

                if prefix_len_pos < self.max_source_length:
                    pos_examples.append(pos_example_str)
                else:
                    break

            # try to add negative examples.
            neg_examples = []
            for idx, neg_example in enumerate(
                instance["Negative Examples"][:num_neg_examples]
            ):
                neg_example_str = f" Negative Example {idx+1} -\n"
                neg_example_str += f"Input: {neg_example['input'].strip()}"
                if not neg_example_str[-1] in string.punctuation:
                    neg_example_str += "."
                neg_example_str += "\n"
                neg_example_str += f" Output: {neg_example['output'].strip()}"
                if not neg_example_str[-1] in string.punctuation:
                    neg_example_str += "."
                neg_example_str += "\n"
                if add_explanation and "explanation" in neg_example:
                    neg_example_str += (
                        f" Explanation: {neg_example['explanation'].strip()}"
                    )
                    if not neg_example_str[-1] in string.punctuation:
                        neg_example_str += "."
                    neg_example_str += "\n"
                neg_example_str += "\n"
                if (
                    len(
                        self.tokenizer(
                            definition
                            + " ".join(pos_examples)
                            + " ".join(neg_examples)
                            + neg_example_str
                            + task_input,
                            max_length=self.max_source_length,
                        )["input_ids"]
                    )
                    < self.max_source_length
                ):
                    neg_examples.append(neg_example_str)
                else:
                    break

            source = (
                task_name
                + definition
                + "".join(pos_examples)
                + "".join(neg_examples)
                + task_input
            )
            if self.is_depth:
                tokenized_source = self.tokenizer(
                    source,
                    padding=constants.PaddingConstants.MAX_LENGTH.value,
                    max_length=self.max_source_length,
                    truncation=True,
                    randomize_sentence_token_ids=False,
                )[constants.DepthDataCollatorConstants.INPUT_IDS]
            else:
                tokenized_source = self.tokenizer(
                    source,
                    padding=constants.PaddingConstants.MAX_LENGTH.value if self.is_depth else self.padding,
                    max_length=self.max_source_length,
                    truncation=True,
                )[constants.DepthDataCollatorConstants.INPUT_IDS]
            if len(tokenized_source) <= self.max_source_length:
                sources.append(source)
            else:
                sources.append(
                    self.tokenizer.decode(
                        tokenized_source[: self.max_source_length],
                        skip_special_tokens=True,
                    )
                )

        if self.text_only:
            model_inputs = {"inputs": sources}
        else:
            if self.is_depth:
                model_inputs = {}
                input_encoding = self.tokenizer(
                    sources,
                    padding=constants.PaddingConstants.MAX_LENGTH.value,
                    max_length=self.max_source_length,
                    truncation=True,
                    return_tensors=self.return_tensors,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                )

                input_ids = np.array(input_encoding["input_ids"])
                token_type_ids = np.array(input_encoding["token_type_ids"])
                model_inputs[constants.DepthDataCollatorConstants.INPUT_IDS] = torch.tensor(input_ids)

                encoder_attention_mask = torch.tensor(
                    create_depth_encoder_self_attention_mask(
                        input_ids=input_ids,
                        input_token_type_ids=token_type_ids,
                        tokenizer=self.tokenizer,
                        sentence_token_ids=self.tokenizer.get_sentence_token_and_eosen_ids()
                    )
                )
                model_inputs[constants.DepthDataCollatorConstants.ENCODER_ATTENTION_MASK] = encoder_attention_mask

            else:
                model_inputs = self.tokenizer(
                    sources,
                    max_length=self.max_source_length,
                    padding=self.padding,
                    return_tensors=self.return_tensors,
                    truncation=True,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                )

        if "output" in batch[0]["Instance"] and batch[0]["Instance"]["output"]:
            # Randomly select one reference if multiple are provided.
            labels = [random.choice(ex["Instance"]["output"]) for ex in batch]
            if self.text_only:
                model_inputs["labels"] = labels
            else:
                if self.is_depth:
                    labels = self.tokenizer.batch_encode_plus(
                        labels,
                        padding=constants.PaddingConstants.MAX_LENGTH.value,
                        max_length=self.max_target_length,
                        truncation=True,
                        return_tensors=self.return_tensors,
                        pad_to_multiple_of=self.pad_to_multiple_of,
                    )
                    labels = np.array(labels[constants.DepthDataCollatorConstants.INPUT_IDS])
                    model_inputs[constants.DepthDataCollatorConstants.LABELS] = torch.tensor(labels)
                    decoder_attention_mask = torch.tril(
                        torch.ones(
                            (
                                len(labels),
                                self.max_target_length,
                                self.max_target_length
                            ),
                            dtype=torch.int8,
                        )
                    )
                    model_inputs[constants.DepthDataCollatorConstants.DECODER_ATTENTION_MASK] = decoder_attention_mask
                    cross_attention_mask = torch.ones(
                        (
                            len(labels),                # batch size
                            self.max_target_length,     # query length (decoder sequence length)
                            self.max_source_length      # key length   (encoder sequence length)
                        ),
                        dtype=torch.int8,
                    )
                    model_inputs[constants.DepthDataCollatorConstants.CROSS_ATTENTION_MASK] = cross_attention_mask

                else:
                    labels = self.tokenizer(
                        labels,
                        max_length=self.max_target_length,
                        padding=self.padding,
                        return_tensors=self.return_tensors,
                        truncation=True,
                        pad_to_multiple_of=self.pad_to_multiple_of,
                    )
                    label_mask = labels["attention_mask"].bool()
                    model_inputs["labels"] = labels["input_ids"].masked_fill(
                        ~label_mask, self.label_pad_token_id
                    )

        else:
            model_inputs["labels"] = None

        return model_inputs

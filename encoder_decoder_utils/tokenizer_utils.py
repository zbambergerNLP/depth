import os
import typing

from transformers import T5TokenizerFast, T5Tokenizer

import logging
import numpy as np
from torch import inf
import torch
import transformers
from typing import (
    List,
    Union,
    Tuple,
)
import nltk

from .constants import (
    DEPTHTokenizerConstants,
)

from transformers.tokenization_utils_base import (
    AddedToken,
    BatchEncoding,
    PreTokenizedInput,
    PreTokenizedInputPair,
    PreTrainedTokenizerBase,
    SpecialTokensMixin,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)

# TODO: Implement a sub-class of the fast T5 tokenizer in order to speed up the tokenization process.
# TODO: Create a set of unit tests for the new tokenizer suggested in the TODO above.

SENTENCE_PIECE_UNDERSCORE = u"\u2581"

VERY_LARGE_INTEGER = int(1e30)  # This is used to set the max input length for a model with infinite size input
LARGE_INTEGER = int(1e20)  # This is used when we need something big but slightly smaller than VERY_LARGE_INTEGER

NO_PAD_TOKEN_FOR_BATCH_MSG = (
    "No padding token is set for this model, therefore no batch can be made with uneven "
    "sequences. Set a padding token or adjust the lengths of the sequences building the "
    "batch so that every sequence is of the same length."
)

UNEVEN_SEQUENCES_FOR_BATCH_MSG = (
    "The sequences building the batch are not of the same size, no tensor "
    "can be built. Set `pad_to_max_length=True` to pad the smaller sequences"
    "up to the larger sequence's length."
)

logger = logging.getLogger(__name__)
# SENT = 'sent'
# SENT_TOKEN = '<sent>'
# END_OF_SENTENCE_TOKEN = '<eosen>'
# CLS = 'cls'
# CLS_TOKEN = '<cls>'
# ADDITIONAL_SPECIAL_TOKENS = 'additional_special_tokens'
nltk.download('punkt')

# Truncation and Padding
TRUNCATION_LONGEST_FIRST = 'longest_first'
PADDING_LONGEST = 'longest'
PADDING_MAX_LENGTH = 'max_length'
PADDING_SIDE_RIGHT = 'right'
PADDING_SIDE_LEFT = 'left'


def prepare_for_model(
        ids: List[int],
        tokenizer: transformers.PreTrainedTokenizer,
        sentence_tokens: List[str],
        max_length: int = 512,
        truncation: Union[bool, str] = TRUNCATION_LONGEST_FIRST,
        padding: Union[bool, str] = True,
        return_tensors: str = None,
        return_token_type_ids: bool = True,
        return_attention_mask: bool = True,
        return_special_tokens_mask: bool = True,
        return_lengths: bool = True,
) -> transformers.BatchEncoding:
    """

    :param ids: List of tokenized input ids. Can be obtained from a string by chaining the `tokenize` and
        `convert_tokens_to_ids` methods.
    :param tokenizer: A tokenizer that decomposes a string into a list of tokens.
    :param sentence_tokens: A list of tokens that represent the beginning of sentences.
    :param max_length: maximum length of the returned list. Will truncate by taking into account the special tokens.
    :param truncation: string selected in the following options:
        - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
            starting from the longest one at each token (when there is a pair of input sequences)
        - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_length)
    :param padding: Activates and controls padding. Accepts the following values:
        True or 'longest': Pad to the longest sequence in the batch (or no padding if only a single sequence if
            provided).
        'max_length': Pad to a maximum length specified with the argument max_length or to the maximum acceptable input
            length for the model if that argument is not provided.
        False or 'do_not_pad' (default): No padding (i.e., can output a batch with sequences of different lengths).
    :param return_tensors: can be set to 'tf' or 'pt' to return respectively TensorFlow tf.constant or PyTorch
        torch.Tensor instead of a list of python integers.
    :param return_token_type_ids: Set to False to avoid returning token_type_ids (default: set to model specifics).
    :param return_attention_mask: Set to False to avoid returning attention mask (default: set to model specifics)
    :param return_special_tokens_mask: Set to True to return special tokens mask information (default False).
    :param return_lengths: If set the resulting dictionary will include the length of each encoded inputs
    :return: A Dictionary of shape::
        {
            input_ids: list[int],
            token_type_ids: list[int] if return_token_type_ids is True (default)
            num_truncated_tokens: int if a ``max_length`` is specified and return_overflowing_tokens is True
            special_tokens_mask: list[int] if ``add_special_tokens`` if set to ``True`` and return_special_tokens_mask is True
            length: int if return_lengths is True
        }

        With the fields:
            - ``input_ids``: list of token ids to be fed to a model
            - ``token_type_ids``: list of token type ids to be fed to a model

            - ``overflowing_tokens``: list of overflowing tokens if a max length is specified.
            - ``num_truncated_tokens``: number of overflowing tokens a ``max_length`` is specified
            - ``special_tokens_mask``: if adding special tokens, this is a list of [0, 1], with 0 specifying special added
                tokens and 1 specifying sequence tokens.
            - ``length``: this is the length of ``input_ids``
    """
    encoded_inputs = {}

    # Add special sentence tokens
    sequence = tokenizer.build_inputs_with_special_tokens(ids)
    sentence_token_ids = tokenizer.convert_tokens_to_ids(sentence_tokens)
    token_type_ids = [1]
    sentence_index = 0
    for token_index, token in enumerate(sequence[1:]):
        if token in sentence_token_ids or token == tokenizer.eos_token:
            sentence_index += 1
        if token == 0:
            sentence_index = 0
        token_type_ids.append(sentence_index)
    token_type_ids = token_type_ids[:max_length]  # Truncate to max length

    seq_len = len(sequence)

    # Truncation: Handle max sequence length
    if max_length and seq_len > max_length:
        sequence, _, _ = tokenizer.truncate_sequences(
            sequence,
            num_tokens_to_remove=seq_len - max_length,
            truncation_strategy=truncation,
        )

    encoded_inputs[DEPTHTokenizerConstants.NUM_TRUNCATED_TOKENS] = seq_len - max_length

    # Build output dictionary
    encoded_inputs[DEPTHTokenizerConstants.INPUT_IDS] = sequence
    if return_token_type_ids:
        encoded_inputs[DEPTHTokenizerConstants.TOKEN_TYPE_IDS] = token_type_ids
    if return_special_tokens_mask:
        encoded_inputs[DEPTHTokenizerConstants.SPECIAL_TOKENS_MASK] = (
            tokenizer.get_special_tokens_mask(sequence, already_has_special_tokens=True))

    # Check lengths
    assert max_length is None or len(encoded_inputs[DEPTHTokenizerConstants.INPUT_IDS]) <= max_length
    if max_length is None and len(encoded_inputs[DEPTHTokenizerConstants.INPUT_IDS]) > tokenizer.model_max_length:
        logger.warning(
            "Token indices sequence length is longer than the specified maximum sequence length "
            "for this model ({} > {}). Running this sequence through the model will result in "
            "indexing errors".format(len(sequence), tokenizer.model_max_length)
        )

    # Padding
    user_specified_padding = (
        # User did not specify 'no padding'
            (isinstance(padding, bool) and padding) or
            (isinstance(padding, str) and padding in [PADDING_LONGEST, PADDING_MAX_LENGTH])
    )
    needs_to_be_padded = user_specified_padding and (
            max_length
            and len(encoded_inputs[DEPTHTokenizerConstants.INPUT_IDS]) < max_length
            or max_length is None
            and len(encoded_inputs[DEPTHTokenizerConstants.INPUT_IDS]) < tokenizer.model_max_length <= LARGE_INTEGER
    )
    if user_specified_padding and max_length is None and tokenizer.model_max_length > LARGE_INTEGER:
        logger.warning(
            "Sequence can't be padded as no maximum length is specified and the model maximum length is too high."
        )

    if needs_to_be_padded:
        difference = (max_length if max_length is not None else tokenizer.model_max_length) - len(
            encoded_inputs[DEPTHTokenizerConstants.INPUT_IDS]
        )
        if tokenizer.padding_side == PADDING_SIDE_RIGHT:
            if return_attention_mask:
                encoded_inputs[DEPTHTokenizerConstants.ATTENTION_MASK] = [1] * len(
                    encoded_inputs[DEPTHTokenizerConstants.INPUT_IDS]) + [0] * difference
            if return_token_type_ids:
                encoded_inputs[DEPTHTokenizerConstants.TOKEN_TYPE_IDS] = (
                        encoded_inputs[DEPTHTokenizerConstants.TOKEN_TYPE_IDS] + [
                    tokenizer.pad_token_type_id] * difference
                )
            if return_special_tokens_mask:
                encoded_inputs[DEPTHTokenizerConstants.SPECIAL_TOKENS_MASK] = (
                        encoded_inputs[DEPTHTokenizerConstants.SPECIAL_TOKENS_MASK] + [1] * difference)
            encoded_inputs[DEPTHTokenizerConstants.INPUT_IDS] = (
                    encoded_inputs[DEPTHTokenizerConstants.INPUT_IDS] + [tokenizer.pad_token_id] * difference)
        elif tokenizer.padding_side == PADDING_SIDE_LEFT:
            if return_attention_mask:
                encoded_inputs[DEPTHTokenizerConstants.ATTENTION_MASK] = (
                        [0] * difference + [1] * len(encoded_inputs[DEPTHTokenizerConstants.INPUT_IDS]))
            if return_token_type_ids:
                encoded_inputs[DEPTHTokenizerConstants.TOKEN_TYPE_IDS] = (
                        [tokenizer.pad_token_type_id] * difference + encoded_inputs[DEPTHTokenizerConstants.TOKEN_TYPE_IDS]
                )
            if return_special_tokens_mask:
                encoded_inputs[DEPTHTokenizerConstants.SPECIAL_TOKENS_MASK] = (
                        [1] * difference + encoded_inputs[DEPTHTokenizerConstants.SPECIAL_TOKENS_MASK])
            encoded_inputs[DEPTHTokenizerConstants.INPUT_IDS] = (
                    [tokenizer.pad_token_id] * difference + encoded_inputs[DEPTHTokenizerConstants.INPUT_IDS])
        else:
            raise ValueError("Invalid padding strategy:" + str(tokenizer.padding_side))
    else:
        if return_attention_mask:
            encoded_inputs[DEPTHTokenizerConstants.ATTENTION_MASK] = [1] * len(
                encoded_inputs[DEPTHTokenizerConstants.INPUT_IDS])
        # TODO: support returning token type IDs and special tokens mask even if the user does not specify the padding
        #  option.

    if return_lengths:
        encoded_inputs[DEPTHTokenizerConstants.INPUT_LENGTH] = len(encoded_inputs[DEPTHTokenizerConstants.INPUT_IDS])

    # Prepare model inputs as tensors if asked
    if return_tensors == "pt":
        encoded_inputs[DEPTHTokenizerConstants.INPUT_IDS] = torch.tensor(
            [encoded_inputs[DEPTHTokenizerConstants.INPUT_IDS]])
        if DEPTHTokenizerConstants.TOKEN_TYPE_IDS in encoded_inputs:
            encoded_inputs[DEPTHTokenizerConstants.TOKEN_TYPE_IDS] = torch.tensor(
                [encoded_inputs[DEPTHTokenizerConstants.TOKEN_TYPE_IDS]])
        if DEPTHTokenizerConstants.ATTENTION_MASK in encoded_inputs:
            encoded_inputs[DEPTHTokenizerConstants.ATTENTION_MASK] = torch.tensor(
                [encoded_inputs[DEPTHTokenizerConstants.ATTENTION_MASK]])
    elif return_tensors is not None:
        logger.warning(
            "Unable to convert output to tensors format {}, PyTorch or TensorFlow is not available.".format(
                return_tensors
            )
        )

    return transformers.BatchEncoding(encoded_inputs)

def convert_to_tensors_(batch_outputs: dict, return_tensors: str) -> None:
    # Do the tensor conversion in batch
    for key, value in batch_outputs.items():
        if return_tensors == "pt":
            try:
                batch_outputs[key] = torch.tensor(value)
            except ValueError:
                raise ValueError(UNEVEN_SEQUENCES_FOR_BATCH_MSG)
            except RuntimeError:
                if None in [item for sequence in value for item in sequence]:
                    raise ValueError(NO_PAD_TOKEN_FOR_BATCH_MSG)
                else:
                    raise

        elif return_tensors is not None:
            logger.warning(
                "Unable to convert output to tensors format {}, PyTorch or TensorFlow is not available.".format(
                    return_tensors
                )
            )


def batch_encode_plus(
        batch_text: Union[List[str], List[List[str]], List[List[int]]],
        tokenizer: transformers.PreTrainedTokenizer,
        seed: int = None,
        max_num_sentences_in_text: int = 20,
        max_length: int = 512,  # In BERT this value is 512.
        truncation: Union[bool, str, transformers.tokenization_utils.TruncationStrategy] = TRUNCATION_LONGEST_FIRST,
        padding: Union[bool, str, transformers.tokenization_utils.PaddingStrategy] = PADDING_MAX_LENGTH,
        return_tensors: str = None,
        return_token_type_ids: bool = True,
        return_attention_mask: bool = True,
        return_special_tokens_mask: bool = True,
) -> transformers.BatchEncoding:
    """

    :param batch_text: The batch of sequences to be encoded. Each sequence can be a string, a list of strings
        (tokenized string using the tokenize` method) or a list of integers (tokenized string ids using the
        `convert_tokens_to_ids` method).
    :param tokenizer: A tokenizer that decomposes a string into a list of tokens.
    :param seed: The seed for the random number generator.
    :param max_num_sentences_in_text: The maximum number of sentences to consider within a single text input. In SLM,
        this value was 20. The higher this value, the more difficult it is to un-shuffle sentences.
    :param max_length: If set to a number, will limit the total sequence returned so that it has a maximum length.
    :param truncation: Activates and controls truncation. Accepts the following values:
        True or 'longest_first': Truncate to a maximum length specified with the argument max_length or to the maximum
            acceptable input length for the model if that argument is not provided. This will truncate token by token,
            removing a token from the longest sequence in the pair if a pair of sequences (or a batch of pairs) is
            provided.
        False or 'do_not_truncate' (default): No truncation (i.e., can output batch with sequence lengths greater than
            the model maximum admissible input size).
    :param padding: Activates and controls padding. Accepts the following values:
        True or 'longest': Pad to the longest sequence in the batch (or no padding if only a single sequence if
            provided).
        'max_length': Pad to a maximum length specified with the argument max_length or to the maximum acceptable input
            length for the model if that argument is not provided.
        False or 'do_not_pad' (default): No padding (i.e., can output a batch with sequences of different lengths).
    :param return_tensors: Can be set to 'tf' or 'pt' to return respectively TensorFlow `tf.constant` or PyTorch
        `torch.Tensor` instead of a list of python integers.
    :param return_token_type_ids: Whether to return token type IDs.
    :param return_attention_mask: Whether to return the attention mask.
    :param return_special_tokens_mask: Set to True to return special tokens mask information.
    :return: A Dictionary of shape:
        {
            input_ids: list[int],
            token_type_ids: list[int] if return_token_type_ids is True (default)
            attention_mask: list[int] if return_attention_mask is True (default)
            overflowing_tokens: list[int] if a ``max_length`` is specified and return_overflowing_tokens is True
            num_truncated_tokens: int if a ``max_length`` is specified and return_overflowing_tokens is True
            special_tokens_mask: list[int] if ``add_special_tokens`` if set to ``True``
            and return_special_tokens_mask is True
        }

        With the fields:

        - ``input_ids``: list of token ids to be fed to a model
        - ``token_type_ids``: list of token type ids to be fed to a model
        - ``attention_mask``: list of indices specifying which tokens should be attended to by the model
        - ``num_truncated_tokens``: number of overflowing tokens a ``max_length`` is specified
        - ``special_tokens_mask``: if adding special tokens, this is a list of [0, 1], with 0 specifying special added
          tokens and 1 specifying sequence tokens.
    """
    if seed is None:
        seed = np.random.randint(low=100, high=10_000)

    def get_input_ids(text_to_tokenize):
        if isinstance(text_to_tokenize, str):
            tokenized_text = tokenize_text(
                text=text_to_tokenize,
                tokenizer=tokenizer,
                seed=seed,
                max_num_sentences_in_text=max_num_sentences_in_text,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
            )
            return tokenizer.convert_tokens_to_ids(tokenized_text)
        elif isinstance(text_to_tokenize, (list, tuple)) and len(text_to_tokenize) > 0 and isinstance(
                text_to_tokenize[0], str):
            return tokenizer.convert_tokens_to_ids(text)
        elif isinstance(text_to_tokenize, (list, tuple)) and len(text_to_tokenize) > 0 and isinstance(
                text_to_tokenize[0], int):
            return text_to_tokenize
        else:
            raise ValueError(
                "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
            )

    # Throw an error if we can pad because there is no padding token
    if (
            (isinstance(padding, str) and (padding == 'longest' or padding == 'max_length')) or
            (isinstance(padding, bool) and padding)
    ) and tokenizer.pad_token_id is None:
        raise ValueError(
            "Unable to set proper padding strategy as the tokenizer does not have a padding token. In this case please "
            "set the `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` or add a new pad token via the "
            "function add_special_tokens if you want to use a padding strategy"
        )

    if isinstance(batch_text, str):
        input_ids = [get_input_ids(batch_text)]
    else:
        input_ids = []
        for text in batch_text:
            ids = get_input_ids(text)
            input_ids.append(ids)
            seed += 1

    if max_length is None:
        max_length = max([len(ids) for ids in input_ids])

    special_tokens = tokenizer.special_tokens_map[DEPTHTokenizerConstants.ADDITIONAL_SPECIAL_TOKENS]
    sentence_tokens = list(filter(lambda token: f'<{DEPTHTokenizerConstants.SENT}' in token, special_tokens))
    batch_outputs = {}

    for example in input_ids:
        # Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by
        # the model. It adds special tokens, truncates sequences if overflowing while taking into account
        # the special tokens and manages a window stride for overflowing tokens
        outputs = prepare_for_model(
            example,
            tokenizer=tokenizer,
            sentence_tokens=sentence_tokens,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_special_tokens_mask=return_special_tokens_mask,
            return_lengths=True,
            return_tensors=None,  # We will convert the whole batch to tensors at the end
        )

        for key, value in outputs.items():
            if key not in batch_outputs:
                batch_outputs[key] = []
            batch_outputs[key].append(value)

    if return_tensors is not None:
        convert_to_tensors_(batch_outputs, return_tensors)
    return transformers.BatchEncoding(batch_outputs)


def add_sentence_tokens_to_text(
        sentences: str,
        sentence_tokens: List[str],
        seed: int,
        max_num_sentences_in_text: int = 20,
        end_of_sentence_token: str = DEPTHTokenizerConstants.END_OF_SENTENCE_TOKEN,
) -> str:
    """Add SENT tokens in the beginning of each sentence.

    Truncate sentences that exceed 'max_num_sentences_in_tex'.

    :param sentences: A string composed of one or more sentences. Strings without punctuation or
        coherent sentence ends are considered as single sentences.
    :param sentence_tokens: All possible sentence tokens. A randomly selected subset of sentence tokens are prepended to
        each sentence. This token is used to aggregate information across tokens within the sentence (as introduced in
        the SLM paper).
    :param max_num_sentences_in_text: The maximum number of sentences to consider within a single
        text input. In SLM, this value was 20. The higher this value, the more difficult it is to
        un-shuffle sentences.
    :param seed: The seed for the random number generator.
    :param end_of_sentence_token: The token that indicates the end of a sentence (<eosen>). This token is used to
        indicate the end of a sentence, which enables the model to realize during inference that it should look back at
        previous sentences.
    :return: A string where sentences are prefixed with SENT tokens.
    """
    np.random.seed(seed)
    segmented_sentences = nltk.tokenize.sent_tokenize(sentences)
    num_sentences = len(segmented_sentences)
    segment_sentence_tokens = np.copy(sentence_tokens)
    np.random.shuffle(segment_sentence_tokens)

    # TODO: Randomly merge sentences if the number of sentences is greater than max_num_sentences_in_text.
    #  This is to avoid having to truncate sentences.
    if num_sentences > max_num_sentences_in_text:
        segmented_sentences = segmented_sentences[:max_num_sentences_in_text]
        segment_sentence_tokens = segment_sentence_tokens[:max_num_sentences_in_text]
    modified_sentences = ''.join([
        f'{end_of_sentence_token}{sentence_token}{sentence}' for sentence_token, sentence in zip(
            segment_sentence_tokens, segmented_sentences)])
    modified_sentences = f'{modified_sentences}{end_of_sentence_token}'
    return modified_sentences


def tokenize_text(
        text: Union[str, List[str]],
        tokenizer: Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast],
        seed: int,
        max_num_sentences_in_text: int = 20,
        padding: Union[bool, str, transformers.tokenization_utils.PaddingStrategy] = True,
        truncation: Union[bool, str, transformers.tokenization_utils.TruncationStrategy] = True,
        max_length: int = 512,
) -> Union[List[str], List[List[str]]]:
    """Tokenize a potentially long text into a sequence of tokens. New sentences are prepended with designated
    sentence tokens.

    :param text: A of potentially long text (i.e., text that consist of multiple sentences). This parameter also
        supports tokenizing a sequence (batch) of long texts.
    :param tokenizer: A tokenizer that decomposes a string into a list of tokens.
    :param seed: The seed for the random number generator.
    :param max_num_sentences_in_text: The maximum number of sentences to consider within a single text input. In SLM,
        this value was 20. The higher this value, the more difficult it is to un-shuffle sentences.
    :param padding: Activates and controls padding. Accepts the following values:
        True or 'longest': Pad to the longest sequence in the batch (or no padding if only a single sequence if
            provided).
        'max_length': Pad to a maximum length specified with the argument max_length or to the maximum acceptable input
            length for the model if that argument is not provided.
        False or 'do_not_pad' (default): No padding (i.e., can output a batch with sequences of different lengths).
    :param truncation: Activates and controls truncation. Accepts the following values:
        True or 'longest_first': Truncate to a maximum length specified with the argument max_length or to the maximum
            acceptable input length for the model if that argument is not provided. This will truncate token by token,
            removing a token from the longest sequence in the pair if a pair of sequences (or a batch of pairs) is
            provided.
        False or 'do_not_truncate' (default): No truncation (i.e., can output batch with sequence lengths greater than
            the model maximum admissible input size).
    :param max_length:  Controls the maximum length to use by one of the truncation/padding parameters. If left unset or
        set to None, this will use the predefined model maximum length if a maximum length is required by one of the
        truncation/padding parameters.
    :return: A list that consists of a sequence of string tokens (either words or sub-words). Includes a designated
        sentence representation token (SENT)).
    """
    # TODO: Should the architecture enable 20 sentences of arbitrary length as in XLNet?

    special_tokens = tokenizer.special_tokens_map[DEPTHTokenizerConstants.ADDITIONAL_SPECIAL_TOKENS]
    sentence_tokens = list(filter(lambda token: f'<{DEPTHTokenizerConstants.SENT}' in token, special_tokens))
    if isinstance(text, str):
        augmented_text = add_sentence_tokens_to_text(
            sentences=text,
            max_num_sentences_in_text=max_num_sentences_in_text,
            sentence_tokens=sentence_tokens,
            seed=seed
        )
        return tokenizer.tokenize(
            text=augmented_text,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )
    elif isinstance(text, list) and isinstance(text[0], str):
        tokenized_text = []
        for text_example in text:
            augmented_text_example = add_sentence_tokens_to_text(
                sentences=text_example,
                max_num_sentences_in_text=max_num_sentences_in_text,
                sentence_tokens=sentence_tokens,
                seed=seed,
            )
            tokenized_text.append(tokenizer.tokenize(
                text=augmented_text_example,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
            ))
        return tokenized_text
    else:
        raise ValueError(
            "Input is not valid. Should be a string, or a list of strings."
        )


def create_discourse_tokenizer(
        model_name: str,
        cache_dir: str = None,
        use_fast: bool = False,
        use_auth_token: bool = False,
        num_sent_tokens: int = 20,
) -> Tuple[transformers.PreTrainedTokenizer, int]:
    """
    Create a tokenizer for encoder-decoder models which includes sentence tokens in it's vocabulary.

    :param model_name: The name of the model and associated tokenizer. The project directory must contain a
        sub-directory with this name.
    :param cache_dir: The directory where the tokenizer will be cached.
    :param use_fast: Whether or not to use a fast tokenizer.
    :param use_auth_token: Whether or not to use an authentication token for HuggingFace's model hub.
    :param num_sent_tokens: The number of designated tokens meant to aggregate sentence representations. These
        tokens are used as part of the sentence un-shuffling pre-training task.
    :return: A transformers.PreTrainedTokenizer
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name,
        cache_dir=cache_dir,
        use_fast=use_fast,
        use_auth_token=use_auth_token,
    )
    additional_special_tokens = tokenizer.special_tokens_map[DEPTHTokenizerConstants.ADDITIONAL_SPECIAL_TOKENS]
    sent_tokens = [f'<{DEPTHTokenizerConstants.SENT}_{i}>' for i in range(num_sent_tokens)]
    additional_special_tokens.extend(sent_tokens)
    additional_special_tokens.append(DEPTHTokenizerConstants.END_OF_SENTENCE_TOKEN)
    special_tokens_dict = {DEPTHTokenizerConstants.ADDITIONAL_SPECIAL_TOKENS: additional_special_tokens}
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict=special_tokens_dict)
    return tokenizer, num_added_tokens


VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "t5-small": "https://huggingface.co/t5-small/resolve/main/spiece.model",
        "t5-base": "https://huggingface.co/t5-base/resolve/main/spiece.model",
        "t5-large": "https://huggingface.co/t5-large/resolve/main/spiece.model",
        "t5-3b": "https://huggingface.co/t5-3b/resolve/main/spiece.model",
        "t5-11b": "https://huggingface.co/t5-11b/resolve/main/spiece.model",
    },
    "tokenizer_file": {
        "t5-small": "https://huggingface.co/t5-small/resolve/main/tokenizer.json",
        "t5-base": "https://huggingface.co/t5-base/resolve/main/tokenizer.json",
        "t5-large": "https://huggingface.co/t5-large/resolve/main/tokenizer.json",
        "t5-3b": "https://huggingface.co/t5-3b/resolve/main/tokenizer.json",
        "t5-11b": "https://huggingface.co/t5-11b/resolve/main/tokenizer.json",
    },
}


# TODO(PVP) - this should be removed in Transformers v5
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "t5-small": 512,
    "t5-base": 512,
    "t5-large": 512,
    "t5-3b": 512,
    "t5-11b": 512,
}

class DepthTokenizer(T5TokenizerFast):

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path,
            max_num_sentences_in_text: int = 20,
            *inputs,
            **kwargs,
    ):
        tokenizer = super().from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
        tokenizer.__class__ = cls
        tokenizer.set_num_sentence_tokens(num_sent_tokens=max_num_sentences_in_text)
        return tokenizer

    def set_num_sentence_tokens(self, num_sent_tokens: int):
        additional_special_tokens = self.special_tokens_map[DEPTHTokenizerConstants.ADDITIONAL_SPECIAL_TOKENS]
        sent_tokens = [f'<{DEPTHTokenizerConstants.SENT}_{i}>' for i in range(num_sent_tokens)]
        additional_special_tokens.extend(sent_tokens)
        additional_special_tokens.append(DEPTHTokenizerConstants.END_OF_SENTENCE_TOKEN)
        special_tokens_dict = {DEPTHTokenizerConstants.ADDITIONAL_SPECIAL_TOKENS: additional_special_tokens}
        self.add_special_tokens(special_tokens_dict=special_tokens_dict)
        self.max_num_sentences_in_text = num_sent_tokens


    # TODO: Migrate to the signature defined here:
    #  https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L2932-L3003
    def __call__(
            self,
            text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
            text_target: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
            add_special_tokens: bool = True,
            padding: transformers.utils.PaddingStrategy = transformers.utils.PaddingStrategy.DO_NOT_PAD,
            truncation: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
            max_length: typing.Optional[int] = None,
            stride: int = 0,
            is_split_into_words: bool = False,
            pad_to_multiple_of: typing.Optional[int] = None,
            return_tensors: typing.Optional[str] = None,
            return_token_type_ids: typing.Optional[bool] = True,
            return_attention_mask: typing.Optional[bool] = True,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            seed: int = 42,

    ) -> BatchEncoding:
        def get_input_ids(text_to_tokenize):
            if isinstance(text_to_tokenize, str):
                tokenized_text = tokenize_text(
                    text=text_to_tokenize,
                    tokenizer=self,
                    seed=seed,
                    max_num_sentences_in_text=self.max_num_sentences_in_text,
                    padding=padding,
                    truncation=truncation,
                    max_length=max_length,
                )
                return self.convert_tokens_to_ids(tokenized_text)
            elif isinstance(text_to_tokenize, (list, tuple)) and len(text_to_tokenize) > 0 and isinstance(
                    text_to_tokenize[0], str):
                return self.convert_tokens_to_ids(text)
            elif isinstance(text_to_tokenize, (list, tuple)) and len(text_to_tokenize) > 0 and isinstance(
                    text_to_tokenize[0], int):
                return text_to_tokenize
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        # Throw an error if we can pad because there is no padding token
        if (
                (
                        isinstance(padding, str) and
                        (padding == 'longest' or padding == 'max_length')
                ) or (
                isinstance(padding, bool) and padding
                )
        ) and self.pad_token_id is None:
            raise ValueError(
                "Unable to set proper padding strategy as the tokenizer does not have a padding token. In this case please "
                "set the `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` or add a new pad token via the "
                "function add_special_tokens if you want to use a padding strategy"
            )

        if isinstance(text, str):
            input_ids = [get_input_ids(text)]
        else:
            input_ids = []
            for entry in text:
                ids = get_input_ids(entry)
                input_ids.append(ids)
                seed += 1

        if max_length is None:
            max_length = max([len(ids) for ids in input_ids])

        special_tokens = self.special_tokens_map[DEPTHTokenizerConstants.ADDITIONAL_SPECIAL_TOKENS]
        sentence_tokens = list(filter(lambda token: f'<{DEPTHTokenizerConstants.SENT}' in token, special_tokens))
        batch_outputs = {}

        for example in input_ids:
            # Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by
            # the model. It adds special tokens, truncates sequences if overflowing while taking into account
            # the special tokens and manages a window stride for overflowing tokens
            # TODO: Move 'prepare_for_model' to a function within this class.
            outputs = prepare_for_model(
                example,
                tokenizer=self,
                sentence_tokens=sentence_tokens,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_attention_mask=return_attention_mask,
                return_token_type_ids=return_token_type_ids,
                return_special_tokens_mask=return_special_tokens_mask,
                return_lengths=return_length,
                return_tensors=return_tensors,  # We will convert the whole batch to tensors at the end
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        if return_tensors is not None:
            self.convert_to_tensors_(batch_outputs, return_tensors)

        return transformers.BatchEncoding(batch_outputs)

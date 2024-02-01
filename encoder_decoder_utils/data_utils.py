import random
import string
import typing

import numpy as np
import torch
from typing import Dict, List
import transformers
from encoder_decoder_utils.constants import (
    TokenizerConstants
)


def tokenize_function(
        examples: typing.Dict[str, typing.Any],
        tokenizer: transformers.PreTrainedTokenizer,
        in_length: int,
) -> Dict[str, np.ndarray]:
    """
    Tokenizes batches of examples for pre-training a T5 model with merging.

    This function is used to tokenize batches of examples for pre-training a T5 model. It flattens examples into a
    single sequence, and then truncates the sequence to the maximum length of the model. The sequence is then
    reshaped into a matrix of shape (num_examples, in_length), where num_examples is the number of examples in the
    batch, and in_length is the maximum length of the model. The result is returned as a dictionary mapping the
    input_ids to the matrix of shape (num_examples, in_length).


    :param examples: A batch in the form of a dictionary mapping, mapping column names to their respective values.
    :param tokenizer: A function which converts string tokens into input_ids and other model inputs.
    :param in_length: The maximum length of the input sequence.
    :return: A dictionary containing a mapping from model input names (e.g., `input_ids`) to model input values.
    """
    assert "text" in examples, "You must pass a dictionary containing a 'text' key."

    tokenizer_out = tokenizer(
        text=examples["text"],
        return_attention_mask=False,
    )

    input_ids = tokenizer_out[TokenizerConstants.INPUT_IDS]

    # TODO: Consider additional approaches to compressing the input_ids. For example, could try to dynamically
    #  concatenate as few examples as possible per row such that the total length of the input_ids is less than
    #  the in_length.
    concatenated_ids = np.concatenate(input_ids)

    total_length = concatenated_ids.shape[0]
    total_length = (total_length // in_length) * in_length

    concatenated_ids = concatenated_ids[:total_length]
    concatenated_ids.reshape(-1, in_length)
    result = {TokenizerConstants.INPUT_IDS: concatenated_ids}

    return result


def tokenizer_function_t5_pre_training(
        examples: typing.Dict[str, typing.List[str]],
        tokenizer: transformers.T5Tokenizer,
        in_length: int,
        text_column_name: str = 'text',
) -> Dict[str, np.ndarray]:
    """
    Tokenizes batches of examples for pre-training a T5 model.

    :param examples: A batch in the form of a dictionary mapping, mapping column names to their respective values.
    :param tokenizer: A function which converts string tokens into input_ids and other model inputs.
    :param text_column_name: Name of the column within the input dictionary that contains the text which will be
        tokenized.
    :return: A dictionary containing the original mappings, as well as the mapping between model input names (e.g.,
        `input_ids`) and model input values (e.g., the tensor corresponding to the input IDs of the model).
    """
    batch_encoding = tokenizer(
        text=examples[text_column_name],
        max_length=in_length,
        padding='max_length',
        truncation=True,
    )
    input_ids = batch_encoding[TokenizerConstants.INPUT_IDS]
    result = {TokenizerConstants.INPUT_IDS: np.array(input_ids)}
    return result


def tokenizer_function_depth_pre_training(
        examples: typing.Dict[str, typing.List[str]],
        tokenizer: transformers.T5TokenizerFast,
        in_length: int,
        text_column_name: str = 'text',
) -> Dict[str, np.ndarray]:
    """
    Tokenizes batches of examples for pre-training a T5 model.

    :param examples: A batch in the form of a dictionary mapping, mapping column names to their respective values.
    :param tokenizer: A function which converts string tokens into input_ids and other model inputs.
    :param in_length: The maximum length of the input sequence.
    :param text_column_name: Name of the column within the input dictionary that contains the text which will be
            tokenized.
    :return: A dictionary containing the original mappings, as well as the mapping between model input names (e.g.,
            `input_ids`) and model input values (e.g., the tensor corresponding to the input IDs of the model).
    """
    batch_encoding = tokenizer(
        text=examples[text_column_name],
        max_length=in_length,
        padding='max_length',
        truncation='only_first',
    )
    result = {
        TokenizerConstants.INPUT_IDS: np.array(batch_encoding[TokenizerConstants.INPUT_IDS]),
        TokenizerConstants.ATTENTION_MASK: np.array(batch_encoding[TokenizerConstants.ATTENTION_MASK]),
        TokenizerConstants.TOKEN_TYPE_IDS: np.array(batch_encoding[TokenizerConstants.TOKEN_TYPE_IDS]),
    }
    return result


def tokenize_function_t5_fine_tuning(
        examples: typing.Dict[str, typing.Any],
        tokenizer: transformers.PreTrainedTokenizer,
        input_column_name: str = 'sentence',
        target_column_name: str = 'label',
        input_max_length: int = 512,
        target_max_length: int = 512,
) -> typing.Dict[str, torch.Tensor]:
    """
        Tokenizes batches of examples for an encoder-decoder model.

        :param examples: A batch in the form of a dictionary mapping, mapping column names to their respective values.
        :param tokenizer: A function which converts string tokens into input_ids and other model inputs.
        :param input_column_name: Name of the column within the input dictionary that contains the text which will be
            tokenized.
        :param target_column_name: Name of the column within the input dictionary that contains the labels which will be
            tokenized.
        :param input_max_length: The maximum length of the input sequence.
        :param target_max_length: The maximum length of the target sequence.

        Returns:
            A dictionary containing the original mappings, as well as the mapping between model input names (e.g.,
                `input_ids`) and model input values (e.g., the tensor corresponding to the input IDs of the model).
    """
    inputs = examples[input_column_name]
    encoding = tokenizer(
        inputs,
        padding='max_length',
        max_length=input_max_length,
        truncation=True,
        return_tensors="pt",
    )
    results = {
        TokenizerConstants.INPUT_IDS: encoding.input_ids,
        TokenizerConstants.ATTENTION_MASK: encoding.attention_mask,
    }

    # Labels are not preprocessed for the T5 model. model_inputs are returned as is
    outputs = examples[target_column_name]
    labels = tokenizer(
        outputs,
        padding='max_length',
        max_length=target_max_length,
        truncation=True,
        return_tensors="pt",
    )[TokenizerConstants.INPUT_IDS]

    # Replace the padding token with -100 to ignore it for loss computation
    labels[labels == tokenizer.pad_token_id] = -100
    results[TokenizerConstants.LABELS] = labels
    return results

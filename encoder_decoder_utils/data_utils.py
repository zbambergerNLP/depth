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
        examples,
        tokenizer,
        in_length) -> Dict[str, np.ndarray]:
    tokenizer_out = tokenizer(
        text=examples["text"],
        return_attention_mask=False,
    )

    input_ids = tokenizer_out["input_ids"]

    # TODO: Consider additional approaches to compressing the input_ids. For example, could try to dynamically
    #  concatenate as few examples as possible per row such that the total length of the input_ids is less than
    #  the in_length.
    concatenated_ids = np.concatenate(input_ids)

    total_length = concatenated_ids.shape[0]
    total_length = (total_length // in_length) * in_length

    concatenated_ids = concatenated_ids[:total_length]
    concatenated_ids.reshape(-1, in_length)
    result = {"input_ids": concatenated_ids}

    return result

def tokenizer_function_t5_pre_training(
        examples: typing.Dict[str, typing.List[str]],
        tokenizer: transformers.T5Tokenizer,
        in_length: int,
        text_column_name: str = 'text',
) -> Dict[str, np.ndarray]:
    """
    Tokenizes batches of examples for pre-training a T5 model.

    Args:
        examples: A batch in the form of a dictionary mapping, mapping column names to their respective values.
        tokenizer: A function which converts string tokens into input_ids and other model inputs.
        text_column_name: Name of the column within the input dictionary that contains the text which will be
            tokenized.

    Returns:
        A dictionary containing the original mappings, as well as the mapping between model input names (e.g.,
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

    Args:
        examples: A batch in the form of a dictionary mapping, mapping column names to their respective values.
        tokenizer: A function which converts string tokens into input_ids and other model inputs.
        text_column_name: Name of the column within the input dictionary that contains the text which will be
            tokenized.

    Returns:
        A dictionary containing the original mappings, as well as the mapping between model input names (e.g.,
            `input_ids`) and model input values (e.g., the tensor corresponding to the input IDs of the model).
    """
    batch_encoding = tokenizer(
        text=examples[text_column_name],
        max_length=in_length,
        padding='max_length',
        truncation='only_first',
    )
    return batch_encoding


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

        Args:
            examples: A batch in the form of a dictionary mapping, mapping column names to their respective values.
            tokenizer: A function which converts string tokens into input_ids and other model inputs.
            input_column_name: Name of the column within the input dictionary that contains the text which will be
                tokenized.
            target_column_name: Name of the column within the input dictionary that contains the labels which will be
                tokenized.
            input_max_length: The maximum length of the input sequence.
            target_max_length: The maximum length of the target sequence.

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
        results = {'input_ids': encoding.input_ids, 'attention_mask': encoding.attention_mask}

        # Labels are not preprocessed for the T5 model. model_inputs are returned as is
        outputs = examples[target_column_name]
        labels = tokenizer(
            outputs,
            padding='max_length',
            max_length=target_max_length,
            truncation=True,
            return_tensors="pt",
        )['input_ids']

        # Replace the padding token with -100 to ignore it for loss computation
        labels[labels == tokenizer.pad_token_id] = -100
        results['labels'] = labels
        return results
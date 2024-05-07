import copy
import typing

import numpy as np
from typing import (
    Dict,
    List,
    Union,
    Tuple,
    Set,
    Iterable,
)
import string
import re
import random
import transformers

from encoder_decoder_utils import tokenizer_utils
from encoder_decoder_utils import constants


# TODO: Replace magic strings with constants.

def _pad_or_truncate(
        sequence: List[Union[int, str]],
        length: int
) -> Iterable[Union[int, str]]:
    """Pad or truncate a list to the specified length

    :param sequence: A list of integers or strings.
    :param length: An integer representing the desired length of the list.
    :return: A list of integers or strings of length `length`.
    """
    if len(sequence) < length:
        return sequence + [0] * (length - len(sequence))
    else:
        return sequence[:length]


def _pad_or_truncate_np(
        sequence: np.ndarray,  # An integer tensor of shape [batch_size, sequence_length] or [sequence_length]
        length: int,
        pad_token: int,
) -> np.ndarray:
    """Pad or truncate a numpy array to the specified length

    :param sequence: A numpy array of integers.
    :param length: An integer representing the desired length of the list.
    :param pad_token: The ID of the padding token.
    :return: A numpy array of integers or strings of length `length`.
    """
    if len(sequence.shape) == 1:
        sequence_length = sequence.shape[0]
        if sequence_length < length:
            return np.pad(
                array=sequence,
                pad_width=(0, length - sequence_length),
                mode="constant",
                constant_values=pad_token,
            )
        else:
            return sequence[:length]
    elif len(sequence.shape) == 2:
        batch_size, sequence_length = sequence.shape
        if sequence_length < length:
            return np.pad(
                array=sequence,
                pad_width=np.array([[0, 0], [0, length - sequence_length]]),
                mode="constant",
                constant_values=pad_token,
            )
        else:
            return sequence[:, :length]
    else:
        raise ValueError(
            f"Expected a sequence of shape [batch_size, sequence_length] or [sequence_length], but got {sequence.shape}"
        )


def shift_tokens_right(
        input_ids: np.array,  # An integer tensor of shape [batch_size, sequence_length]
        pad_token_id: int,
        decoder_start_token_id: int,
) -> np.ndarray:
    """
    Shift input ids one token to the right.

    Taken from:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_flax_t5.py#L61

    Args:
        input_ids: An integer tensor of shape [batch_size, sequence_length].
        pad_token_id: The pad token id.
        decoder_start_token_id: The decoder start token id.

    Returns:
        An integer tensor of shape [batch_size, sequence_length].
    """
    shifted_input_ids = np.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = decoder_start_token_id

    shifted_input_ids = np.where(
        shifted_input_ids == constants.T5TokenizerConstants.PAD_TOKEN_ID,
        pad_token_id,
        shifted_input_ids,
    )
    return shifted_input_ids


# TODO: Vectorize this function.
def merge_segments(
        offset: int,
        n_grams: int,
        covered_indices: Set[int],
        whole_words: str,
        start_inds: List[int],
        segments_to_merge: List[List[int]],
        ngrams_vocab_set: Set[str],
):
    """
    Merge segments of length n_grams into a single segment if they are in the ngrams_vocab_set.

    NOTE: This function is meant for a new feature that masks spans of words in a sequence based on PMI.
        This function is not yet integrated into the main codebase.

    Args:
        offset: The offset of the current segment in the whole sequence.
        n_grams: The length of the segment to merge.
        covered_indices: A set of indices that were already merged.
        whole_words: The whole sequence.
        start_inds: The indices of the start of each word in the whole sequence.
        segments_to_merge: A list of segments to merge.
        ngrams_vocab_set: The set of ngrams to use for PMI-based corruption.
    """
    # TODO: Vectorize this function.
    possible_merges = []
    for i in range(len(start_inds) - n_grams):
        segment = whole_words[start_inds[i]:start_inds[i + n_grams] - 1]
        if segment in ngrams_vocab_set:
            possible_merges.append(list(range(offset + i, offset + i + n_grams)))

    random.shuffle(possible_merges)
    for seg_inds in possible_merges:
        if len(set(seg_inds).intersection(covered_indices)) > 0:
            continue
        covered_indices.update(seg_inds)
        segments_to_merge.append(seg_inds)


# TODO: Vectorize this function.
def pmi_word_mask(
        input_tokens: List[str],
        pmi_vocab: Set[str],
        max_predictions=512,
        mlm_probability=0.5,
) -> List[int]:
    """
    Create a mask for PMI-based corruption for a sample.

    Initially, we map which tokens are part of the same word, and then we mask the entire word.
    Next, we mask ngrams that are in the ngrams_vocab_set, from 5 to 2 grams, while avoiding overlapping ngrams.

    NOTE: This function is meant for a new feature that masks spans of words in a sequence based on PMI.
        This function is not yet integrated into the main codebase.

    Args:
        input_tokens: A tensor of tokens.
        pmi_vocab: The set of ngrams to use for PMI-based corruption.
        max_predictions: The maximum number of tokens to mask.
        mlm_probability: The probability of masking a token.

    Returns:
        A list of 0/1 in the length of the input tokens, 1 means the token should be masked.
    """
    # TODO: Vectorize this function.
    # TODO: Break this function into smaller functions.
    whole_words_indexes = []
    whole_words_lists = [[]]
    whole_words = whole_words_lists[0]
    for (i, token) in enumerate(input_tokens):
        if (
                token == constants.T5TokenizerConstants.START_TOKEN or
                token == constants.T5TokenizerConstants.PAD_TOKEN
        ):
            whole_words_lists.append(
                []  # to separate parts as we don't want them to be considered part of an ngram
            )
            whole_words = whole_words_lists[-1]
            continue
        # now, we mark the indices of token that start a whole word that can be masked
        if (
                len(whole_words_indexes) >= 1 and
                not token.startswith(constants.T5TokenizerConstants.SPACE_TOKEN) and
                token not in string.punctuation
        ):
            whole_words_indexes[-1].append(i)
            whole_words[-1] = whole_words[-1] + token.strip(constants.T5TokenizerConstants.SPACE_TOKEN).lower()
        else:
            whole_words_indexes.append([i])
            whole_words.append(token.strip(constants.T5TokenizerConstants.SPACE_TOKEN).lower())
    offset = 0
    covered_indices = set()
    segments_to_merge = []
    for whole_words in whole_words_lists:
        if len(whole_words) == 0:
            continue
        added_offset = len(whole_words)
        whole_words = ' '.join(whole_words)
        start_inds = [0] + [m.start() + 1 for m in re.finditer(' ', whole_words)] + [len(whole_words) + 1]
        for n_grams in range(5, 1, -1):
            merge_segments(
                offset,
                n_grams,
                covered_indices,
                whole_words,
                start_inds,
                segments_to_merge,
                pmi_vocab,
            )
        offset += added_offset
    segments_to_merge.extend([i] for i in set(range(len(whole_words_indexes))).difference(covered_indices))

    candidates = []
    for seg_to_merge in segments_to_merge:
        candidates.append(sum([whole_words_indexes[i] for i in seg_to_merge], []))

    # whole_words_indexes is a list of lists of ints, each list is the
    # indices part of the whole word to be masked, i.e. the segment to be considered for masking
    random.shuffle(candidates)
    candidates = sorted(candidates, reverse=True, key=lambda x: len(x))
    num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * mlm_probability))))
    masked_lms = []

    # aux list that index every token to the parent segment index
    # to make sure later the entire segment is masked correctly
    indexer = list(range(len(input_tokens)))
    covered_indexes = set()

    # list of 0/1 in the length of the input tokens, 1 means the token should be masked
    mask_labels = [0] * len(input_tokens)
    for index_set in candidates:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip these candidates.
        if len(covered_indexes) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:  # not sure how is it possible, the sets should be disjoint
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        head_index = min(index_set)
        mask_labels[head_index] = 1
        for index in index_set:
            covered_indexes.add(index)
            indexer[index] = head_index
            mask_labels[index] = 1

    return mask_labels


# TODO: Vectorize this function.
def pmi_noise_mask(
        examples: transformers.BatchEncoding,
        pmi_vocab: Set[str],
        tokenizer: typing.Union[transformers.T5Tokenizer, tokenizer_utils.DepthTokenizer]
) -> np.ndarray:  # Size: [batch_size, input_length]
    """
    Create a mask for PMI-based corruption in encoder-decoder models (e.g., T5).

    NOTE: This function is meant for a new feature that masks spans of words in a sequence based on PMI.
        This function is not yet integrated into the main codebase.

    Args:
        examples: A BatchEncoding containing the input sequences, expected shape [batch_size, input_length].
        pmi_vocab: The set of ngrams to use for PMI-based corruption.
        tokenizer: The tokenizer to use for PMI-based corruption.

    Returns:
        A mask array (0/1) of shape [batch_size, input_length] where 1 means the token should be masked.
    """
    mask_labels = []
    for example in examples[constants.T5TokenizerConstants.INPUT_IDS]:
        ref_tokens = [tokenizer._convert_id_to_token(int(input_id)) for input_id in example]
        mask_labels_for_sample = pmi_word_mask(ref_tokens, pmi_vocab)
        mask_labels.append(mask_labels_for_sample)

    return np.array(mask_labels)


def random_spans_noise_mask(
        sequence_length: int,
        maximum_length: int,
        noise_density: float,
        mean_noise_span_length: float = 3.0,
        random_roll: bool = True):
    """Initialize spans to mask tokens from input text as part of pre-training.

    Noise mask consisting of random spans of noise tokens. The number of noise tokens and the number of noise spans
    and non-noise spans are determined deterministically as follows:
        - num_noise_tokens = round(sequence_length * noise_density)
        - num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
    Spans alternate between non-noise and noise, beginning with non-noise. Subject to the above restrictions, all masks
    are equally likely.

    Adopted from:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/2ce1574a0c2f5ed65a08e87cc38ad8ceb222b239/t5/data/preprocessors.py#L2895

    Args:
        sequence_length: an int32 scalar (length of the incoming token sequence)
        maximum_length: an int32 scalar (length of the resulting padded or truncated mask).
        noise_density: a float - approximate density of output mask
        mean_noise_span_length: a number
        random_roll: bool, whether to roll the mask by a random integer offset in [0, sequence_length). Set random_roll
            to True to get a more uniform distribution of masked positions. Specifically, when random_roll is False
            (default) and a single span is enough to satisfy the noise density requirement, this fuction masks only the
            last few positions.

    Returns:
        a boolean tensor with shape [sequence_length] denoting the location of masked spans. True denotes a mask on the
        corresponding token while False denotes that the corresponding token is unmasked.
    """
    if noise_density == 0.0:
        return np.zeros(sequence_length, np.bool_)

    orig_length = sequence_length

    # increase length to avoid degeneracy
    sequence_length = np.maximum(sequence_length, 2)

    def to_int(x: np.ndarray):
        return x.astype(np.int32)

    def to_float(x: np.ndarray):
        return x.astype(np.float32)

    # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
    num_noise_tokens = to_int(np.round(to_float(sequence_length) * noise_density))
    num_noise_tokens = np.minimum(np.maximum(num_noise_tokens, 1), sequence_length - 1)
    num_noise_spans = to_int(
        np.round(to_float(num_noise_tokens) / mean_noise_span_length))

    # avoid degeneracy by ensuring positive number of noise spans
    num_noise_spans = np.maximum(num_noise_spans, 1)
    num_nonnoise_tokens = sequence_length - num_noise_tokens

    # pick the lengths of the noise spans and the non-noise spans
    def _random_segmentation(
            num_items: int,
            num_segments: int,
    ) -> np.ndarray:
        """Partition a sequence of items randomly into non-empty segments.

        Precondition: in order to ensure determinism, set the numpy seed before calling this function.

        Args:
            num_items: an integer scalar > 0
            num_segments: an integer scalar in [1, num_items]

        Returns:
            A tensor with shape [num_segments] containing positive integers that add up to num_items.
        """
        first_in_segment = to_int(np.less(np.arange(num_items - 1), num_segments - 1))
        np.random.shuffle(first_in_segment)
        first_in_segment = np.pad(first_in_segment, np.array([[1, 0]]))
        segment_id = np.cumsum(first_in_segment)
        _, segment_length = np.unique(segment_id, return_counts=True)
        return segment_length

    noise_span_lengths = _random_segmentation(
        num_noise_tokens,
        num_noise_spans,
    )
    nonnoise_span_lengths = _random_segmentation(
        num_nonnoise_tokens,
        num_noise_spans,
    )

    # Identify the indices of the beginning of masked spans.
    interleaved_span_lengths = np.reshape(
        np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
        [num_noise_spans * 2])
    span_starts = np.cumsum(interleaved_span_lengths)[:-1]

    span_start_indicator = np.zeros([sequence_length], dtype=np.int32)
    np.put_along_axis(span_start_indicator, span_starts, values=1, axis=0)
    span_num = np.cumsum(span_start_indicator)
    is_noise = np.equal(span_num % 2, 1)
    mask = is_noise[:orig_length]

    if random_roll:
        offset = np.random.uniform(low=0, high=sequence_length, size=[1]).astype(np.int32)
        mask = np.roll(mask, shift=offset, axis=0)

    # Pad to a consistent length
    if sequence_length < maximum_length:
        num_values_to_add = maximum_length - sequence_length
        mask = np.concatenate(
            [mask, np.zeros([num_values_to_add], dtype=bool)],
            axis=0,
        )

    return mask


def filter_input_ids_for_t5(
        vocab_size: int,
        input_ids: np.ndarray,  # An integer tensor of shape [batch_size, input_length]
        sentinel_ids: np.ndarray,  # An integer tensor of shape [batch_size, input_length]
        token_type_ids: np.ndarray = None,  # An integer tensor of shape [batch_size, input_length]
) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
    """
    Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.

    Args:
        vocab_size: An integer representing the size of the vocabulary.
        input_ids: An integer tensor of shape [batch_size, input_length].
        sentinel_ids: An integer tensor of shape [batch_size, input_length], where non-sentinels are 0s, sentinel
            continuations are -1, and sentinel starts are integers within the vocabulary's sentinel token IDs.
        token_type_ids: An integer tensor of shape [batch_size, input_length], where each sentence corresponds to a
            unique integer from 1 to k (where k is the number of sentences). Padding corresponds with type 0, and, if
            present, is at the end of an example.

    Returns:
        A tensor of shape [batch_size, input_length] in which sentinel continuations are removed.
    """
    large_num = vocab_size + 100
    input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)

    # Concatenate -1 to the end of each example's input IDs (i.e., along axis 1, where axis 0 is the batch dimension).
    expanded_input_ids_full = np.concatenate(
        (input_ids_full,
         -np.ones(input_ids_full.shape[0], dtype=input_ids_full.dtype).reshape(-1, 1),
         ),
        axis=1,
    )

    # Create a tensor of indices using broadcasting.
    indices = np.zeros(expanded_input_ids_full.shape, dtype=np.int64)
    indices += np.arange(expanded_input_ids_full.shape[1])

    # Replace -1s within input IDs (span continuations) with a large number that is out of vocabulary.
    indices[expanded_input_ids_full == -1] = large_num

    # Ensure that input IDs that were span continuations appear at the end of the sequence
    gather_indices = np.sort(indices, axis=1)

    # Ensure that all indices of span continuations (now at the end of 'gather_indices') correspond to the last index
    gather_indices[gather_indices == large_num] = expanded_input_ids_full.shape[1] - 1
    gather_indices = gather_indices.astype(np.int8)

    # Shift all -1s within input IDs to the end of the sequence, and then replace them with 0s.
    modified_input_ids = np.take_along_axis(
        arr=expanded_input_ids_full,
        indices=gather_indices[:, :-1],
        axis=1,
    )
    modified_input_ids = np.where(
        modified_input_ids == -1,
        0,
        modified_input_ids,
    )

    # If token type IDs are not provided, then we are done. Otherwise, we need to filter the token type IDs as well.
    if token_type_ids is None:
        return modified_input_ids, None

    expanded_token_type_ids = np.concatenate(
        (
            token_type_ids,
            -np.ones(token_type_ids.shape[0], dtype=token_type_ids.dtype).reshape(-1, 1)
        ),
        axis=1,
    )
    modified_token_type_ids = np.take_along_axis(expanded_token_type_ids, gather_indices[:, :-1], axis=1)
    modified_token_type_ids = np.where(modified_token_type_ids == -1, 0, modified_token_type_ids)
    return modified_input_ids, modified_token_type_ids


def filter_target_ids_for_t5(
        input_ids: np.ndarray,
        input_ids_sentinel: np.ndarray,
        vocab_size: int,
) -> np.ndarray:
    """
    Filter the target IDs for T5.

    Args:
        input_ids: An integer tensor of shape [batch_size, input_length].
        input_ids_sentinel: An integer tensor of shape [batch_size, input_length], where non-sentinels are 0s, sentinel
            continuations are -1, and sentinel starts are integers within the vocabulary's sentinel token IDs.
        vocab_size: An integer representing the size of the vocabulary.

    Returns:
        A tensor of shape [batch_size, input_length] in which sentinel continuations are removed.
    """
    # TODO: Support target_length != input_length
    # Shift the input IDs to the right by one by concatenating a 0 (pad) to the beginning of each example's input IDs.
    shifted_input_ids = np.concatenate(
        (
            np.zeros((input_ids.shape[0], 1), dtype=input_ids.dtype),
            input_ids[:, :-1],
        ),
        axis=1,
    )

    # Shift the input IDs to the right by one by concatenating a 0 (pad) to the beginning of each example's input IDs.
    shifted_input_ids_sentinel = np.concatenate(
        (
            np.zeros((input_ids_sentinel.shape[0], 1), dtype=input_ids_sentinel.dtype),
            input_ids_sentinel[:, :-1],
        ),
        axis=1,
    )

    result = copy.deepcopy(input_ids_sentinel)

    # Replace the input IDs with the shifted input IDs where the sentinel IDs are not 0.
    result = np.where(
        shifted_input_ids_sentinel != 0,
        shifted_input_ids,
        result,
    )

    large_num = vocab_size + 100

    # Concatenate 0 to the end of each example's input IDs (i.e., along axis 1, where axis 0 is the batch dimension).
    expanded_result = np.concatenate(
        (
            result,
            np.zeros(result.shape[0], dtype=result.dtype).reshape(-1, 1),
        ),
        axis=1,
    )

    # Create a tensor of indices using broadcasting.
    indices = np.zeros(expanded_result.shape, dtype=np.int64)
    indices += np.arange(expanded_result.shape[1])

    # Replace 0s within the result with a large number that is out of vocabulary.
    indices[expanded_result == 0] = large_num

    # Ensure that padding tokens within our result are moved to the end.
    gather_indices = np.sort(indices, axis=1)

    # Ensure that all indices of 0's (now at the end of 'gather_indices') correspond to the last index
    # of a sequence in input IDs (since those values must correspond to 0)
    gather_indices[gather_indices == large_num] = expanded_result.shape[1] - 1
    gather_indices = gather_indices.astype(np.int8)

    # Shift all -1s within input IDs to the end of the sequence, and then replace them with 0s.
    modified_result = np.take_along_axis(expanded_result, gather_indices[:, :-1], axis=1)
    return modified_result


def create_sentinel_ids_for_t5(
        mask_indices: np.ndarray,  # An integer tensor of shape [batch_size, input_length]
        vocab_size: int,
) -> np.ndarray:
    """
    Create sentinel ids given the indices that should be masked.

    The start indices of each mask are replaced by the sentinel ids in increasing
    order. Consecutive mask indices to be deleted are replaced with `-1`.

    Args:
        mask_indices: The tensor containing mask indices.
        vocab_size: The size of the vocabulary that is used by both the model and tokenizer.

    Returns:
        A tensor of the same size as 'mask_indices', where the beginning of masked spans are denoted by the
        ID of some sentinel token, continuations of spans are denoted with an ID of -1, and non-masked tokens are
        denoted with an ID of 0.
    """
    # Create a tensor of the same size as `mask_indices` where the first element of each mask is replaced with a '1',
    # and the rest of the elements of each mask are replaced with a '0'.
    start_indices = mask_indices - np.roll(
        a=mask_indices,
        shift=1,
        axis=-1,
    ) * mask_indices
    start_indices[:, 0] = mask_indices[:, 0]

    # Create a tensor of the same size as `mask_indices` where the first element of each mask is replaced with a '1',
    # the remaining elements of each mask are replaced with a '-1', and non-masked tokens are replaced with a '0'.
    sentinel_ids = np.where(
        start_indices != 0,
        np.cumsum(start_indices, axis=-1),
        start_indices,
    )

    # Replace the tokens at the beginning of masked spans with the sentinel ids in decreasing order.
    sentinel_ids = np.where(
        sentinel_ids != 0,
        vocab_size - sentinel_ids,
        0)
    sentinel_ids -= mask_indices - start_indices
    return sentinel_ids


def corrupt_for_vanilla_t5(
        examples: Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]], transformers.BatchEncoding],
        vocab_size: int,
        input_length: int,
        target_length: int,
        pad_token_id: int,
        eos_token_id: int,
        decoder_start_token_id: int,
        noise_density: float = 0.5,
        pmi: bool = False,
        ngram_vocab_set: Set[str] = None,
        tokenizer=None,
) -> Dict[str, np.ndarray]:
    """Apply corruption to the input examples for T5, create targets, prepare all model inputs.

    Args:
        examples: A list of dictionaries containing the input and target sequences.
        vocab_size: The size of the vocabulary that is used by both the model and tokenizer.
        input_length: The length of the input sequence.
        target_length: The length of the target sequence.
        pad_token_id: The ID of the padding token.
        eos_token_id: The ID of the end of sentence token.
        decoder_start_token_id: The ID of the decoder start token.
        noise_density: The density of the noise to be applied to the input sequence.
        pmi: Whether to use PMI-based corruption.
        ngram_vocab_set: The set of ngrams to use for PMI-based corruption.
        tokenizer: The tokenizer to use for PMI-based corruption.
    Returns:
        A dictionary containing the input and target sequences, as well as the model inputs.
    """
    # convert list to dict and tensorize input
    if isinstance(examples, list) and isinstance(examples[0], dict):
        batch = {
            k: np.array(
                [examples[i][k] for i in range(len(examples))]
            ) for k, v in examples[0].items()
        }
    else:
        batch = examples

    input_ids = batch[constants.T5TokenizerConstants.INPUT_IDS]
    batch_size, expandend_input_length = input_ids.shape
    if pmi:
        mask_indices = pmi_noise_mask(examples, ngram_vocab_set, tokenizer)
    else:
        mask_indices = np.stack(
            [
                random_spans_noise_mask(
                    sequence_length=expandend_input_length,
                    maximum_length=expandend_input_length,
                    noise_density=noise_density,
                ) for _ in range(batch_size)
            ],
        )

    is_special_token = np.isin(
        element=input_ids,
        test_elements=np.array([pad_token_id, eos_token_id]),
    )
    mask_indices = np.where(is_special_token, False, mask_indices)

    input_ids_sentinel = create_sentinel_ids_for_t5(
        vocab_size=vocab_size,
        mask_indices=mask_indices.astype(np.int8),
    )

    batch[constants.T5TokenizerConstants.INPUT_IDS] = filter_input_ids_for_t5(
        vocab_size=vocab_size,
        input_ids=input_ids,
        sentinel_ids=input_ids_sentinel,
    )[0]
    batch[constants.T5TokenizerConstants.INPUT_IDS] = _pad_or_truncate_np(
        sequence=batch[constants.T5TokenizerConstants.INPUT_IDS],
        length=input_length,
        pad_token=pad_token_id,
    )
    labels = filter_target_ids_for_t5(
        input_ids=input_ids,
        input_ids_sentinel=input_ids_sentinel,
        vocab_size=vocab_size,
    )
    labels = _pad_or_truncate_np(
        sequence=labels,
        length=target_length,
        pad_token=pad_token_id,
    )
    labels[labels == pad_token_id] = constants.T5TokenizerConstants.PAD_TOKEN_ID
    batch[constants.T5TokenizerConstants.LABELS] = labels
    if batch[constants.T5TokenizerConstants.INPUT_IDS].shape[-1] != input_length:
        raise ValueError(
            "`input_ids` are incorrectly preprocessed. `input_ids` length is "
            f"{batch[constants.T5TokenizerConstants.INPUT_IDS].shape[-1]}, but should be {input_length}."
        )
    if batch[constants.T5TokenizerConstants.LABELS].shape[-1] != target_length:
        raise ValueError(
            "`labels` are incorrectly preprocessed. "
            f"`labels` length is {batch[constants.T5TokenizerConstants.LABELS].shape[-1]},"
            f" but should be {target_length}."
        )
    # to check that tokens are correctly preprocessed, one can run `self.tokenizer.batch_decode(input_ids)` and
    # `self.tokenizer.batch_decode(labels)` here
    batch[constants.T5TokenizerConstants.DECODER_INPUT_IDS] = shift_tokens_right(
        input_ids=batch[constants.T5TokenizerConstants.LABELS],
        pad_token_id=pad_token_id,
        decoder_start_token_id=decoder_start_token_id,
    )
    return batch


# TODO: Vectorize this function
def create_depth_encoder_self_attention_mask(
        input_ids: np.ndarray,  # An integer tensor of shape [batch_size, input_length]
        input_token_type_ids,  # An integer tensor of shape [batch_size, input_length]
        tokenizer,
        sentence_token_ids: List[int],
) -> np.ndarray:  # A binary integer tensor of shape [batch_size, input_length, input_length]
    """
    Create a self attention mask for the encoder.

    The self attention mask abides by the following rules:
    * Sentence tokens (IDs corresponding to tokens of the form <sent_i>, where i is some sentence index): Can only
        attend to other tokens within the same sentence. This is enforced via the token_type_ids tensor,
        where entry [i, j] corresponds to the token type ID of the token at input_ids[i, j].
    * Non-special tokens (IDs that are not sentence tokens, padding tokens or EOS tokens): Have bidirectional attention
        across all tokens within the example (i.e., can attend to all tokens within the example) except for <pad>
        tokens.
    * Padding tokens (The ID that corresponds to the <pad> token>): Cannot attend to any other tokens within the
        example. Also, no other tokens can attend to padding tokens.

    :param input_ids: An integer tensor of shape [batch_size, input_length] corresponding to input token IDs.
    :param input_token_type_ids: An integer tensor of shape [batch_size, sequence_length] corresponding to sentences of
        each token within input_ids. The token at input_ids[i, j] belongs to sentence input_token_type_ids[i, j].
    :param tokenizer: The tokenizer used to generate token IDs. Contains dedicated tokens for sentences and sentinel
        tokens.
    :param sentence_token_ids: A list of integers corresponding to the IDs of sentence tokens. The IDs are expected to
        be in the vocabulary of the tokenizer, and each new sentence in the input is preceded by a sentence token.
    :return: A binary integer tensor of shape [batch_size, input_length, input_length] which serves as the attention
        mask for the encoder.
    """
    assert len(input_ids.shape) == 2, "The input IDs must be a 2D tensor."
    assert len(input_token_type_ids.shape) == 2, "The input token type IDs must be a 2D tensor."
    assert input_ids.shape == input_token_type_ids.shape, (
        "The input IDs and input token type IDs must have the same shape. "
        f"Input ids shape: {input_ids.shape}. Input token type IDs shape: {input_token_type_ids.shape}."
    )

    encoder_is_padding = np.equal(input_ids, 0)
    batch_size, input_sequence_length = input_ids.shape

    batch_encoder_self_attention_mask = []
    for example_index, example in enumerate(input_ids):
        for token_index, token in enumerate(example):
            # Dealing with a sentence token. Can only attend to other sentence tokens within the same sentence.
            if np.isin(token, np.array(sentence_token_ids, dtype=input_ids.dtype)):
                sentence_index = input_token_type_ids[example_index, token_index]
                attention_mask = np.equal(input_token_type_ids[example_index], sentence_index).astype(np.int64)

            # Dealing with a non-special token. Can attend to all tokens within the example
            else:
                attention_mask = np.ones([input_sequence_length], dtype=np.int64)
            batch_encoder_self_attention_mask.append(attention_mask)
    batch_encoder_self_attention_mask = np.stack(batch_encoder_self_attention_mask, axis=0).reshape(
        [batch_size, input_sequence_length, input_sequence_length])
    return batch_encoder_self_attention_mask


# TODO: Vectorize this function
def create_depth_cross_attention_mask(
        input_ids: np.ndarray,  # An integer tensor of shape [batch_size, input_length]
        target_ids: np.ndarray,  # An integer tensor of shape [batch_size, target_length]
        sentence_token_ids: List[int],
        tokenizer: tokenizer_utils.DepthTokenizer,
) -> np.ndarray:  # A binary integer tensor of shape [batch_size, target_length, input_length]
    """
    Create a cross attention mask for the decoder.

    The cross attention mask abides by the following rules:
    * Sentence tokens (in the decoder): Can only attend to sentence tokens in the encoder.
    * Non-special tokens (in the decoder): Have autoregressive attention. I.e., they are able to attend to all
        previously predicted tokens in the encoder.
    * Padding tokens (in the decoder): Cannot attend to any other tokens in the encoder. Also, no other tokens in the
        encoder can attend to padding tokens.

    :param input_ids: An integer tensor of shape [batch_size, input_length] corresponding to input token IDs. This
        tensor corresponds to the encoder's input.
    :param target_ids: An integer tensor of shape [batch_size, target_length] corresponding to target token IDs. This
        tensor corresponds to the decoder's input.
    :param sentence_token_ids: A list of integers corresponding to the IDs of sentence tokens. The IDs are expected to
        be in the vocabulary of the tokenizer, and each new sentence in the input is preceded by a sentence token.
    :param tokenizer: The tokenizer used to generate token IDs. Contains dedicated tokens for sentences and sentinel
        tokens. The tokenizer is used to determine the IDs of padding tokens and EOS tokens.
    :return: A binary integer tensor of shape [batch_size, target_length, input_length] which serves as the attention
        mask for cross attention (from the decoder to the encoder). For example, entry [x, y, z] specifies whether
        the y'th token in the decoder can attend to the z'th token in the encoder for the x'th example.
    """
    assert input_ids.shape[0] == target_ids.shape[0], "The batch size of the input and target IDs must be the same."
    assert len(input_ids.shape) == 2, "The input IDs must be a 2D tensor."
    assert len(target_ids.shape) == 2, "The target IDs must be a 2D tensor."

    batch_size, input_sequence_length = input_ids.shape
    _, target_sequence_length = target_ids.shape

    batch_cross_attention_mask = []
    for example_index, example in enumerate(target_ids):
        for token_index, token in enumerate(example):
            if np.isin(token, np.array(sentence_token_ids, dtype=target_ids.dtype)):
                # Bidirectional attention mask over the input
                attention_mask = np.ones([input_sequence_length], dtype=np.int64)
                # Mask out all tokens in the encoder which are not sentence tokens.
                attention_mask = np.where(
                    np.isin(input_ids[example_index], np.array(sentence_token_ids)),
                    attention_mask,
                    0).astype(np.int64)

            else:
                # Non-special token, should attend to all tokens in the encoder.
                attention_mask = np.ones([input_sequence_length], dtype=np.int64)
            batch_cross_attention_mask.append(attention_mask)
    batch_cross_attention_mask = np.stack(batch_cross_attention_mask, axis=0).reshape(
        [batch_size, target_sequence_length, input_sequence_length])
    return batch_cross_attention_mask


# TODO: Vectorize this function
def create_depth_decoder_self_attention_mask(
        target_ids: np.ndarray,  # An integer tensor of shape [batch_size, target_length]
        target_token_type_ids: np.ndarray,  # An integer tensor of shape [batch_size, target_length]
        sentence_token_ids: List[int],
        tokenizer: tokenizer_utils.DepthTokenizer,
) -> np.ndarray:  # A binary integer tensor of shape [batch_size, target_length, target_length]
    """
    Create a self attention mask for the decoder.

    The self decoder attention mask abides by the following rules:
    * Sentence tokens (in the decoder): Can only attend to sentence tokens that were previously predicted in the
        decoder.
    * Non-special tokens (in the decoder): Have autoregressive attention. I.e., they are able to attend to all
        previously predicted tokens in the decoder.
    * Padding tokens (in the decoder): Cannot attend to any other tokens in the decoder. Also, no other tokens in the
        decoder can attend to padding tokens.

    :param target_ids: An integer tensor of shape [batch_size, target_length] corresponding to target token IDs. This
        tensor corresponds to the decoder's input.
    :param target_token_type_ids: An integer tensor of shape [batch_size, target_length] corresponding to sentences of
        each token within target_ids. The token at target_ids[i, j] belongs to sentence target_token_type_ids[i, j].
    :param sentence_token_ids: A list of integers corresponding to the IDs of sentence tokens. The IDs are expected to
        be in the vocabulary of the tokenizer, and each new sentence in the input is preceded by a sentence token.
    :param tokenizer: The tokenizer used to generate token IDs. Contains dedicated tokens for sentences and sentinel
        tokens. The tokenizer is used to determine the IDs of padding tokens and EOS tokens.
    :return: A binary integer tensor of shape [batch_size, target_length, target_length] which serves as the attention
        mask for self attention (within the decoder). I.e., entry [x, y, z] specifies whether the y'th token in
        the decoder can attend to the z'th token in the decoder for the x'th example.
    """
    assert len(target_ids.shape) == 2, "The target IDs must be a 2D tensor."
    assert len(target_token_type_ids.shape) == 2, "The target token type IDs must be a 2D tensor."
    assert target_ids.shape == target_token_type_ids.shape, (
        "The target IDs and target token type IDs must have the same shape. "
        f"Target ids shape: {target_ids.shape}. Target token type IDs shape: {target_token_type_ids.shape}."
    )

    batch_size, target_sequence_length = target_ids.shape

    batch_decoder_self_attention_mask = []
    for example_index, example in enumerate(target_ids):
        for token_index, token in enumerate(example):
            if np.isin(token, np.array(sentence_token_ids, dtype=target_ids.dtype)):
                # Create an auto-regressive mask
                attention_mask = np.ones([target_sequence_length], dtype=np.int64)
                attention_mask[token_index + 1:] = 0
                # Mask out all past tokens in the decoder which are not sentence tokens.
                attention_mask = np.where(
                    np.isin(target_ids[example_index], np.array(sentence_token_ids)),
                    attention_mask,
                    0).astype(np.int64)

            else:
                # Non-special token, should attend to all past tokens in the decoder.
                attention_mask = np.ones([target_sequence_length], dtype=np.int64)
                attention_mask[token_index + 1:] = 0
            batch_decoder_self_attention_mask.append(attention_mask)

    batch_decoder_self_attention_mask = np.stack(batch_decoder_self_attention_mask, axis=0).reshape(
        [batch_size, target_sequence_length, target_sequence_length])
    return batch_decoder_self_attention_mask


def create_depth_attention_masks(
        input_ids: np.ndarray,  # An integer tensor of shape [batch_size, input_length]
        target_ids: np.ndarray,  # An integer tensor of shape [batch_size, target_length]
        input_token_type_ids: np.ndarray,  # An integer tensor of shape [batch_size, input_length]
        target_token_type_ids: np.ndarray,  # An integer tensor of shape [batch_size, target_length]
        tokenizer: tokenizer_utils.DepthTokenizer,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create an attention mask given input and target token IDs.

    Implements a hierarchical attention mask which decomposes into three components:
    1. A hierarchical self attention mask for the encoder
    2. A hierarchical self attention mask for the decoder
    3. A cross attention mask for the decoder (attending to tokens in the encoder).

    The hierarchical self attention mask in the encoder bides by the following rules:
    * Sentence tokens (IDs corresponding to tokens of the form <sent_i>, where i is some sentence index): Can only
        attend to other sentence tokens within the same sentence. This is enforced via the token_type_ids tensor,
        where entry [i, j] corresponds to the token type ID of the token at input_ids[i, j].
    * Non-special tokens (IDs that are not sentence tokens, padding tokens or EOS tokens): Have bidirectional attention
        across all tokens within the example (i.e., can attend to all tokens within the example) except for <pad>
        tokens.
    * Padding tokens (The ID that corresponds to the <pad> token>): Cannot attend to any other tokens within the
        example. Also, no other tokens can attend to padding tokens.

    The hierarchical self attention mask in the decoder abides by the following rules:
    * Sentence tokens: Can only attend to sentence tokens that were previously predicted in the decoder.
    * Non-special tokens: Have autoregressive attention. I.e., they are able to attend to all previously predicted
        tokens in the decoder.

    The cross attention mask abides by the following rules:
    * Sentence tokens: Can only attend to sentence tokens in the encoder.
    * Non-special tokens: Have autoregressive attention. I.e., they are able to attend to all previously predicted
        tokens in the encoder.

    :precondition: The input and target token IDs have the same (padded) sequence length.

    :param input_ids: An integer tensor of shape [batch_size, input_sequence_length] corresponding to input token IDs.
    :param target_ids: An integer tensor of shape [batch_size, target_sequence_length] corresponding to target token
        IDs.
    :param input_token_type_ids: An integer tensor of shape [batch_size, sequence_length] corresponding to sentences of
        each token within input_ids. The token at input_ids[i, j] belongs to sentence input_token_type_ids[i, j].
        Padding tokens have a token type ID of 0.
    :param target_token_type_ids: An integer tensor of shape [batch_size, sequence_length] corresponding to sentences of
        each token within target_ids. The token at target_ids[i, j] belongs to sentence target_token_type_ids[i, j].
        Padding tokens have a token type ID of 0.
    :param tokenizer: The tokenizer used to generate token IDs. Contains dedicated tokens for sentences and sentinel
        tokens.
    :return: A 3-tuple of binary integer tensors of shape [batch_size, sequence_length, sequence_length] which serves
        as the attention masks for a hierarchical model.
        The tuple consists of:
        1. batch_encoder_self_attention_mask
        2. batch_cross_attention_mask
        3. batch_decoder_self_attention_mask
    """
    if len(input_ids.shape) == 1:
        input_ids = np.expand_dims(input_ids, axis=0)

    _, target_sequence_length = target_ids.shape

    # create the encoder's self attention mask
    batch_encoder_self_attention_mask = create_depth_encoder_self_attention_mask(
        input_ids=input_ids,
        input_token_type_ids=input_token_type_ids,
        tokenizer=tokenizer,
        sentence_token_ids=tokenizer.get_sentence_token_ids(),
    )

    # create the cross-attention mask from decoder to encoder
    batch_cross_attention_mask = create_depth_cross_attention_mask(
        input_ids=input_ids,
        target_ids=target_ids,
        sentence_token_ids=tokenizer.get_sentence_token_ids(),
        tokenizer=tokenizer,
    )

    # Create the decoder's self attention mask
    batch_decoder_self_attention_mask = create_depth_decoder_self_attention_mask(
        target_ids=target_ids,
        target_token_type_ids=target_token_type_ids,
        sentence_token_ids=tokenizer.get_sentence_token_ids(),
        tokenizer=tokenizer,
    )

    return batch_encoder_self_attention_mask, batch_cross_attention_mask, batch_decoder_self_attention_mask


def create_sentinel_ids_for_depth(
        tokenizer: tokenizer_utils.DepthTokenizer,
        mask_indices: np.ndarray,
        # An integer tensor of shape [batch_size, sequence_length] or shape [sequence_length].
        random_sentinel_order: bool = True,
) -> np.ndarray:  # Integer array of shape [batch_size, sequence_length]
    """
    Sentinel ids creation given the indices that should be masked.
    The start indices of each mask are replaced by the sentinel ids in increasing
    order. Consecutive mask indices to be deleted are replaced with `-1`.

    Adapted from:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/2ce1574a0c2f5ed65a08e87cc38ad8ceb222b239/t5/data/preprocessors.py#L2895

    :param tokenizer: A pre-trained transformers tokenizer with designed sentinel tokens.
    :param mask_indices: A binary tensor where 0 indicates the absence of a mask, integers > 0 represent
        the beginnings of new mask spans, and -1 indicates the continuation of a mask span.
    :param random_sentinel_order: If True, the order of the sentinel tokens is randomized. If False, then sentinels are
        assigned in decreasing order within each example.
    :return: A tensor with the same shape as 'mask_indices', where masked span starts are replaced by (unique) sentinel
        ids, continuations of those spans are replaced by -1, and non-masked tokens have value 0.
    """
    if len(mask_indices.shape) == 1:
        mask_indices = np.expand_dims(mask_indices, axis=0)

    batch_size, sequence_length = mask_indices.shape

    # Create a tensor of shape [batch_size, sequence_length] where 1s represent the beginning of masked spans, and 0s
    # represent everything else.
    start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
    start_indices[:, 0] = mask_indices[:, 0]

    # Initialize a list of possible sentinel tokens.
    special_tokens = tokenizer.special_tokens_map['additional_special_tokens']
    sentinel_tokens = list(filter(lambda token: '<extra_id_' in token, special_tokens))
    possible_sentinel_ids = tokenizer.convert_tokens_to_ids(sentinel_tokens)

    if random_sentinel_order:
        # Randomize the order of the sentinel ids.
        sentinel_ids = np.where(
            start_indices,
            np.random.choice(possible_sentinel_ids, size=(batch_size, sequence_length)),
            start_indices
        )
    else:
        last_sentinel_id = possible_sentinel_ids[-1]
        # Enumerate the start indices of each mask span (i.e., the first span starts at index 1, the second at index 2,
        # etc...)
        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)

        # Replace the sentinel ids with the last possible sentinel id minus the enumerated id. This ensures that the
        # sentinel ids are in decreasing order.
        sentinel_ids = np.where(sentinel_ids != 0, (last_sentinel_id - (sentinel_ids - 1)), 0)

    # Introduce span continuations to the result tensor. These are represented as -1s.
    sentinel_ids -= mask_indices - start_indices

    return sentinel_ids


# TODO: Vectorize this function
def create_model_input_for_corrupted_batch(
        input_ids: np.array,
        input_ids_sentinel: np.array,
        token_type_ids: np.array,
        batch_size: int,
        padded_sequence_length: int,
        sequence_lengths: np.array,
        sentence_token_ids: List[int],
) -> Tuple[np.array, np.array, np.array, np.array]:
    """

    :param input_ids: An integer tensor of shape [batch_size, padded_sequence_length] containing the token ids.
    :param input_ids_sentinel: An integer tensor of shape [batch_size, padded_sequence_length] containing the sentinel
        tokens. 0's represent the absence of a sequence token. A positive integer indicates the start of a masked span.
        Finally, -1's represent the continuation of a span.
    :param token_type_ids: An integer tensor of shape [batch_size, padded_sequence_length] containing the token type
        ids.
    :param batch_size: An integer representing the batch size.
    :param sequence_lengths: A tensor of shape [batch_size] containing the lengths of each sequence in the batch
        (not including padding tokens).
    :param padded_sequence_length: An integer representing the length of the padded sequence.
    :param sentence_token_ids: A list of integers representing the token ids of sentence tokens.
    :return: A 4-tuple of integer tensors containing:
            1. the modified input ids
            2. input token type ids
            3. label ids
            4. label token type ids
    """
    modified_input_ids = []
    modified_input_token_type_ids = []
    modified_label_ids = []
    modified_label_token_type_ids = []
    for example_index in range(batch_size):
        example_input_ids = []
        example_input_token_type_ids = []
        example_label_ids = []
        example_label_token_type_ids = []

        # Iterate over non-padding tokens in each sequence
        for position_index in range(sequence_lengths[example_index]):
            sentinel_token = input_ids_sentinel[example_index, position_index]
            input_token = input_ids[example_index, position_index]
            input_token_type_ids = token_type_ids[example_index, position_index]
            if sentinel_token == 0:
                example_input_ids.append(input_token)
                example_input_token_type_ids.append(input_token_type_ids)

                # Sentence tokens are part of the label as well.
                if input_token in sentence_token_ids:
                    example_label_ids.append(input_token)
                    example_label_token_type_ids.append(input_token_type_ids)

            elif sentinel_token == -1:
                example_label_ids.append(input_token)
                example_label_token_type_ids.append(input_token_type_ids)

            else:  # Sentinel token.
                example_input_ids.append(sentinel_token)
                example_input_token_type_ids.append(input_token_type_ids)
                example_label_ids.append(sentinel_token)
                example_label_ids.append(input_token)
                # Represent the token types for both the sentinel and the input token.
                example_label_token_type_ids.extend([input_token_type_ids] * 2)

        modified_input_ids.append(
            _pad_or_truncate(
                example_input_ids,
                padded_sequence_length,
            )
        )
        modified_input_token_type_ids.append(
            _pad_or_truncate(
                example_input_token_type_ids,
                padded_sequence_length,
            )
        )
        modified_label_ids.append(
            _pad_or_truncate(
                example_label_ids,
                padded_sequence_length,
            )
        )
        modified_label_token_type_ids.append(
            _pad_or_truncate(
                example_label_token_type_ids,
                padded_sequence_length,
            )
        )
    return (
        modified_input_ids,
        modified_input_token_type_ids,
        modified_label_ids,
        modified_label_token_type_ids,
    )


def shuffle_inputs(
        sentence_unique_ids: np.ndarray,
        sentence_start_indices: np.ndarray,
        sentence_lengths: np.ndarray,
        padding_token_id: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Shuffle sentences within inputs.

    :param sentence_unique_ids: An integer tensor of shape [num_sentences] containing the unique token type IDs
        corresponding to each sentence. Note that num_sentences corresponds with the number of sentences included
        within a single input example.
    :param sentence_start_indices: An integer tensor of shape [num_sentences] containing the start indices of each
        sentence within the input. Padding tokens have a token type ID of 0.
    :param sentence_lengths: An integer tensor of shape [num_sentences] containing the lengths of each sentence.
    :param padding_token_id: The token ID of the padding token of a vocabulary (accessible via the tokenizer).
    :return: A 4-tuple consisting of:
        - The shuffled order of sentences (e.g., [1, 2, 3] -> [2, 1, 3])
        - The lengths of the shuffled sentences. (e.g., [2, 1, 3] -> [5, 10, 15] if sentence 2 is of length 5, sentence
            1 is of length 10, and sentence 3 is of length 15)  .
        - The start indices of the shuffled sentences.
        - The shuffled token type IDs themselves.
    """
    # If inputs are padded, then ignore padding (index 0)
    start_index = 1 if padding_token_id in sentence_unique_ids else 0
    shuffled_order = sentence_unique_ids[start_index:]
    np.random.shuffle(shuffled_order)
    if start_index == 1:
        shuffled_order = np.concatenate([shuffled_order, np.array([0])], axis=0)
        shuffled_unique_ids = np.concatenate(
            [sentence_unique_ids[start_index:], np.array([sentence_unique_ids[0]])])
    else:
        shuffled_order -= 1
        shuffled_unique_ids = sentence_unique_ids

    shuffled_lengths = np.asarray([sentence_lengths[i] for i in shuffled_order])
    shuffled_token_type_ids = np.concatenate(
        [
            np.tile(unique_id, reps=shuffled_lengths[id_index])
            for id_index, unique_id in enumerate(shuffled_unique_ids)
        ]
    )
    shuffled_start_indices = np.asarray([sentence_start_indices[i] for i in shuffled_order])
    shuffled_unique_indices = shuffled_order if start_index == 1 else shuffled_order + 1
    if start_index == 0:
        shuffled_token_type_ids += 1
    return shuffled_unique_indices, shuffled_lengths, shuffled_start_indices, shuffled_token_type_ids


# TODO: Further decompose into smaller functions
def corrupt_for_depth(
        examples: Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]], transformers.BatchEncoding],
        tokenizer: tokenizer_utils.DepthTokenizer,
        pad_token_id: int,
        decoder_start_token_id: int,
        noise_density: float = 0.5,
        mean_noise_span_length: float = 3.0,
        input_length: int = 512,
        target_length: int = 512,
        pmi: bool = False,
        ngram_vocab_set: Set[str] = None,
        do_shuffle: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Corrupt the input for a depth model.

    :param examples: A dictionary containing the tokenized textual examples. The dictionary should contain the following
        keys:
        - 'input_ids': An integer tensor of shape [batch_size, input_length] containing the token ids.
        - 'token_type_ids': An integer tensor of shape [batch_size, input_length] containing the token type ids.
    :param tokenizer: The tokenizer used to generate token IDs. Contains dedicated tokens for sentences and sentinel
        tokens.
    :param pad_token_id: The token ID of the padding token of a vocabulary (accessible via the tokenizer).
    :param decoder_start_token_id: The token ID of the start token of the decoder.
    :param noise_density: The density of the noise to be applied to the input. This is a value in the range [0, 1].
    :param mean_noise_span_length: The mean length of the noise spans to be applied to the input. This is an integer
        value that is in the range (0, max_sequence_length). Note that for T5-based models, max_sequence_length is
        typically 512.
    :param pmi: A boolean indicating whether to use pointwise mutual information (PMI) to corrupt the input.
    :param ngram_vocab_set: A set of strings representing the vocabulary of the model. This is used to corrupt the input
        using PMI.
    :param do_shuffle: A boolean indicating whether to shuffle the sentences within the input.
    :return: A dictionary containing the corrupted input. The dictionary contains the following keys:
        - 'input_ids': An integer tensor of shape [batch_size, input_length] containing the token ids.
        - 'target_ids': An integer tensor of shape [batch_size, target_length] containing the token ids of the target
        - 'labels': An integer tensor of shape [batch_size, target_length] containing the token ids of the target. This
            tensor is used to compute the loss, and is the same as 'target_ids' except that the padding tokens are
            replaced with -100, and the start token of the 'target_ids' is shifted one to the right.
        - 'encoder_attention_mask': A binary integer tensor of shape [batch_size, input_length, input_length] which
            serves as the attention mask for the encoder.
        - 'decoder_attention_mask': A binary integer tensor of shape [batch_size, target_length, target_length] which
            serves as the attention mask for the decoder.
        - 'decoder_cross_attention_mask': A binary integer tensor of shape [batch_size, target_length, input_length]
            which serves as the attention mask for cross attention (from the decoder to the encoder).
        - 'length': An integer tensor of shape [batch_size] containing the lengths of each sequence in the batch,
            including both the input and target tokens (not including padding tokens).
    """
    # convert list to dict and tensorize input
    if isinstance(examples, list) and isinstance(examples[0], dict):
        batch = {
            k: np.array(
                [examples[i][k] for i in range(len(examples))]
            ) for k, v in examples[0].items()
        }
    else:
        batch = examples

    input_ids = batch[constants.DEPTHTokenizerConstants.INPUT_IDS]
    token_type_ids = batch[constants.DEPTHTokenizerConstants.TOKEN_TYPE_IDS]

    # Corrupt the text with span masking and sentence shuffling. Create inputs for both the encoder and decoder.
    batch_size, expanded_input_length = input_ids.shape
    sequence_lengths = np.sum(np.not_equal(token_type_ids, 0).astype(np.int32), axis=1)

    if pmi:
        mask_indices: np.ndarray = pmi_noise_mask(examples, ngram_vocab_set, tokenizer)
    else:
        mask_indices = np.reshape(
            np.concatenate(
                [
                    random_spans_noise_mask(
                        sequence_length=example_sequence_length,
                        maximum_length=expanded_input_length,
                        noise_density=noise_density,
                        mean_noise_span_length=mean_noise_span_length)
                    for example_sequence_length in sequence_lengths]
            ),
            newshape=[batch_size, expanded_input_length],
        )

    # Shift the span mask by two in order to account for the initial end of sentence and start of sentence tokens.
    span_mask = np.concatenate(
        [
            np.zeros(
                [batch_size, 2],
                dtype=np.bool_,
            ),
            mask_indices[:, :-2].astype(np.bool_),
        ],
        axis=1,
    )

    # Identify special tokens.
    special_tokens = tokenizer.all_special_ids

    # Ensure mask is only applied to non-special tokens.
    augmented_input_span_mask = np.where(
        np.isin(
            input_ids,
            special_tokens,
            invert=True,
        ),
        span_mask,
        False,
    )

    # Create a sentinel mask, where 0s indicate a lack of mask, positive values indicate the start of a masked span,
    #  and -1 indicates the continuation of a masked span.
    input_ids_sentinel = create_sentinel_ids_for_depth(tokenizer, augmented_input_span_mask.astype(np.int8))

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
            padded_sequence_length=expanded_input_length,
            # Ensure that all sentence tokens are included in the target
            sentence_token_ids=tokenizer.get_sentence_token_and_eosen_ids(),
        )
    )

    modified_encoder_input_ids = np.array(modified_encoder_input_ids)
    modified_encoder_token_type_ids = np.array(modified_encoder_token_type_ids)

    modified_label_ids = np.array(modified_label_ids)

    # Shift the target ids by one to the right. This is done to ensure that the model predicts the next token in the
    #  sequence.
    modified_decoder_input_ids = shift_tokens_right(
        modified_label_ids,
        pad_token_id=pad_token_id,
        decoder_start_token_id=decoder_start_token_id,
    )

    # T5 assumes that labels which correspond with pad tokens are replaced with -100. These tokens are ignored
    #  when computing the loss.
    modified_label_ids[modified_label_ids == pad_token_id] = -100

    # Prepend a 1 to the token type ids to account for the initial token of the decoder (i.e., the decoder start token).
    modified_decoder_token_type_ids = np.zeros(modified_label_ids.shape, dtype=np.int64)
    modified_decoder_token_type_ids[..., 1:] = modified_label_ids[..., :-1].copy()
    modified_decoder_token_type_ids[..., 0] = 1  # TODO: replace with decoder start token id

    if do_shuffle:
        batch_encoder_input_ids = []
        batch_encoder_token_type_ids = []

        for example_index in range(batch_size):

            example_input_ids = modified_encoder_input_ids[example_index]
            example_token_type_ids = modified_encoder_token_type_ids[example_index]

            if tokenizer.eos_token_id not in example_input_ids:
                # If there is no <EOS> token, then the example ends with two padding tokens.
                # We replace these padding tokens with <EOSEN> and <EOS> tokens respectively, and associate
                # these tokens with the last sentence.
                example_input_ids[-2] = tokenizer.end_of_sentence_token_id
                example_input_ids[-1] = tokenizer.eos_token_id
                example_token_type_ids[-2] = example_token_type_ids[-3]
                example_token_type_ids[-1] = example_token_type_ids[-3]

            # Ignore the first token (<EOSEN>) of each example as well as the last token (<EOS>) of each example.
            location_of_eos = np.argwhere(example_input_ids == tokenizer.eos_token_id)[-1][0]
            token_type_ids_of_sentences = example_token_type_ids[1:location_of_eos]
            input_ids_of_sentences = example_input_ids[1:location_of_eos]

            # Identify the unique sentence ids, the number of tokens in each sentence, and the start index of each
            #  sentence.
            (example_encoder_sentence_ids,
             example_encoder_sentence_start_indices,
             example_encoder_sentence_lengths) = np.unique(
                token_type_ids_of_sentences,
                return_counts=True,
                return_index=True,
            )

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
                    example_encoder_shuffled_sentence_start_indices + example_encoder_shuffled_sentence_lengths
            )
            example_encoder_shuffled_input_ids = np.concatenate(
                [
                    input_ids_of_sentences[start_index : end_index] for
                    start_index, end_index in
                    zip(example_encoder_shuffled_sentence_start_indices, example_encoder_shuffled_end_indices)
                 ]
            )
            example_encoder_shuffled_token_type_ids = np.concatenate(
                [
                    token_type_ids_of_sentences[start_index : end_index] for
                    start_index, end_index in
                    zip(example_encoder_shuffled_sentence_start_indices, example_encoder_shuffled_end_indices)
                ]
            )

            # Prepend the first new sentence with an <EOSEN> token.
            example_encoder_shuffled_input_ids = np.concatenate(
                [
                    np.array([tokenizer.end_of_sentence_token_id]),
                    example_encoder_shuffled_input_ids
                ]
            )
            example_encoder_shuffled_token_type_ids = np.concatenate(
                [
                    np.array([example_encoder_shuffled_token_type_ids[0]]),
                    example_encoder_shuffled_token_type_ids
                ]
            )

            # Add the <EOS> token at the end of the last sentence
            # shuffled_input_ids[example_encoder_sentence_start_indices[0]] = tokenizer.eos_token_id
            example_encoder_shuffled_input_ids = np.concatenate(
                [
                    example_encoder_shuffled_input_ids,
                    np.array([tokenizer.eos_token_id])
                ]
            )


            # Add the token type ID of the last sentence to the <EOS> token's token type ID.
            last_sentence_token_type_id = example_encoder_shuffled_token_type_ids[-1]
            # example_encoder_shuffled_token_type_ids[example_encoder_sentence_start_indices[0]] = (
            #     example_encoder_shuffled_token_type_ids[-1])
            example_encoder_shuffled_token_type_ids = np.concatenate(
                [
                    example_encoder_shuffled_token_type_ids,
                    np.array([last_sentence_token_type_id])
                ]
            )

            # Pad or truncate the input to the maximum sequence length.
            example_encoder_shuffled_input_ids = _pad_or_truncate_np(
                sequence=example_encoder_shuffled_input_ids,
                length=input_length,
                pad_token=tokenizer.pad_token_id,
            )
            example_encoder_shuffled_token_type_ids = _pad_or_truncate_np(
                sequence=example_encoder_shuffled_token_type_ids,
                length=input_length,
                pad_token=tokenizer.pad_token_type_id,
            )

            batch_encoder_input_ids.append(example_encoder_shuffled_input_ids)
            batch_encoder_token_type_ids.append(example_encoder_shuffled_token_type_ids)

        modified_encoder_input_ids = np.stack(batch_encoder_input_ids, axis=0)
        modified_encoder_token_type_ids = np.stack(batch_encoder_token_type_ids, axis=0)
    else:
        modified_encoder_token_type_ids = np.array(modified_encoder_token_type_ids)
        modified_encoder_input_ids = np.array(modified_encoder_input_ids)

    modified_label_ids = np.array(modified_label_ids)

    batch_encoder_self_attention_mask, batch_cross_attention_mask, batch_decoder_self_attention_mask = (
        create_depth_attention_masks(
            input_ids=modified_encoder_input_ids,
            target_ids=modified_decoder_input_ids,
            input_token_type_ids=modified_encoder_token_type_ids,
            target_token_type_ids=modified_decoder_token_type_ids,
            tokenizer=tokenizer,
        )
    )

    length = np.sum(
        np.not_equal(modified_encoder_input_ids, pad_token_id).astype(np.int32),
    ) + np.sum(
        np.not_equal(modified_decoder_input_ids, pad_token_id).astype(np.int32),
    )

    return {
        constants.DepthDataCollatorConstants.INPUT_IDS: modified_encoder_input_ids,
        constants.DepthDataCollatorConstants.TARGET_IDS: modified_decoder_input_ids,
        constants.DepthDataCollatorConstants.ENCODER_ATTENTION_MASK: batch_encoder_self_attention_mask,
        constants.DepthDataCollatorConstants.DECODER_ATTENTION_MASK: batch_decoder_self_attention_mask,
        constants.DepthDataCollatorConstants.CROSS_ATTENTION_MASK: batch_cross_attention_mask,
        constants.DepthDataCollatorConstants.LABELS: modified_label_ids,
        constants.DepthDataCollatorConstants.IS_SHUFFLED: np.array([do_shuffle] * batch_size).reshape([batch_size, 1]),
        constants.DepthDataCollatorConstants.LENGTH: np.array([length] * batch_size).reshape([batch_size, 1]),
    }


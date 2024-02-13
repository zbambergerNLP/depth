from transformers import T5TokenizerFast

import logging
import numpy as np
from torch import TensorType
import torch
import transformers
from typing import (
    List,
    Union,
    Optional,
)
import nltk
from transformers.utils import PaddingStrategy

from encoder_decoder_utils.constants import (
    DEPTHTokenizerConstants,

)

from transformers.tokenization_utils_base import (
    BatchEncoding,
    PreTokenizedInput,
    TextInput,
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
nltk.download('punkt')

# TODO: Migrate the following constants to the constants file.
# Truncation and Padding
TRUNCATION_LONGEST_FIRST = 'longest_first'
PADDING_LONGEST = 'longest'
PADDING_MAX_LENGTH = 'max_length'
PADDING_SIDE_RIGHT = 'right'
PADDING_SIDE_LEFT = 'left'

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
            text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
            text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
            text_target: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
            text_pair_target: Optional[
                Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]
            ] = None,
            add_special_tokens: bool = True,
            padding: Union[bool, str, PaddingStrategy] = False,
            truncation: Union[bool, str, TruncationStrategy] = None,
            max_length: Optional[int] = None,
            stride: int = 0,
            is_split_into_words: bool = False,
            pad_to_multiple_of: Optional[int] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            **kwargs,
    ) -> BatchEncoding:
        def get_input_ids(text_to_tokenize):
            if isinstance(text_to_tokenize, str):
                tokenized_text = self.tokenize_text(
                    text=text_to_tokenize,
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
                # seed += 1

        if max_length is None:
            max_length = max([len(ids) for ids in input_ids])

        batch_outputs = {}

        for example in input_ids:
            # Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by
            # the model. It adds special tokens, truncates sequences if overflowing while taking into account
            # the special tokens and manages a window stride for overflowing tokens
            # TODO: Move 'prepare_for_model' to a function within this class.
            outputs = self.prepare_for_model(
                ids=example,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors=return_tensors,
                return_token_type_ids=True,
                return_length=True,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        if return_tensors is not None:
            self.convert_to_tensors_(batch_outputs, return_tensors)

        return transformers.BatchEncoding(batch_outputs)

    def get_sentence_tokens(self):
        special_tokens = self.special_tokens_map[DEPTHTokenizerConstants.ADDITIONAL_SPECIAL_TOKENS]
        sentence_tokens = list(filter(lambda token: f'<{DEPTHTokenizerConstants.SENT}' in token, special_tokens))
        return sentence_tokens

    def get_sentence_token_ids(self):
        return self.convert_tokens_to_ids(self.get_sentence_tokens())

    def get_sentence_token_and_eosen_ids(self):
        sentence_tokens = self.get_sentence_tokens()
        sentence_tokens.append(DEPTHTokenizerConstants.END_OF_SENTENCE_TOKEN)
        return self.convert_tokens_to_ids(sentence_tokens)

    @property
    def end_of_sentence_token(self):
        return DEPTHTokenizerConstants.END_OF_SENTENCE_TOKEN

    @property
    def end_of_sentence_token_id(self):
        return self.convert_tokens_to_ids(self.end_of_sentence_token)


    def add_sentence_tokens_to_text(
            self,
            sentences: str,
    ) -> str:
        """Add SENT tokens in the beginning of each sentence.

        Truncate sentences that exceed 'max_num_sentences_in_tex'.

        :param sentences: A string composed of one or more sentences. Strings without punctuation or
            coherent sentence ends are considered as single sentences.
        :return: A string where sentences are prefixed with SENT tokens.
        """
        segmented_sentences = nltk.tokenize.sent_tokenize(sentences)
        num_sentences = len(segmented_sentences)
        segment_sentence_tokens = np.copy(self.get_sentence_tokens())
        np.random.shuffle(segment_sentence_tokens)

        # TODO: Randomly merge sentences if the number of sentences is greater than max_num_sentences_in_text.
        #  This is to avoid having to truncate sentences.
        if num_sentences > self.max_num_sentences_in_text:
            segmented_sentences = segmented_sentences[:self.max_num_sentences_in_text]
            segment_sentence_tokens = segment_sentence_tokens[:self.max_num_sentences_in_text]
        modified_sentences = ''.join([
            f'{DEPTHTokenizerConstants.END_OF_SENTENCE_TOKEN}{sentence_token}{sentence}' for sentence_token, sentence
            in zip(segment_sentence_tokens, segmented_sentences)])
        modified_sentences = f'{modified_sentences}{DEPTHTokenizerConstants.END_OF_SENTENCE_TOKEN}'
        return modified_sentences

    def tokenize_text(
            self,
            text: Union[str, List[str]],
            padding: Union[bool, str, transformers.tokenization_utils.PaddingStrategy] = True,
            truncation: Union[bool, str, transformers.tokenization_utils.TruncationStrategy] = True,
            max_length: int = 512,
    ) -> Union[List[str], List[List[str]]]:
        """Tokenize a potentially long text into a sequence of tokens. New sentences are prepended with designated
        sentence tokens.

        :param text: A of potentially long text (i.e., text that consist of multiple sentences). This parameter also
            supports tokenizing a sequence (batch) of long texts.
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

        if isinstance(text, str):
            augmented_text = self.add_sentence_tokens_to_text(
                sentences=text,
            )
            return self.tokenize(
                text=augmented_text,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
            )
        elif isinstance(text, list) and isinstance(text[0], str):
            tokenized_text = []
            for text_example in text:
                augmented_text_example = self.add_sentence_tokens_to_text(
                    sentences=text_example,
                )
                tokenized_text.append(self.tokenize(
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

    def prepare_for_model(
            self,
            ids: List[int],
            pair_ids: Optional[List[int]] = None,
            add_special_tokens: bool = True,
            padding: Union[bool, str, PaddingStrategy] = False,
            truncation: Union[bool, str, TruncationStrategy] = None,
            max_length: Optional[int] = None,
            stride: int = 0,
            pad_to_multiple_of: Optional[int] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            prepend_batch_axis: bool = False,
            **kwargs,
    ) -> BatchEncoding:

        assert pair_ids is None, "This method does not support pair_ids."

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        encoded_inputs = {}

        # Add special sentence tokens
        sequence = self.build_inputs_with_special_tokens(ids)
        len_ids = len(sequence)
        sentence_token_ids = self.get_sentence_token_ids()
        token_type_ids = [1]
        sentence_index = 0

        overflowing_tokens = []
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and len_ids > max_length:
            sequence, _, overflowing_tokens = self.truncate_sequences(
                sequence,
                num_tokens_to_remove=len_ids - max_length,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )
            len_ids = len(sequence)

        if return_overflowing_tokens:
            encoded_inputs[DEPTHTokenizerConstants.OVERFLOWING_TOKENS] = overflowing_tokens
            encoded_inputs[DEPTHTokenizerConstants.NUM_TRUNCATED_TOKENS] = len_ids - max_length

        # Add token type ids corresponding to the sentence in which the token is located
        # NOTE: The first sentence index is 1. Each sentence starts with a sentence token, except for the first sentence,
        # which starts with an <EOSEN> token before the first sentence token. In this way, each sentence token is preceded
        # by an <EOSEN> token.
        for token_index, token in enumerate(sequence[1:]):
            if token in sentence_token_ids or token == self.eos_token:
                sentence_index += 1
            if token == 0:
                sentence_index = 0
            token_type_ids.append(sentence_index)
        token_type_ids = token_type_ids[:max_length]  # Truncate to max length

        # Build output dictionary
        encoded_inputs[DEPTHTokenizerConstants.INPUT_IDS] = sequence
        if return_token_type_ids:
            encoded_inputs[DEPTHTokenizerConstants.TOKEN_TYPE_IDS] = token_type_ids
        if return_special_tokens_mask:
            encoded_inputs[DEPTHTokenizerConstants.SPECIAL_TOKENS_MASK] = (
                self.get_special_tokens_mask(sequence, already_has_special_tokens=True))

        # Check lengths
        assert max_length is None or len(encoded_inputs[DEPTHTokenizerConstants.INPUT_IDS]) <= max_length
        if max_length is None and len(encoded_inputs[DEPTHTokenizerConstants.INPUT_IDS]) > self.model_max_length:
            logger.warning(
                "Token indices sequence length is longer than the specified maximum sequence length "
                "for this model ({} > {}). Running this sequence through the model will result in "
                "indexing errors".format(len(sequence), self.model_max_length)
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
                and len(encoded_inputs[DEPTHTokenizerConstants.INPUT_IDS]) < self.model_max_length <= LARGE_INTEGER
        )
        if user_specified_padding and max_length is None and self.model_max_length > LARGE_INTEGER:
            logger.warning(
                "Sequence can't be padded as no maximum length is specified and the model maximum length is too high."
            )

        if needs_to_be_padded:
            difference = (max_length if max_length is not None else self.model_max_length) - len(
                encoded_inputs[DEPTHTokenizerConstants.INPUT_IDS]
            )
            if self.padding_side == PADDING_SIDE_RIGHT:
                if return_attention_mask:
                    encoded_inputs[DEPTHTokenizerConstants.ATTENTION_MASK] = [1] * len(
                        encoded_inputs[DEPTHTokenizerConstants.INPUT_IDS]) + [0] * difference
                if return_token_type_ids:
                    encoded_inputs[DEPTHTokenizerConstants.TOKEN_TYPE_IDS] = (
                            encoded_inputs[DEPTHTokenizerConstants.TOKEN_TYPE_IDS] + [
                        self.pad_token_type_id] * difference
                    )
                if return_special_tokens_mask:
                    encoded_inputs[DEPTHTokenizerConstants.SPECIAL_TOKENS_MASK] = (
                            encoded_inputs[DEPTHTokenizerConstants.SPECIAL_TOKENS_MASK] + [1] * difference)
                encoded_inputs[DEPTHTokenizerConstants.INPUT_IDS] = (
                        encoded_inputs[DEPTHTokenizerConstants.INPUT_IDS] + [self.pad_token_id] * difference)
            elif self.padding_side == PADDING_SIDE_LEFT:
                if return_attention_mask:
                    encoded_inputs[DEPTHTokenizerConstants.ATTENTION_MASK] = (
                            [0] * difference + [1] * len(encoded_inputs[DEPTHTokenizerConstants.INPUT_IDS]))
                if return_token_type_ids:
                    encoded_inputs[DEPTHTokenizerConstants.TOKEN_TYPE_IDS] = (
                            [self.pad_token_type_id] * difference + encoded_inputs[
                        DEPTHTokenizerConstants.TOKEN_TYPE_IDS]
                    )
                if return_special_tokens_mask:
                    encoded_inputs[DEPTHTokenizerConstants.SPECIAL_TOKENS_MASK] = (
                            [1] * difference + encoded_inputs[DEPTHTokenizerConstants.SPECIAL_TOKENS_MASK])
                encoded_inputs[DEPTHTokenizerConstants.INPUT_IDS] = (
                        [self.pad_token_id] * difference + encoded_inputs[DEPTHTokenizerConstants.INPUT_IDS])
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))
        else:
            if return_attention_mask:
                encoded_inputs[DEPTHTokenizerConstants.ATTENTION_MASK] = [1] * len(
                    encoded_inputs[DEPTHTokenizerConstants.INPUT_IDS])
            # TODO: support returning token type IDs and special tokens mask even if the user does not specify the padding
            #  option.

        if return_length:
            encoded_inputs[DEPTHTokenizerConstants.INPUT_LENGTH] = len(
                encoded_inputs[DEPTHTokenizerConstants.INPUT_IDS])

        return transformers.BatchEncoding(encoded_inputs)

    @staticmethod
    def convert_to_tensors_(batch_outputs: dict, return_tensors: str) -> None:
        # Do the tensor conversion in batch
        for key, value in batch_outputs.items():
            if return_tensors == "pt" and not isinstance(value, torch.Tensor):
                try:
                    batch_outputs[key] = torch.tensor(value)
                except ValueError:
                    raise ValueError(UNEVEN_SEQUENCES_FOR_BATCH_MSG)
                except RuntimeError:
                    if None in [item for sequence in value for item in sequence]:
                        raise ValueError(NO_PAD_TOKEN_FOR_BATCH_MSG)
                    else:
                        raise
            
            if return_tensors == 'np' and not isinstance(value, np.ndarray):
                try:
                    batch_outputs[key] = np.array(value)
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


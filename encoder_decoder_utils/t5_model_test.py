import typing

import numpy as np
import torch
import unittest

from transformers import T5Tokenizer, PreTrainedModel
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from absl.testing import parameterized

from encoder_decoder_utils import constants
from encoder_decoder_utils import tokenizer_utils
from encoder_decoder_utils import corruption
from encoder_decoder_utils import t5_model


class TestCustomT5Model(parameterized.TestCase):
    """
    Tests for the custom T5 model.

    The tests are parameterized, so that we can test multiple inputs at once.

    Our custom T5 model is a hierarchical encoder-decoder model. This heirarchy is created by adding special tokens
    to the input and target sequences. Namely, we add <eosen> tokens to the end of each sentence, and <sent_i> tokens
    to the beginning of each sentence, where i is the index of the sentence in the sequence. We also add <extra_id_i>
    tokens as part of the traditional span-masking objective (i.e. the objective used in the original T5 paper).
    """

    @staticmethod
    def set_seed(seed: int = 42):
        """
        Set the seed for reproducibility.

        :param seed: The seed to set.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setUp(self):
        self.t5_tokenizer = T5Tokenizer.from_pretrained(
            pretrained_model_name_or_path=constants.ModelHuggingFaceName.T5_BASE.value,
        )
        self.depth_tokenizer = tokenizer_utils.DepthTokenizer.from_pretrained(
            pretrained_model_name_or_path=constants.ModelHuggingFaceName.T5_BASE.value,
        )

    @staticmethod
    def get_model(model_implementation: str = constants.ModelImplementation.DEPTH.value) -> PreTrainedModel:
        if model_implementation == constants.ModelImplementation.DEPTH.value:
            return t5_model.DepthForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=constants.ModelHuggingFaceName.T5_BASE.value,
            )
        elif model_implementation == constants.ModelImplementation.HUGGINGFACE_T5.value:
            return T5ForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=constants.ModelHuggingFaceName.T5_BASE.value,
            )
        else:
            raise ValueError(f"Invalid model implementation: {model_implementation}")

    @parameterized.named_parameters(
        {
            constants.UnitTestConstants.TESTCASE_NAME: "Test single sentence no padding",
            constants.TokenizerConstants.INPUT_IDS: [
                [
                    "<eosen>", "<sent_17>", "hello", "<extra_id_82>", "<eosen>"
                ]
            ],
            # "input_token_type_ids": torch.tensor([[1, 1, 1, 1, 1]]),
            constants.DepthDataCollatorConstants.INPUT_TOKEN_TYPE_IDS: [
                [1, 1, 1, 1, 1],
            ],
            constants.TokenizerConstants.LABEL_IDS: [
                [
                    "<eosen>", "<sent_17>", "<extra_id_82>", "world", "<eosen>"
                ]
            ],
            constants.DepthDataCollatorConstants.LABEL_TOKEN_TYPE_IDS: [
                [1, 1, 1, 1, 1],
            ],
            constants.UnitTestConstants.MODEL_IMPLEMENTATION: constants.ModelImplementation.DEPTH.value,
            constants.UnitTestConstants.EXPECTED_ENCODER_SELF_ATTENTION_MASK: [
                [
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],  # sentence token in the input
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                ],
            ],
            constants.UnitTestConstants.EXPECTED_DECODER_SELF_ATTENTION_MASK: [
                [
                    [1, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],  # sentence token in target
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                ],
            ],
            constants.UnitTestConstants.EXPECTED_CROSS_ATTENTION_MASK: [
                [
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [0, 1, 0, 0, 0], # sentence token in target
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                ],
            ],
        },
        {
            constants.UnitTestConstants.TESTCASE_NAME: "T5 single sentence no padding",
            constants.TokenizerConstants.INPUT_IDS: [
                [
                    "hello", "<extra_id_82>",
                ]
            ],
            constants.DepthDataCollatorConstants.INPUT_TOKEN_TYPE_IDS: [
                [1, 1],
            ],
            constants.TokenizerConstants.LABEL_IDS: [
                [
                    "<extra_id_82>", "world",
                ]
            ],
            constants.DepthDataCollatorConstants.LABEL_TOKEN_TYPE_IDS: [
                [1, 1],
            ],
            constants.UnitTestConstants.MODEL_IMPLEMENTATION: constants.ModelImplementation.HUGGINGFACE_T5.value,
            constants.UnitTestConstants.EXPECTED_ENCODER_SELF_ATTENTION_MASK: [
                [
                    [1, 1],
                    [1, 1],
                ],
            ],
            constants.UnitTestConstants.EXPECTED_DECODER_SELF_ATTENTION_MASK: [
                [
                    [1, 0],
                    [1, 1],
                ],
            ],
            constants.UnitTestConstants.EXPECTED_CROSS_ATTENTION_MASK: [
                [
                    [1, 1],
                    [1, 1],
                ],
            ],
        },
        {
            constants.UnitTestConstants.TESTCASE_NAME: "T5 mulitple sentences no padding",
            constants.TokenizerConstants.INPUT_IDS: [
                [
                    "hi", "<extra_id_82>",  "how", "<extra_id_13>", "this", "morning?"
                ]
            ],
            constants.DepthDataCollatorConstants.INPUT_TOKEN_TYPE_IDS: [
                [1, 1, 1, 1, 1, 1],
            ],
            constants.TokenizerConstants.LABEL_IDS: [
                [
                    "<extra_id_82>", "Ben.", "<extra_id_13>", "are", "you"
                ]
            ],
            constants.DepthDataCollatorConstants.LABEL_TOKEN_TYPE_IDS: [
                [1, 1, 1, 1, 1],
            ],
            constants.UnitTestConstants.MODEL_IMPLEMENTATION: constants.ModelImplementation.HUGGINGFACE_T5.value,
            constants.UnitTestConstants.EXPECTED_ENCODER_SELF_ATTENTION_MASK: [
                [
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                ],
            ],
            constants.UnitTestConstants.EXPECTED_DECODER_SELF_ATTENTION_MASK: [
                [
                    [1, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                ],
            ],
            constants.UnitTestConstants.EXPECTED_CROSS_ATTENTION_MASK: [
                [
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                ],
            ],
        },
        {
            constants.UnitTestConstants.TESTCASE_NAME: "T5 multiple sentences with padding",
            constants.TokenizerConstants.INPUT_IDS: [
                # 15 tokens, last 3 are padding
                [
                    "my", "name", "is", "<extra_id_82>", "and", "i", "study", "at", "the", "university", "of",
                    "<extra_id_13>",
                    "<pad>", "<pad>", "<pad>",
                ]
            ],
            constants.DepthDataCollatorConstants.INPUT_TOKEN_TYPE_IDS: [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
            ],
            constants.TokenizerConstants.LABEL_IDS: [
                # 10 tokens, the last 5 are padding
                [
                    "<extra_id_82>", "john", "<extra_id_13>", "california", "berkeley",
                    "<pad>", "<pad>", "<pad>", "<pad>", "<pad>",
                ]
            ],
            constants.DepthDataCollatorConstants.LABEL_TOKEN_TYPE_IDS: [
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
            ],
            constants.UnitTestConstants.MODEL_IMPLEMENTATION: constants.ModelImplementation.HUGGINGFACE_T5.value,
            constants.UnitTestConstants.EXPECTED_ENCODER_SELF_ATTENTION_MASK: [
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # Padding
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # Padding
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # Padding
                ],
            ],
            constants.UnitTestConstants.EXPECTED_DECODER_SELF_ATTENTION_MASK: [
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 0, 0 ,0, 0], # Padding
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0], # Padding
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0], # Padding
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0], # Padding
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # Padding
                ],
            ],
            constants.UnitTestConstants.EXPECTED_CROSS_ATTENTION_MASK: [
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # Padding
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # Padding
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # Padding
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # Padding
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # Padding
                ],
            ],
        },
        {
            constants.UnitTestConstants.TESTCASE_NAME: "test multiple sentences no padding",
            constants.TokenizerConstants.INPUT_IDS: [
                [
                    "<eosen>",
                    "<sent_17>", "hello", "<extra_id_82>", "<eosen>",
                    "<sent_9>", "<extra_id_13>", "are", "you?", "<eosen>"
                ]
            ],
            constants.DepthDataCollatorConstants.INPUT_TOKEN_TYPE_IDS: [
                [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            ],
            constants.TokenizerConstants.LABEL_IDS: [
                [
                    "<eosen>",
                    "<sent_17>", "<extra_id_82>", "world", "<eosen>",
                    "<sent_9>", "<extra_id_13>", "how", "<eosen>"
                ]
            ],
            constants.DepthDataCollatorConstants.LABEL_TOKEN_TYPE_IDS: [
                [1, 1, 1, 1, 1, 2, 2, 2, 2],
            ],
            constants.UnitTestConstants.MODEL_IMPLEMENTATION: constants.ModelImplementation.DEPTH.value,
            # 10 x 10 attention mask corresponding to the 10 tokens in the input sequence
            constants.UnitTestConstants.EXPECTED_ENCODER_SELF_ATTENTION_MASK: [
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ],
            ],
            # 9 x 9 attention mask corresponding to the 9 tokens in the target sequence
            constants.UnitTestConstants.EXPECTED_DECODER_SELF_ATTENTION_MASK: [
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0],  # Padding token in decoder
                    [1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0],  # sentence token #1
                    [1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0],  # sentence token #2
                    [1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                ],
            ],
            # 9 x 10 attention mask corresponding to the 9 tokens in the target sequence attending to 10 tokens in the
            # input sequence
            constants.UnitTestConstants.EXPECTED_CROSS_ATTENTION_MASK: [
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Padding token in decoder
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # sentence token #1
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # sentence token #2
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ],
            ],
        },
        {
            constants.UnitTestConstants.TESTCASE_NAME: "Test single sentence with padding",
            constants.TokenizerConstants.INPUT_IDS: [
                [
                    "<eosen>",
                    "<sent_17>", "hello", "<extra_id_82>", "<eosen>",
                    "<pad>", "<pad>", "<pad>", "<pad>", "<pad>"
                ]
            ],
            constants.DepthDataCollatorConstants.INPUT_TOKEN_TYPE_IDS: [
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            ],
            constants.TokenizerConstants.LABEL_IDS: [
                [
                    "<eosen>",
                    "<sent_17>", "<extra_id_82>", "world", "<eosen>",
                    "<pad>", "<pad>", "<pad>", "<pad>"
                ]
            ],
            constants.DepthDataCollatorConstants.LABEL_TOKEN_TYPE_IDS: [
                    [1, 1, 1, 1, 1, 0, 0, 0, 0]
            ],
            constants.UnitTestConstants.MODEL_IMPLEMENTATION: constants.ModelImplementation.DEPTH.value,
            # 10 x 10 attention mask corresponding to the 10 tokens in the input sequence
            constants.UnitTestConstants.EXPECTED_ENCODER_SELF_ATTENTION_MASK: [
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0], # Sentence token
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ],
            ],
            # 9 x 9 attention mask corresponding to the 9 (of which 4 are padding) tokens in the target sequence
            constants.UnitTestConstants.EXPECTED_DECODER_SELF_ATTENTION_MASK: [
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0],  # Sentence token in decoder
                    [1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                ],
            ],
            # 9 x 10 attention mask corresponding to the 9 tokens in the target sequence attending to 10 tokens in the
            # input sequence
            constants.UnitTestConstants.EXPECTED_CROSS_ATTENTION_MASK: [
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Sentence token in decoder
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Padding token in decoder
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Padding token in decoder
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Padding token in decoder
                ],
            ],
        },
        {
            constants.UnitTestConstants.TESTCASE_NAME: "Test multiple sentences with padding",
            constants.TokenizerConstants.INPUT_IDS: [
                [
                    "<eosen>",
                    "<sent_17>", "hello", "<extra_id_82>", "<eosen>",
                    "<sent_9>", "<extra_id_13>", "are", "you?", "<eosen>",
                    "<pad>", "<pad>", "<pad>", "<pad>", "<pad>"
                ]
            ],
            constants.DepthDataCollatorConstants.INPUT_TOKEN_TYPE_IDS: [
                [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0],
            ],
            constants.TokenizerConstants.LABEL_IDS: [
                [
                    "<eosen>",
                    "<sent_17>", "<extra_id_82>", "world", "<eosen>",
                    "<sent_9>", "<extra_id_13>", "hello", "<eosen>",
                    "<pad>", "<pad>", "<pad>", "<pad>"
                ]
            ],
            constants.DepthDataCollatorConstants.LABEL_TOKEN_TYPE_IDS: [
                [1, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0],
            ],
            constants.UnitTestConstants.MODEL_IMPLEMENTATION: constants.ModelImplementation.DEPTH.value,
            # 15 x 15 attention mask corresponding to the 15 tokens in the input sequence (including padding) attending
            # to 15 tokens in the input sequence (including padding). Padding tokens can attend only themselves.
            constants.UnitTestConstants.EXPECTED_ENCODER_SELF_ATTENTION_MASK: [
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Sentence token
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # Sentence token
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # Padding token
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # Padding token
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # Padding token
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # Padding token
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # Padding token
                ],
            ],
            # 13 x 15 attention mask corresponding to 13 tokens in the target sequence attending to 15 tokens in the
            # input sequence.
            # NOTE: Within cross attention, padding tokens from the decoder to all tokens in the encoder.
            # We are able to do this since loss will not be computed over padding tokens anyway.
            constants.UnitTestConstants.EXPECTED_CROSS_ATTENTION_MASK: [
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Padding token in decoder
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Initial <EOSEN> in decoder
                    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Sentence token in decoder
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Sentence token in decoder
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # padding token in decoder
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # padding token in decoder
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # padding token in decoder
                ]
            ],
            # 13 x 13 attention mask corresponding to 13 tokens in the target sequence attending to 13 tokens in the
            # target sequence
            constants.UnitTestConstants.EXPECTED_DECODER_SELF_ATTENTION_MASK: [
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # padding
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # <EOSEN>
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # sentence token #17
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # sentinel token #82
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # "world"
                    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # <EOSEN>
                    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # sentence token #9
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # sentinel token #13
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # "hello"
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],  # <EOSEN>
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # padding
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # padding
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # padding
                ],
            ],
        },
    )
    def test_forward(
            self,
            input_ids: typing.List[typing.List[str]],
            input_token_type_ids: typing.List[typing.List[int]],
            label_ids: typing.List[typing.List[str]],
            label_token_type_ids: typing.List[typing.List[int]],
            model_implementation: str,
            expected_encoder_self_attention_mask: typing.List[typing.List[typing.List[int]]],
            expected_cross_attention_mask: typing.List[typing.List[typing.List[int]]],
            expected_decoder_self_attention_mask: typing.List[typing.List[typing.List[int]]],
            seed: int = 42,
    ):
        """

        :param input_ids: A batch of input_ids. A tensor of shape (batch_size, input_sequence_length)
        :param input_token_type_ids: A tensor of shape (batch_size, input_sequence_length) where the value at
            position (i, j) is the sentence index of the jth token in the ith input sequence.
        :param label_ids: A batch of labels. A tensor of shape (batch_size, target_sequence_length)
        :param label_token_type_ids: A tensor of shape (batch_size, target_sequence_length) where the value at
            position (i, j) is the sentence index of the jth token in the ith label sequence.
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
        self.set_seed(seed)
        model = self.get_model(
            model_implementation=model_implementation,
        )
        tokenizer = (
            self.t5_tokenizer if
            model_implementation == constants.ModelImplementation.HUGGINGFACE_T5 else
            self.depth_tokenizer
        )
        tokenized_input_ids = np.array(
            [tokenizer.convert_tokens_to_ids(tokens)
             for tokens in input_ids]
        )
        tokenized_label_ids = np.array(
            [tokenizer.convert_tokens_to_ids(tokens)
             for tokens in label_ids]
        )
        input_token_type_ids = np.array(input_token_type_ids)
        label_token_type_ids = np.array(label_token_type_ids)

        # Shift label tokens to the right in order to create the target input IDs.
        tokenized_target_ids = corruption.shift_tokens_right(
            tokenized_label_ids,
            tokenizer.pad_token_id,
            tokenizer.pad_token_id,
        )
        target_token_type_ids = corruption.shift_tokens_right(
            label_token_type_ids,
            tokenizer.pad_token_id,
            tokenizer.pad_token_id,
        )

        batch_encoder_self_attention_mask, batch_cross_attention_mask, batch_decoder_self_attention_mask = (
            corruption.create_depth_attention_masks(
                input_ids=tokenized_input_ids,
                target_ids=tokenized_target_ids,
                input_token_type_ids=input_token_type_ids,
                target_token_type_ids=target_token_type_ids,
                tokenizer=tokenizer,
            )
        )

        if model_implementation == constants.ModelImplementation.HUGGINGFACE_T5.value:
            # T5 model
            outputs = model.forward(
                input_ids=torch.tensor(tokenized_input_ids, dtype=torch.long),
                labels=torch.tensor(tokenized_label_ids, dtype=torch.long),
                output_attentions=True,
            )
        else:
            # Depth model
            outputs = model.forward(
                input_ids=torch.tensor(tokenized_input_ids, dtype=torch.long),
                encoder_attention_mask=torch.tensor(batch_encoder_self_attention_mask, dtype=torch.int8),
                target_ids=torch.tensor(tokenized_target_ids, dtype=torch.long),
                cross_attention_mask=torch.tensor(batch_cross_attention_mask, dtype=torch.int8),
                decoder_attention_mask=torch.tensor(batch_decoder_self_attention_mask, dtype=torch.int8),
                labels=torch.tensor(tokenized_label_ids, dtype=torch.long),
                output_attentions=True,
            )

        # Attention outputs consist of tuples of length num_layers where each element is a tensor of shape
        # (batch_size, num_heads, input_sequence_length, input_sequence_length)
        encoder_attentions = outputs.encoder_attentions
        cross_attentions = outputs.cross_attentions
        decoder_attentions = outputs.decoder_attentions

        model.eval()
        for layer_index in range(len(encoder_attentions)):
            for head_index in range(encoder_attentions[layer_index].shape[1]):

                # Test encoder self-attention
                is_masked_encoder_self_attention = torch.where(
                    condition=(
                        (encoder_attentions[layer_index][:, head_index, :, :] == 0)
                    ),
                    input=torch.zeros(encoder_attentions[layer_index][:, head_index, :, :].shape),
                    other=torch.ones(encoder_attentions[layer_index][:, head_index, :, :].shape)
                )
                self.assertTrue(
                    torch.all(
                        is_masked_encoder_self_attention == torch.tensor(expected_encoder_self_attention_mask)
                    ),
                    f'In layer {layer_index} and head {head_index} of the encoder self-attention,\n'
                    f'expected masks: {expected_encoder_self_attention_mask},\n'
                    f'actual masks: {is_masked_encoder_self_attention.int().tolist()}\n'
                    f'attention weights: {encoder_attentions[layer_index][:, head_index, :, :].tolist()}'
                )

        for layer_index in range(len(decoder_attentions)):
            for head_index in range(decoder_attentions[layer_index].shape[1]):
                # Test decoder self-attention
                is_masked_decoder_self_attention = torch.where(
                    condition=(
                        (decoder_attentions[layer_index][:, head_index, :, :] == 0)
                    ),
                    input=torch.zeros(decoder_attentions[layer_index][:, head_index, :, :].shape),
                    other=torch.ones(decoder_attentions[layer_index][:, head_index, :, :].shape)
                )
                self.assertTrue(
                    torch.all(
                        is_masked_decoder_self_attention == torch.tensor(expected_decoder_self_attention_mask)
                    ),
                    f'In layer {layer_index} and head {head_index} of the decoder self-attention,\n'
                    f'expected masks: {expected_decoder_self_attention_mask},\n'
                    f'actual masks: {is_masked_decoder_self_attention.int().tolist()}\n'
                    f'attention weights: {decoder_attentions[layer_index][:, head_index, :, :].tolist()}'
                )

                is_masked_cross_attention = torch.where(
                    condition=(
                        (cross_attentions[layer_index][:, head_index, :, :] == 0)
                    ),
                    input=torch.zeros(cross_attentions[layer_index][:, head_index, :, :].shape),
                    other=torch.ones(cross_attentions[layer_index][:, head_index, :, :].shape)
                )
                self.assertTrue(
                    torch.all(
                        is_masked_cross_attention == torch.tensor(expected_cross_attention_mask)
                    ),
                    f'In layer {layer_index} and head {head_index} of the cross-attention,\n'
                    f'expected masks: {expected_cross_attention_mask},\n'
                    f'actual masks: {is_masked_cross_attention.int().tolist()}\n'
                    f'attention weights: {cross_attentions[layer_index][:, head_index, :, :].tolist()}'
                )


if __name__ == '__main__':
    unittest.main()

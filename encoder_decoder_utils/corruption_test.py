import typing
import unittest
import numpy as np
import torch
import transformers.models.t5
from encoder_decoder_utils import corruption as corruption_lib
from encoder_decoder_utils import constants
from encoder_decoder_utils import test_constants
from absl.testing import parameterized
import random
import nltk
from encoder_decoder_utils.tokenizer_utils import DepthTokenizer

# Test inputs
EXAMPLE_1 = 'Hello world! I am learning to use tokenizers. Did you know they are this cool?'
EXAMPLE_2 = 'His lecture was so boring... I couldn\'t help but doze off.'
EXAMPLE_3 = 'Here is a first sentence! This is a second. What about a third? Four is enough!'

nltk.download('punkt')


class CorruptionTest(parameterized.TestCase):

    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        transformers.set_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setUp(self):
        self.depth_tokenizer = DepthTokenizer.from_pretrained(
            pretrained_model_name_or_path=constants.ModelHuggingFaceName.T5_BASE.value,
            max_num_sentences_in_text=constants.DEPTHTokenizerConstants.NUM_SENT_TOKENS,
        )
        self.t5_tokenizer = transformers.T5Tokenizer.from_pretrained(
            pretrained_model_name_or_path=constants.ModelHuggingFaceName.T5_BASE.value,
        )

    @parameterized.named_parameters(
        {constants.UnitTestConstants.TESTCASE_NAME: 'short_sequence_length',
         constants.TokenizerConstants.INPUT_LENGTH: 5,
         constants.DepthDataCollatorConstants.NOISE_DENSITY: 0.5,
         constants.DepthDataCollatorConstants.MEAN_NOISE_SPAN_LENGTH: 2,
         constants.UnitTestConstants.SEED: 42,
         constants.UnitTestConstants.EXPECTED_SPAN_MASKS: [False, True, True, False, False]
         },
        {constants.UnitTestConstants.TESTCASE_NAME: 'medium_sequence_length',
         constants.TokenizerConstants.INPUT_LENGTH: 10,
         constants.DepthDataCollatorConstants.NOISE_DENSITY: 0.5,
         constants.DepthDataCollatorConstants.MEAN_NOISE_SPAN_LENGTH: 3,
         constants.UnitTestConstants.SEED: 42,
         constants.UnitTestConstants.EXPECTED_SPAN_MASKS: [
             True, False, False, False, True, True, True, False, False, True
         ],
         },
        {constants.UnitTestConstants.TESTCASE_NAME: 'medium_sequence_length_different_seed',
         constants.TokenizerConstants.INPUT_LENGTH: 10,
         constants.DepthDataCollatorConstants.NOISE_DENSITY: 0.5,
         constants.DepthDataCollatorConstants.MEAN_NOISE_SPAN_LENGTH: 3,
         constants.UnitTestConstants.SEED: 48,
         constants.UnitTestConstants.EXPECTED_SPAN_MASKS: [
             True, False, False, False, True, True, True, True, False, False
         ],
         },
        {constants.UnitTestConstants.TESTCASE_NAME: 'medium_sequence_length_lower_density',
         constants.TokenizerConstants.INPUT_LENGTH: 10,
         constants.DepthDataCollatorConstants.NOISE_DENSITY: 0.3,
         constants.DepthDataCollatorConstants.MEAN_NOISE_SPAN_LENGTH: 3,
         constants.UnitTestConstants.SEED: 42,
         constants.UnitTestConstants.EXPECTED_SPAN_MASKS: [
             False, False, True, True, True, False, False, False, False, False
         ],
         },
        {constants.UnitTestConstants.TESTCASE_NAME: 'medium_sequence_length_higher_density',
         constants.TokenizerConstants.INPUT_LENGTH: 10,
         constants.DepthDataCollatorConstants.NOISE_DENSITY: 0.9,
         constants.DepthDataCollatorConstants.MEAN_NOISE_SPAN_LENGTH: 9,
         constants.UnitTestConstants.SEED: 42,
         constants.UnitTestConstants.EXPECTED_SPAN_MASKS: [True, False, True, True, True, True, True, True, True, True],
         },
    )
    def test_random_spans_noise_mask(
            self,
            input_length: int,
            noise_density: float,
            mean_noise_span_length: int,
            seed: int,
            expected_span_masks: typing.List[bool],
    ):
        self.set_seed(seed)
        span_masks = corruption_lib.random_spans_noise_mask(
            sequence_length=input_length,
            maximum_length=input_length,
            noise_density=noise_density,
            mean_noise_span_length=mean_noise_span_length,
        )
        np.testing.assert_array_equal(span_masks, expected_span_masks)

    @parameterized.named_parameters(
        {constants.UnitTestConstants.TESTCASE_NAME: 'short_sequence_length',
         constants.TokenizerConstants.INPUT_LENGTH: 5,
         constants.DepthDataCollatorConstants.NOISE_DENSITY: 0.5,
         constants.DepthDataCollatorConstants.MEAN_NOISE_SPAN_LENGTH: 2,
         constants.UnitTestConstants.SEED: 42,
         constants.UnitTestConstants.EXPECTED_INPUT_IDS_SENTINEL: [[0, 32099, -1, 0, 0]],
         },
        {constants.UnitTestConstants.TESTCASE_NAME: 'medium_sequence_length',
         constants.TokenizerConstants.INPUT_LENGTH: 10,
         constants.DepthDataCollatorConstants.NOISE_DENSITY: 0.5,
         constants.DepthDataCollatorConstants.MEAN_NOISE_SPAN_LENGTH: 3,
         constants.UnitTestConstants.SEED: 42,
         constants.UnitTestConstants.EXPECTED_INPUT_IDS_SENTINEL: [[32099, 0, 0, 0, 32098, -1, -1, 0, 0, 32097]],
         },
        {constants.UnitTestConstants.TESTCASE_NAME: 'medium_sequence_length_different_seed',
         constants.TokenizerConstants.INPUT_LENGTH: 10,
         constants.DepthDataCollatorConstants.NOISE_DENSITY: 0.5,
         constants.DepthDataCollatorConstants.MEAN_NOISE_SPAN_LENGTH: 3,
         constants.UnitTestConstants.SEED: 48,
         constants.UnitTestConstants.EXPECTED_INPUT_IDS_SENTINEL: [[32099, 0, 0, 0, 32098, -1, -1, -1, 0, 0]],
         },
        {constants.UnitTestConstants.TESTCASE_NAME: 'medium_sequence_length_lower_density',
         constants.TokenizerConstants.INPUT_LENGTH: 10,
         constants.DepthDataCollatorConstants.NOISE_DENSITY: 0.3,
         constants.DepthDataCollatorConstants.MEAN_NOISE_SPAN_LENGTH: 3,
         constants.UnitTestConstants.SEED: 42,
         constants.UnitTestConstants.EXPECTED_INPUT_IDS_SENTINEL: [[0, 0, 32099, -1, -1, 0, 0, 0, 0, 0]],
         },
        {constants.UnitTestConstants.TESTCASE_NAME: 'medium_sequence_length_higher_density',
         constants.TokenizerConstants.INPUT_LENGTH: 10,
         constants.DepthDataCollatorConstants.NOISE_DENSITY: 0.9,
         constants.DepthDataCollatorConstants.MEAN_NOISE_SPAN_LENGTH: 9,
         constants.UnitTestConstants.SEED: 42,
         constants.UnitTestConstants.EXPECTED_INPUT_IDS_SENTINEL: [[32099, 0, 32098, -1, -1, -1, -1, -1, -1, -1]],
         },
    )
    def test_create_sentinel_ids_for_t5(
            self,
            input_length: int,
            noise_density: float,
            mean_noise_span_length: int,
            seed: int,
            expected_input_ids_sentinel: typing.List[typing.List[int]],
    ):
        self.set_seed(seed)
        tokenizer = transformers.T5Tokenizer.from_pretrained(constants.ModelHuggingFaceName.T5_BASE.value)
        span_masks = corruption_lib.random_spans_noise_mask(
            sequence_length=input_length,
            noise_density=noise_density,
            mean_noise_span_length=mean_noise_span_length,
            maximum_length=input_length,

        )
        input_ids_sentinel = corruption_lib.create_sentinel_ids_for_t5(
            np.expand_dims(span_masks, axis=0).astype(np.int8),
            vocab_size=len(tokenizer),
        )
        np.testing.assert_array_equal(input_ids_sentinel, np.array(expected_input_ids_sentinel))

    @parameterized.named_parameters(
        {
            constants.UnitTestConstants.TESTCASE_NAME: 'basic_test',
            constants.UnitTestConstants.EXAMPLES: [
                'Hello world!',
                'Here is an example with a longer sentence',
                'An example with multiple sentences? Might be tricky... Worth testing!',
            ],
            constants.TokenizerConstants.INPUT_LENGTH: 20,
            constants.TokenizerConstants.TARGET_LENGTH: 10,
            constants.UnitTestConstants.EXPECTED_MODIFIED_INPUT_IDS: np.array(
                [
                    [32099, 296, 55, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [947, 32099, 46, 677, 28, 3, 9, 1200, 32098, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [32099, 28, 1317, 16513, 32098, 16114, 233, 16990, 2505, 32097, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
            ),
            # Recall that in the target of T5, padding tokens (usually 0) are replaced with -100.
            constants.UnitTestConstants.EXPECTED_MODIFIED_LABEL_IDS: np.array(
                [
                    [32099, 8774, -100, -100, -100, -100, -100, -100, -100, -100],
                    [32099, 19, 32098, 7142, -100, -100, -100, -100, -100, -100],
                    [32099, 389, 677, 32098, 58, 23840, 36, 32097, 55, -100],
                ],
            ),
        },
        {
            constants.UnitTestConstants.TESTCASE_NAME: 'truncated targets',
            constants.UnitTestConstants.EXAMPLES: [EXAMPLE_1, EXAMPLE_2, EXAMPLE_3],
            constants.TokenizerConstants.INPUT_LENGTH: 20,
            constants.TokenizerConstants.TARGET_LENGTH: 10,
            constants.UnitTestConstants.EXPECTED_MODIFIED_INPUT_IDS: np.array(
                [
                    [32099, 296, 32098, 183, 32097, 169, 14145, 8585, 7, 5, 3963, 25, 214, 32096, 1, 0, 0, 0, 0, 0],
                    [32099, 47, 32098, 27, 2654, 31, 17, 199, 32097, 776, 326, 5, 1, 0, 0, 0, 0, 0, 0, 0],
                    [947, 19, 32099, 100, 19, 3, 9, 511, 5, 32098, 81, 32097, 1, 0, 0, 0, 0, 0, 0, 0],
                ],
            ),
            constants.UnitTestConstants.EXPECTED_MODIFIED_LABEL_IDS: np.array(

                [
                    # Truncated 33, 48, 1633, which are masked by sentinel 32096
                    [32099, 8774, 32098, 55, 27, 32097, 1036, 12, 32096, 79],
                    # No truncation
                    [32099, 978, 7177, 32098, 78, 13006, 233, 32097, 68, 103],
                    # Truncated 9, 1025, 58, which are masked by sentinel 32097
                    [32099, 3, 9, 166, 7142, 55, 32098, 363, 32097, 3],
                ],
            ),
        },
    )
    def test_corrupt_for_vanilla_t5(
            self,
            examples,
            input_length,
            target_length,
            expected_modified_input_ids,
            expected_modified_label_ids,
            seed=42,
    ):
        """

        :param examples: A list (batch) of strings corresponding to the examples to be corrupted.
        :param input_length: The length of the input sequence.
        :param target_length: The length of the target sequence.
        :param expected_modified_input_ids: A tensor of shape [batch_size, input_length] corresponding to the expected
            modified input ids.
        :param expected_modified_label_ids: A tensor of shape [batch_size, target_length] corresponding to the expected
            modified label ids.
        :param seed: The seed to use for the test.
        """
        # Set seed
        np.random.seed(seed)

        tokenizer = transformers.T5TokenizerFast.from_pretrained('t5-small')
        tokenized_examples = tokenizer(
            examples,
            max_length=input_length,
            truncation='only_first',
            padding='longest',
            return_tensors='np'
        )
        batch_encoding = corruption_lib.corrupt_for_vanilla_t5(
            examples=tokenized_examples,
            vocab_size=len(tokenizer),
            input_length=input_length,
            target_length=target_length,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            decoder_start_token_id=tokenizer.pad_token_id,
        )
        np.testing.assert_almost_equal(
            actual=batch_encoding['input_ids'],
            desired=expected_modified_input_ids,
            err_msg=f'Excepted: {expected_modified_input_ids}\nActual: {batch_encoding["input_ids"]}',
        )
        np.testing.assert_almost_equal(
            actual=batch_encoding['labels'],
            desired=expected_modified_label_ids,
            err_msg=f'Excepted: {expected_modified_label_ids}\nActual: {batch_encoding["labels"]}',
        )

    @parameterized.named_parameters(
        {
            constants.UnitTestConstants.TESTCASE_NAME: "No n-grams in PMI Vocab",
            constants.UnitTestConstants.EXAMPLES: "Ofek went to Taub.",
            constants.UnitTestConstants.EXPECTED_SPAN_MASKS: [1, 1, 1, 1, 0, 0, 0, 0],
            constants.TokenizerConstants.INPUT_LENGTH: 512,
            constants.DepthDataCollatorConstants.NOISE_DENSITY: 0.5,
        },
        {
            constants.UnitTestConstants.TESTCASE_NAME: "Gibberish",
            constants.UnitTestConstants.EXAMPLES: "asdvbdsasd asdvewasdf",
            constants.UnitTestConstants.EXPECTED_SPAN_MASKS: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            constants.TokenizerConstants.INPUT_LENGTH: 512,
            constants.DepthDataCollatorConstants.NOISE_DENSITY: 0.5,
        },
        {
            constants.UnitTestConstants.TESTCASE_NAME: "Some n-grams in PMI Vocab",
            constants.UnitTestConstants.EXAMPLES: (
                    "I have to tell everything that is happening, but what happens after that, i don't know"
            ),
            constants.UnitTestConstants.EXPECTED_SPAN_MASKS: [
                1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0
            ],
            constants.TokenizerConstants.INPUT_LENGTH: 512,
            constants.DepthDataCollatorConstants.NOISE_DENSITY: 0.5,
        },
        {
            constants.UnitTestConstants.TESTCASE_NAME: "extra punctuation",
            constants.UnitTestConstants.EXAMPLES: (
                    "I want to tell you, maybe ask you? maybe yell! maybe scream & yell - then tell you.",
            ),
            constants.UnitTestConstants.EXPECTED_SPAN_MASKS: [
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1
            ],
            constants.TokenizerConstants.INPUT_LENGTH: 512,
            constants.DepthDataCollatorConstants.NOISE_DENSITY: 0.5,
        },
        {
            constants.UnitTestConstants.TESTCASE_NAME: "Testing high mlm_probability",
            constants.UnitTestConstants.EXAMPLES: "butter and jelly pupil total expenditures stained glass windows",
            constants.UnitTestConstants.EXPECTED_SPAN_MASKS: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            constants.TokenizerConstants.INPUT_LENGTH: 512,
            constants.DepthDataCollatorConstants.NOISE_DENSITY: 1,
        },
        {
            constants.UnitTestConstants.TESTCASE_NAME: "Testing low mlm_probability",
            constants.UnitTestConstants.EXAMPLES: "butter and jelly pupil total expenditures stained glass windows",
            constants.UnitTestConstants.EXPECTED_SPAN_MASKS: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            constants.TokenizerConstants.INPUT_LENGTH: 512,
            constants.DepthDataCollatorConstants.NOISE_DENSITY: 0,
        },

    )
    def test_pmi_mask_word(
            self,
            examples: str,
            expected_span_masks: typing.List[int],
            input_length: int,
            noise_density: float,
            seed: int = 42,
    ):
        """
        Test different use cases of the method "pmi_word_mask"
        :param input_tokens: input tokens to test
        :param max_predictions: max predictions to test
        :param mlm_probability: mlm probability to test
        :return: None
        """
        random.seed(seed)
        input_tokens = self.t5_tokenizer(
            examples,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
        )[constants.TokenizerConstants.INPUT_IDS]
        input_tokens = input_tokens.squeeze()
        ref_tokens = []
        for input_id in input_tokens:
            token = self.t5_tokenizer._convert_id_to_token(input_id.item())
            ref_tokens.append(token)
        mask_labels_for_sample = corruption_lib.pmi_word_mask(
            ref_tokens,
            test_constants.PMI_DEMO_VOCAB,
            input_length,
            noise_density,
        )
        self.assertIsNotNone(mask_labels_for_sample)
        self.assertListEqual(mask_labels_for_sample, expected_span_masks)
        self.assertEquals(len(mask_labels_for_sample), len(ref_tokens))

    @parameterized.named_parameters(
        {
            constants.UnitTestConstants.TESTCASE_NAME: f"test stand use of pmi_noise_mask",
            constants.UnitTestConstants.EXAMPLES: test_constants.DEMO_TEXTS,
            constants.DepthDataCollatorConstants.PMI_VOCAB: test_constants.PMI_DEMO_VOCAB,
        },
    )
    def test_pmi_noise_mask(
            self,
            examples: typing.List[str],
            pmi_vocab: typing.Set[str],
    ):
        """
        Test the method "pmi_noise_mask". This method will test the standard use case of the method as expected to
        happen in pre-training.
        :param examples: examples to test
        :param pmi_vocab: pmi vocab to test
        """
        tokenized_examples = self.t5_tokenizer(
            examples,
            return_tensors="np",
            add_special_tokens=False,
            truncation=True,
            padding=True,
        )
        predicted_mask_labels = corruption_lib.pmi_noise_mask(
            tokenized_examples,
            pmi_vocab,
            self.t5_tokenizer,
        )
        # TODO: check the correctness of the output relative to the input. If you set the seed initially, you should
        #  get the same output for the same input.
        self.assertIsNotNone(predicted_mask_labels)

    @parameterized.named_parameters(
        {
            constants.UnitTestConstants.TESTCASE_NAME: 'shuffle_multi_sentence',
            constants.TokenizerConstants.INPUT_IDS: np.array(
                [
                    [
                        32128, 31999, 55,                                       # Sentence 1, 1 span mask
                        32145, 27, 31998, 169, 14145, 8585, 31997, 5,           # Sentence 2, 2 span masks
                        32143, 3963, 25, 31996,                                 # Sentence 3, 1 span mask
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],           # Padding
                    [
                        32137, 978, 31999, 233,                                 # Sentence 1, 1 span mask
                        32134, 27, 2654, 31, 31998, 68, 103, 31997,             # Sentence 2, 2 span masks
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Padding
                    [
                        32143, 31999, 3, 31998, 7142, 55,                       # Sentence 1, 2 span masks
                        32142, 31997, 19, 3, 9, 511, 5,                         # Sentence 2, 1 span mask
                        32137, 363, 31996,                                      # Sentence 3, 1 span mask
                        32130, 5933, 19, 31995,                                 # Sentence 4, 1 span mask
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                           # Padding
                    ],
                ]
            ),
            constants.DEPTHTokenizerConstants.TOKEN_TYPE_IDS: np.array(
                [
                    [
                        1, 1, 1,
                        2, 2, 2, 2, 2, 2, 2, 2,
                        3, 3, 3, 3,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    ],
                    [
                        1, 1, 1, 1,
                        2, 2, 2, 2, 2, 2, 2, 2,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    ],
                    [
                        1, 1, 1, 1, 1, 1,
                        2, 2, 2, 2, 2, 2, 2,
                        3, 3, 3,
                        4, 4, 4, 4,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    ],
                ]
            ),
            constants.UnitTestConstants.EXPECTED_INPUT_IDS: np.array(
                [
                    [
                        32128, 31999, 55,
                        32145, 27, 31998, 169, 14145, 8585, 31997, 5,
                        32143, 3963, 25, 31996,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    ],
                    [
                        32134, 27, 2654, 31, 31998, 68, 103, 31997,
                        32137, 978, 31999, 233,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    ],
                    [
                        32143, 31999, 3, 31998, 7142, 55,
                        32142, 31997, 19, 3, 9, 511, 5,
                        32130, 5933, 19, 31995,
                        32137, 363, 31996,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    ],
                ],
            ),
            constants.UnitTestConstants.EXPECTED_TOKEN_TYPE_IDS: np.array(
                [
                    [
                        1, 1, 1,
                        2, 2, 2, 2, 2, 2, 2, 2,
                        3, 3, 3, 3,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    ],
                    [
                        2, 2, 2, 2, 2, 2, 2, 2,
                        1, 1, 1, 1,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [
                        1, 1, 1, 1, 1, 1,
                        2, 2, 2, 2, 2, 2, 2,
                        4, 4, 4, 4,
                        3, 3, 3,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    ],
                ],
            ),
        }
    )
    def test_shuffle_sentences(
            self,
            input_ids,
            token_type_ids,
            expected_input_ids,
            expected_token_type_ids,
            seed=42):
        """
        Test the shuffling of sentences within the input_ids tensor.

        :param input_ids: The input_ids tensor. Shape: (batch_size, sequence_length)
        :param token_type_ids: The token_type_ids tensor. Shape: (batch_size, sequence_length). The token ID located in
         input_ids[i][j] is located in the sentence denoted by token_type_ids[i][j].
        :param expected_input_ids: The expected input_ids tensor after shuffling. Shape: (batch_size, sequence_length).
        :param expected_token_type_ids: The expected token_type_ids tensor after shuffling. Shape: (batch_size,
            sequence_length).
        :param seed: The seed for the random number generator.
        """
        self.set_seed(seed)
        for example_index in range(input_ids.shape[0]):
            sentence_ids, sentence_start_indices, sentence_lengths = np.unique(
                token_type_ids[example_index], return_counts=True, return_index=True)
            shuffled_order, shuffled_lengths, shuffled_start_indices, shuffled_token_type_ids = (
                corruption_lib.shuffle_inputs(
                    sentence_unique_ids=sentence_ids,
                    sentence_start_indices=sentence_start_indices,
                    sentence_lengths=sentence_lengths
                )
            )
            shuffled_end_indices = shuffled_start_indices + shuffled_lengths
            shuffled_input_ids = np.concatenate(
                [input_ids[example_index][start_index:end_index] for start_index, end_index in zip(
                    shuffled_start_indices, shuffled_end_indices)
                 ]
            )
            np.testing.assert_array_equal(shuffled_input_ids, expected_input_ids[example_index])
            np.testing.assert_array_equal(shuffled_token_type_ids, expected_token_type_ids[example_index])

    @parameterized.named_parameters(
        {constants.UnitTestConstants.TESTCASE_NAME: 'multiple_inputs_consisting_of_a_single_sentence',
         constants.UnitTestConstants.EXAMPLES: [
             'hello world!',
             'I am a machine learning researcher.',
             'She was born in New York, but not in the city.',
         ],
         constants.DepthDataCollatorConstants.NOISE_DENSITY: 0.5,
         constants.UnitTestConstants.SEED: 42,
         constants.DepthDataCollatorConstants.MEAN_NOISE_SPAN_LENGTH: 3,
         constants.UnitTestConstants.EXPECTED_INPUT_IDS: np.array(
             [
                 [
                     32120, 32100, 21820, 296, 55, 32120,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 ],
                 [
                     32120, 32119, 27, 183, 3, 9, 1437, 1036, 18658, 5, 32120,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 ],
                 [
                     32120, 32110, 451, 47, 2170, 16, 368, 1060, 6, 68, 59, 16, 8, 690, 5, 32120,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 ],
             ]
         ),
         constants.UnitTestConstants.EXPECTED_MODIFIED_INPUT_IDS: np.array(
             [
                 [
                     32120, 32100, 21820, 296, 32011, 32120,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 ],
                 [
                     32120, 32119, 27, 32067, 18658, 32001, 32120,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 ],
                 [
                     32120, 32110, 32004, 47, 32004, 68, 59, 16, 8, 32068, 32120,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 ],
             ]
         ),
         constants.UnitTestConstants.EXPECTED_TOKEN_TYPE_IDS: np.array(
             [
                 [
                     1, 1, 1, 1, 1, 1,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 ],
                 [
                     1, 1, 1, 1, 1, 1, 1,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 ],
                 [
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 ],
             ],
         ),
         constants.UnitTestConstants.EXPECTED_MODIFIED_LABEL_IDS: np.array(
             [
                 [
                     32120, 32100, 32011, 55, 32120,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 ],
                 [
                     32120, 32119, 32067, 183, 3, 9, 1437, 1036, 32001, 5, 32120,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 ],
                 [
                     32120, 32110, 32004, 451, 32004, 2170, 16, 368, 1060, 6, 32068, 690, 5, 32120,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 ],
             ],
         ),
         constants.UnitTestConstants.EXPECTED_LABEL_TOKEN_TYPE_IDS: np.array(
             [
                 [
                     1, 1, 1, 1, 1,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 ],
                 [
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 ],
                 [
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                 ],
             ],
         )
         },
        {constants.UnitTestConstants.TESTCASE_NAME: 'multiple_inputs_consisting_of_a_multiple_sentences',
         constants.UnitTestConstants.EXAMPLES: [EXAMPLE_1, EXAMPLE_2, EXAMPLE_3],
         constants.DepthDataCollatorConstants.NOISE_DENSITY: 0.5,
         constants.UnitTestConstants.SEED: 42,
         constants.DepthDataCollatorConstants.MEAN_NOISE_SPAN_LENGTH: 3,
         constants.UnitTestConstants.EXPECTED_INPUT_IDS: np.array(
                [
                    [
                        32120, 32100, 8774, 296, 55, 32120,
                        32117, 27, 183, 1036, 12, 169, 14145, 8585, 7, 5, 32120,
                        32115, 3963, 25, 214, 79, 33, 48, 1633, 58, 32120,
                        0, 0, 0,
                    ],
                    [
                        32120, 32119, 978, 7177, 47, 78, 13006, 233, 32120,
                        32116, 27, 2654, 31, 17, 199, 68, 103, 776, 326, 5, 32120,
                        0, 0, 0, 0, 0, 0, 0, 0, 0,
                    ],
                    [
                        32120, 32110, 947, 19, 3, 9, 166, 7142, 55, 32120,
                        32107, 100, 19, 3, 9, 511, 5, 32120,
                        32100, 363, 81, 3, 9, 1025, 58, 32120,
                        32102, 5933, 19, 631,
                    ],
                ],
         ),
         constants.UnitTestConstants.EXPECTED_MODIFIED_INPUT_IDS: np.array(
             [
                 # Masked tokens in each sentence are:
                 # Sentence 1:
                 #      32100 = [8774, 296, 55]
                 # Sentence 2:
                 #      32085 = [27, 183]
                 #      32049 = [169]
                 # Sentence 3:
                 #      32071 = [79]
                 #      32087 = [48]
                 [
                     32120, 32100, 32010, 32120,
                     32117, 32085, 1036, 12, 32049, 14145, 8585, 7, 5, 32120,
                     32115, 3963, 25, 214, 32071, 33, 32087, 1633, 58, 32120,
                     0, 0, 0, 0, 0, 0,
                 ],
                 # Masked tokens in each sentence are:
                 # Sentence 1:
                 #     32038 = [978, 7177]
                 #     32072 = [78]
                 #     32056 = [233]
                 # Sentence 2:
                 #     32038 = [27, 2654]
                 [
                     32120, 32119, 32038, 47, 32072, 13006, 32056, 32120,
                     32116, 32038, 31, 17, 199, 68, 103, 776, 326, 5, 32120,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 ],
                 # Masked tokens in each sentence are:
                 # Sentence 1:
                 #    32041 = [947]
                 #    32048 = [166, 7142, 55]
                 # Sentence 2:
                 #    32044 = [9, 511]
                 # Sentence 3:
                 #    32013 = [3, 9]
                 #    32099 = [58]
                 # Sentence 4:
                 #   32047 = [5933, 19, 631]
                 [
                     32120, 32110, 32041, 19, 3, 9, 32048, 32120,
                     32107, 100, 19, 3, 32044, 5, 32120,
                     32100, 363, 81, 32013, 1025, 32099, 32120,
                     32102, 32047,
                     0, 0, 0, 0, 0, 0,
                 ],
             ],
         ),
         constants.UnitTestConstants.EXPECTED_TOKEN_TYPE_IDS: np.array(
             [
                 [
                     1, 1, 1, 1,
                     2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                     3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                     0, 0, 0, 0, 0, 0,
                 ],
                 [
                     1, 1, 1, 1, 1, 1, 1, 1,
                     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 ],
                 [
                     1, 1, 1, 1, 1, 1, 1, 1,
                     2, 2, 2, 2, 2, 2, 2,
                     3, 3, 3, 3, 3, 3, 3,
                     4, 4,
                     0, 0, 0, 0, 0, 0,
                 ],
             ],
         ),
         constants.UnitTestConstants.EXPECTED_MODIFIED_LABEL_IDS: np.array(
             [
                 [
                     32120, 32100, 32010, 8774, 296, 55, 32120,
                     32117, 32085, 27, 183, 32049, 169, 32120,
                     32115, 32071, 79, 32087, 48, 32120,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                 ],
                 [
                     32120, 32119, 32038, 978, 7177, 32072, 78, 32056, 233, 32120,
                     32116, 32038, 27, 2654, 32120,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 ],
                 [
                     32120, 32110, 32041, 947, 32048, 166, 7142, 55, 32120,
                     32107, 32044, 9, 511, 32120,
                     32100, 32013, 3, 9, 32099, 58, 32120,
                     32102, 32047, 5933, 19, 631,
                     0, 0, 0, 0,
                 ],
             ],
         ),
         constants.UnitTestConstants.EXPECTED_LABEL_TOKEN_TYPE_IDS: np.array(
             [
                 [
                     1, 1, 1, 1, 1, 1, 1,
                     2, 2, 2, 2, 2, 2, 2,
                     3, 3, 3, 3, 3, 3,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                 ],
                 [
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     2, 2, 2, 2, 2,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 ],
                 [
                     1, 1, 1, 1, 1, 1, 1, 1, 1,
                     2, 2, 2, 2, 2,
                     3, 3, 3, 3, 3, 3, 3,
                     4, 4, 4, 4, 4,
                     0, 0, 0, 0,
                 ],
             ],
         )
         })
    def test_create_inputs_and_targets(
            self,
            examples: typing.List[str],
            noise_density: float,
            seed: int,
            mean_noise_span_length: int,
            expected_input_ids: typing.List[typing.List[int]],
            expected_modified_input_ids: typing.List[typing.List[int]],
            expected_token_type_ids: typing.List[typing.List[int]],
            expected_modified_label_ids: typing.List[typing.List[int]],
            expected_label_token_type_ids: typing.List[typing.List[int]],
            sequence_length: int = 30,
    ):
        self.set_seed(seed)
        batch_encodings = self.depth_tokenizer(
            examples,
            return_tensors=None,
            max_length=sequence_length,
            padding=transformers.tokenization_utils_base.PaddingStrategy.MAX_LENGTH,
            truncation=transformers.tokenization_utils_base.TruncationStrategy.ONLY_FIRST,
        )
        input_ids = np.array(batch_encodings[constants.DEPTHTokenizerConstants.INPUT_IDS])

        np.testing.assert_array_equal(
            input_ids,
            np.array(expected_input_ids),
            err_msg=f'Expected input IDs to be {expected_input_ids} but got {input_ids}'
        )

        token_type_ids = np.array(batch_encodings[constants.DEPTHTokenizerConstants.TOKEN_TYPE_IDS])
        batch_size, padded_sequence_length = input_ids.shape
        sequence_lengths = np.sum(np.not_equal(token_type_ids, 0).astype(np.int32), axis=1)

        span_mask = np.reshape(
            np.concatenate(
                [
                    corruption_lib.random_spans_noise_mask(
                        sequence_length=example_sequence_length,
                        maximum_length=padded_sequence_length,
                        noise_density=noise_density,
                        mean_noise_span_length=mean_noise_span_length)
                    for example_sequence_length in sequence_lengths]
            ),
            newshape=[batch_size, padded_sequence_length],
        )

        # Shift the span mask by two in order to account for the initial end of sentence and start of sentence tokens.
        span_mask = np.concatenate(
            [np.zeros([batch_size, 2], dtype=bool),
             np.array(span_mask[:, :-2], dtype=bool)],
            axis=1,
        )

        # Identify special tokens.
        special_tokens = self.depth_tokenizer.all_special_ids
        sentence_token_ids = self.depth_tokenizer.get_sentence_token_and_eosen_ids()

        # Ensure mask is only applied to non-special tokens.
        augmented_input_span_mask = np.where(np.isin(input_ids, special_tokens, invert=True), span_mask, False)

        # Create a sentinel mask, where 0s indicate a lack of mask, positive values indicate the start of a masked span,
        #  and -1 indicates the continuation of a masked span.
        input_ids_sentinel = corruption_lib.create_sentinel_ids_for_depth(
            tokenizer=self.depth_tokenizer,
            mask_indices=augmented_input_span_mask.astype(np.int8),
        )

        modified_input_ids, modified_input_token_type_ids, modified_label_ids, modified_label_token_type_ids = (
            corruption_lib.create_model_input_for_corrupted_batch(
                input_ids=input_ids,
                input_ids_sentinel=input_ids_sentinel,
                token_type_ids=token_type_ids,
                batch_size=batch_size,
                sequence_lengths=sequence_lengths,
                padded_sequence_length=padded_sequence_length,
                sentence_token_ids=sentence_token_ids,
            )
        )

        np.testing.assert_array_equal(
            modified_input_ids,
            np.array(expected_modified_input_ids),
            err_msg=f'Expected modified input IDs to be {expected_modified_input_ids} but got {modified_input_ids}'
        )
        np.testing.assert_array_equal(
            modified_input_token_type_ids,
            np.array(expected_token_type_ids),
            err_msg=f'Expected modified token type IDs to be {expected_token_type_ids} '
                    f'but got {modified_input_token_type_ids}'
        )
        np.testing.assert_array_equal(
            modified_label_ids,
            np.array(expected_modified_label_ids),
            err_msg=f'Expected modified label IDs to be {expected_modified_label_ids} but got {modified_label_ids}'
        )
        np.testing.assert_array_equal(
            modified_label_token_type_ids,
            np.array(expected_label_token_type_ids),
            err_msg=f'Expected modified label token type IDs to be {expected_label_token_type_ids} '
                    f'but got {modified_label_token_type_ids}'
        )


class TestCreateAttentionMask(parameterized.TestCase):

    @staticmethod
    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        transformers.set_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setUp(self):
        self.tokenizer = DepthTokenizer.from_pretrained(
            pretrained_model_name_or_path=constants.ModelHuggingFaceName.T5_BASE.value,
            max_num_sentences_in_text=constants.DEPTHTokenizerConstants.NUM_SENT_TOKENS,
        )

    @parameterized.named_parameters(
        {
            constants.UnitTestConstants.TESTCASE_NAME: "Test single sentence no padding",
            constants.TokenizerConstants.INPUT_IDS: [
                ["<eosen>", "<sent_17>", "hello", "<extra_id_82>", "<eosen>", "<eos>"]
            ],
            constants.DepthDataCollatorConstants.INPUT_TOKEN_TYPE_IDS: [[1, 1, 1, 1, 1, 1]],
            constants.DepthDataCollatorConstants.TARGET_IDS: [
                ["<pad>", "<eosen>", "<sent_17>", "<extra_id_82>", "world", "<eosen>",]
            ],
            constants.DepthDataCollatorConstants.TARGET_TOKEN_TYPE_IDS: [[1, 1, 1, 1, 1, 1]],
            constants.UnitTestConstants.EXPECTED_ENCODER_SELF_ATTENTION_MASK: [
                [
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],  # sentence token in the input
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                ],
            ],
            constants.UnitTestConstants.EXPECTED_DECODER_SELF_ATTENTION_MASK: [
                [
                    [1, 0, 0, 0, 0, 0],  # Padding token
                    [1, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],  # sentence token in target attending to sentence token in input
                    [1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1],
                ],
            ],
            constants.UnitTestConstants.EXPECTED_CROSS_ATTENTION_MASK: [
                [
                    [1, 1, 1, 1, 1, 1],  # Padding token
                    [1, 1, 1, 1, 1, 1],
                    [0, 1, 0, 0, 0, 0],  # sentence token in target attending to sentence token in input
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                ],
            ],
        },
        {
            constants.UnitTestConstants.TESTCASE_NAME: "Test multiple sentences no padding",
            constants.TokenizerConstants.INPUT_IDS: [
                [
                    "<eosen>", "<sent_17>", "hello", "<extra_id_82>", "<eosen>",
                    "<sent_9>", "<extra_id_13>", "are", "you?", "<eosen>", "<eos>"
                ]
            ],
            constants.DepthDataCollatorConstants.INPUT_TOKEN_TYPE_IDS: [[1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]],
            constants.DepthDataCollatorConstants.TARGET_IDS: [
                [
                    "<pad>", "<eosen>", "<sent_17>", "<extra_id_82>", "world", "<eosen>",
                    "<sent_9>", "<extra_id_13>", "how", "<eosen>",
                ]
            ],
            constants.DepthDataCollatorConstants.TARGET_TOKEN_TYPE_IDS: [[0, 1, 1, 1, 1, 1, 2, 2, 2, 2]],
            # 11 x 11 attention mask corresponding to the 11 tokens in the input sequence
            constants.UnitTestConstants.EXPECTED_ENCODER_SELF_ATTENTION_MASK: [
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # sentence token #17 attending to tokens in sentence 1
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],  # sentence token #9 attending to tokens in sentence 2
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ],
            ],
            # 10 x 10 attention mask corresponding to the 10 tokens in the target sequence
            constants.UnitTestConstants.EXPECTED_DECODER_SELF_ATTENTION_MASK: [
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # sentence token #17 attending to previous sentence tokens in target
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0], # sentence token #9 attending to previous sentence tokens in target
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ],
            ],
            # 10 x 11 attention mask corresponding to the 10 tokens in the target sequence attending to 11 tokens in the
            # input sequence
            constants.UnitTestConstants.EXPECTED_CROSS_ATTENTION_MASK: [
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # sentence token #17 attending to sentence tokens in input
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # sentence token #9 attending to sentence tokens in input
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ],
            ],
        },
        {
            constants.UnitTestConstants.TESTCASE_NAME: "Test single sentence with padding",
            constants.TokenizerConstants.INPUT_IDS: [
                [
                    "<eosen>", "<sent_17>", "hello", "<extra_id_82>", "<eosen>", "<eos>",
                    "<pad>", "<pad>", "<pad>", "<pad>", "<pad>",
                ]
            ],
            constants.DepthDataCollatorConstants.INPUT_TOKEN_TYPE_IDS: [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]],
            constants.DepthDataCollatorConstants.TARGET_IDS: [
                [
                    "<pad>", "<eosen>", "<sent_17>", "<extra_id_82>", "world", "<eosen>", "<eos>",
                    "<pad>", "<pad>", "<pad>",
                ]
            ],
            constants.DepthDataCollatorConstants.TARGET_TOKEN_TYPE_IDS: [[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]],
            # 11 x 11 attention mask corresponding to the 10 tokens in the input sequence
            constants.UnitTestConstants.EXPECTED_ENCODER_SELF_ATTENTION_MASK: [
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # End of sentence token
                    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # Sentence token
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # End Of Sequence (<EOS>) token
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Pad
                ],
            ],
            # 10 x 10 attention mask corresponding to the 10 (of which 3 are padding) tokens in the target sequence
            constants.UnitTestConstants.EXPECTED_DECODER_SELF_ATTENTION_MASK: [
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Pad
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # End of sentence (<EOSEN>) token
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Sentence token in target
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # End of sentence (<EOSEN>) token
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],  # End of sequence (<EOS>) token
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # Pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # Pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Pad
                 ],
            ],
            # 10 x 11 attention mask corresponding to the 10 tokens in the target sequence attending to 11 tokens in the
            # input sequence
            constants.UnitTestConstants.EXPECTED_CROSS_ATTENTION_MASK: [
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ],
            ],
        },
        {
            constants.UnitTestConstants.TESTCASE_NAME: "Test multiple sentences with padding",
            constants.TokenizerConstants.INPUT_IDS: [
                [
                    "<eosen>",
                    "<sent_17>", "hello", "<extra_id_82>", "<eosen>",
                    "<sent_9>", "<extra_id_13>", "are", "you?", "<eosen>", "<eos>",
                    "<pad>", "<pad>", "<pad>", "<pad>",
                ]
            ],
            # Of length 11 + 4 = 15
            constants.DepthDataCollatorConstants.INPUT_TOKEN_TYPE_IDS: [
                [
                    1,
                    1, 1, 1, 1,
                    2, 2, 2, 2, 2, 2,
                    0, 0, 0, 0,
                ],
            ],
            constants.DepthDataCollatorConstants.TARGET_IDS: [
                [
                    "<pad>", "<eosen>",
                    "<sent_17>", "<extra_id_82>", "world", "<eosen>",
                    "<sent_9>", "<extra_id_13>", "hello", "<eosen>", "<eos>",
                    "<pad>", "<pad>",
                ],
            ],
            # Of length 11 + 2 = 13
            constants.DepthDataCollatorConstants.TARGET_TOKEN_TYPE_IDS: [
                [
                    0, 0,
                    1, 1, 1, 1,
                    2, 2, 2, 2, 2,
                    0, 0,
                ],
            ],
            # 15 x 15 attention mask corresponding to the 15 tokens in the input sequence (including padding) attending
            # to 15 tokens in the input sequence (including padding).
            constants.UnitTestConstants.EXPECTED_ENCODER_SELF_ATTENTION_MASK: [
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # <eosen>
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # <sent_17> attending to tokens in sentence 1
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # <sent_9> attending to tokens in sentence 2
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # <eosen>
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # <eos>
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Pad
                ],
            ],
            # 13 x 15 attention mask corresponding to 13 tokens in the target sequence attending to 15 tokens in the
            # input sequence
            constants.UnitTestConstants.EXPECTED_CROSS_ATTENTION_MASK: [
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # <eosen>
                    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # <sent_17> attending to input sentence tokens
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # <sent_9> attending to input sentence tokens
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # <eosen>
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # <eos>
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Pad
                ],
            ],
            # 13 x 13 attention mask corresponding to 13 tokens in the target sequence attending to 13 tokens in the
            # target sequence
            constants.UnitTestConstants.EXPECTED_DECODER_SELF_ATTENTION_MASK: [
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # <sent_17> attending previous sentence tokens in target
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # <sent_9> attending previous sentence tokens in target
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],  # <eosen>
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # <eos>
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # Pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Pad
                ],
            ],
        },
    )
    def test_create_attention_mask(
            self,
            input_ids: typing.List[typing.List[str]],
            input_token_type_ids: typing.List[typing.List[int]],
            target_ids: typing.List[typing.List[str]],
            target_token_type_ids: typing.List[typing.List[int]],
            expected_encoder_self_attention_mask: typing.List[typing.List[int]],
            expected_cross_attention_mask: typing.List[typing.List[int]],
            expected_decoder_self_attention_mask: typing.List[typing.List[int]],
    ):
        """Test the `create_depth_attention_masks` function with various inputs.
        :param input_ids: An integer tensor of shape [batch_size, sequence_length] corresponding to input token IDs.
        :param input_token_type_ids: An integer tensor of shape [batch_size, sequence_length] corresponding to sentences
            of each token within input_ids. The token at input_ids[i, j] belongs to sentence input_token_type_ids[i, j].
            Padding tokens have a token type ID of 0.
        :param target_ids: An integer tensor of shape [batch_size, sequence_length] corresponding to target token IDs.
        :param target_token_type_ids: An integer tensor of shape [batch_size, sequence_length] corresponding to
            sentences of each token within target_ids. The token at target_ids[i, j] belongs to sentence
            target_token_type_ids[i, j]. Padding tokens have a token type ID of 0.
        :param expected_encoder_self_attention_mask: An integer tensor of shape
            [batch_size, sequence_length, sequence_length] corresponding to the expected encoder self attention mask.
        :param expected_cross_attention_mask: An integer tensor of shape
            [batch_size, sequence_length, sequence_length] corresponding to the expected cross attention mask.
        :param expected_decoder_self_attention_mask: An integer tensor of shape
            [batch_size, sequence_length, sequence_length] corresponding to the expected decoder self attention mask.
        """
        # TODO: Create additional unit tests with a batch size >1
        tokenized_input_ids = np.array(
            [self.tokenizer.convert_tokens_to_ids(tokens)
            for tokens in input_ids]
        )
        tokenized_target_ids = np.array(
            [self.tokenizer.convert_tokens_to_ids(tokens)
            for tokens in target_ids]
        )
        input_token_type_ids = np.array(input_token_type_ids)
        target_token_type_ids = np.array(target_token_type_ids)

        encoder_self_attention_mask, cross_attention_mask, decoder_self_attention_mask = (
            corruption_lib.create_depth_attention_masks(
                tokenized_input_ids,
                tokenized_target_ids,
                input_token_type_ids,
                target_token_type_ids,
                self.tokenizer,
            )
        )
        np.testing.assert_almost_equal(
            actual=encoder_self_attention_mask,
            desired=np.array(expected_encoder_self_attention_mask),
            err_msg=f'Expected encoder self attention mask to be {expected_encoder_self_attention_mask}\n'
                    f'Instead got {encoder_self_attention_mask.tolist()}'
        )
        np.testing.assert_almost_equal(
            actual=cross_attention_mask,
            desired=np.array(expected_cross_attention_mask),
            err_msg=f'Expected cross attention mask to be {expected_cross_attention_mask}\n'
                    f'Instead got {cross_attention_mask.tolist()}'
        )
        np.testing.assert_almost_equal(
            actual=decoder_self_attention_mask,
            desired=np.array(expected_decoder_self_attention_mask),
            err_msg=f'Expected decoder self attention mask to be {expected_decoder_self_attention_mask}\n'
                    f'Instead got {decoder_self_attention_mask.tolist()}'
        )

class TestDepthCorruption(parameterized.TestCase):

    @staticmethod
    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        transformers.set_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setUp(self):
        self.tokenizer = DepthTokenizer.from_pretrained(
            pretrained_model_name_or_path=constants.ModelHuggingFaceName.T5_BASE.value,
            max_num_sentences_in_text=constants.DEPTHTokenizerConstants.NUM_SENT_TOKENS,
        )

    @parameterized.named_parameters(
        {
            constants.UnitTestConstants.TESTCASE_NAME: 'three examples with single short sentences',
            constants.UnitTestConstants.EXAMPLES: [
                'This is a test sentence.',
                'Here is a much longer test sentence than the first one.',
                'And indeed, this is the longest test sentence of them all, as it far exceeds the norm.',
            ],
            constants.DepthDataCollatorConstants.NOISE_DENSITY: 0.3,
            constants.DepthDataCollatorConstants.MEAN_NOISE_SPAN_LENGTH: 3.0,
            constants.UnitTestConstants.DO_SHUFFLE: False,
            constants.UnitTestConstants.EXPECTED_INPUT_IDS: [
                [
                    32120, 32100, 100, 19, 3, 9, 794, 7142, 5, 32120, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ],
                [
                    32120, 32119, 947, 19, 3, 9, 231, 1200, 794, 7142, 145, 8, 166, 80, 5, 32120, 1, 0, 0, 0,
                ],
                [
                    32120, 32110, 275, 5071, 6, 48, 19, 8, 14783, 794, 7142, 13, 135, 66, 6, 38, 34, 623, 8193, 7,
                ],
            ],
            constants.UnitTestConstants.EXPECTED_MODIFIED_INPUT_IDS: [
                # Sentence 1:
                #   - 32037 = [7142, 5]
                [
                    32120, 32100, 100, 19, 3, 9, 794, 32037, 32120, 1,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ],
                # Sentence 1:
                #   - 32035 = [80, 5]
                [
                    32120, 32119, 947, 19, 3, 9, 231, 1200, 794, 7142, 145, 8, 166, 32035, 32120, 1,
                    0, 0, 0, 0,
                ],
                # Sentence 1:
                #  - 32091 = [5071, 6]
                #  - 32049 = [794, 7142, 13, 135]
                [
                    32120, 32110, 275, 32091, 48, 19, 8, 14783, 32049, 66, 6, 38, 34, 623, 8193, 7,
                    0, 0, 0, 0,
                ],
            ],
            constants.UnitTestConstants.EXPECTED_MODIFIED_LABEL_IDS: [
                [
                    32120, 32100, 32037, 7142, 5, 32120,
                    -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                ],
                [
                    32120, 32119, 32035, 80, 5, 32120,
                    -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                ],
                [
                    32120, 32110, 32091, 5071, 6, 32049, 794, 7142, 13, 135,
                    -100, -100, -100, -100, -100, -100, -100, -100, -100, -100
                ],
            ],
            constants.UnitTestConstants.EXPECTED_TARGET_IDS: [
                [
                    0, 32120, 32100, 32037, 7142, 5, 32120,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ],
                [
                    0, 32120, 32119, 32035, 80, 5, 32120,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ],
                [
                    0, 32120, 32110, 32091, 5071, 6, 32049, 794, 7142, 13, 135,
                    0, 0, 0, 0, 0, 0, 0, 0, 0,
                ],
            ],
            constants.UnitTestConstants.EXPECTED_ENCODER_SELF_ATTENTION_MASK: [
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                ],
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ], [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ],
            ],
            constants.UnitTestConstants.EXPECTED_DECODER_SELF_ATTENTION_MASK: [
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ],
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ],
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ],
            ],
            constants.UnitTestConstants.EXPECTED_CROSS_ATTENTION_MASK: [
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ],
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ],
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ],
            ],
            constants.UnitTestConstants.EXPECTED_LENGTH: 64,
            constants.UnitTestConstants.EXPECTED_IS_SHUFFLED: False,
        },
        {
            constants.UnitTestConstants.TESTCASE_NAME: "Multiple shuffled sentences with padding and truncation",
            constants.UnitTestConstants.EXAMPLES: [
                'First sentence. A second one. My third sentence. Final sentence.',
                'This is a sentence. Another one. And another.',
                'Hi Chris! I am Zach. It is is a pleasure to meet you. I am a big fan of your work.',
            ],
            constants.DepthDataCollatorConstants.NOISE_DENSITY: 0.3,
            constants.DepthDataCollatorConstants.MEAN_NOISE_SPAN_LENGTH: 3,
            constants.UnitTestConstants.DO_SHUFFLE: True,
            constants.TokenizerConstants.INPUT_LENGTH: 30,
            constants.TokenizerConstants.TARGET_LENGTH: 30,
            # Original order of sentence IDs:
            #
            #   - example 1: 32100, 32117, 32115, 32101
            #   - example 2: 32119, 32116, 32115
            #   - example 3: 32110, 32107, 32100, 32102
            # Shuffled order of sentence IDs:
            #
            #   - example 1: 32101, 32115, 32117, 32100
            #   - example 2: 32115, 32119, 32116,
            #   - example 3: 32107, 32100, 32102, 32110
            constants.UnitTestConstants.EXPECTED_INPUT_IDS: [
                # Padded example (4 sentences)
                [
                    32120,
                    32100, 1485, 7142, 5, 32120,
                    32117, 71, 511, 80, 5, 32120,
                    32115, 499, 1025, 7142, 5, 32120,
                    32101, 6514, 7142, 5, 32120, 1,
                    0, 0, 0, 0, 0, 0,
                ],
                # Padded example (3 sentences)
                [
                    32120,
                    32119, 100, 19, 3, 9, 7142, 5, 32120,
                    32116, 2351, 80, 5, 32120,
                    32115, 275, 430, 5, 32120, 1,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ],
                # Truncated example (4 sentences)
                [
                    32120,
                    32110, 2018, 4409, 55, 32120,
                    32107, 27, 183, 22045, 5, 32120,
                    32100, 94, 19, 19, 3, 9, 5565, 12, 942, 25, 5, 32120,
                    32102, 27, 183, 3, 9, 600
                ],
            ],
            constants.UnitTestConstants.EXPECTED_MODIFIED_INPUT_IDS: [
                # Sentence 1 (32101):
                #  - 32010 = [5]
                # Sentence 2 (32115):
                #  No corruption
                # Sentence 3 (32117):
                #  No corruption
                # Sentence 4 (32100):
                # - 32063 = [1485, 7142, 5]
                [
                    32120,
                    32101, 6514, 7142, 32010, 32120,
                    32115, 499, 1025, 7142, 5, 32120,
                    32117, 71, 511, 80, 5, 32120,
                    32100, 32063, 32120,
                    1,
                    0, 0, 0, 0, 0, 0, 0, 0
                ],
                # Sentence 1 (32115):
                #  No corruption
                # Sentence 2 (32119):
                #  - 32057 = [19, 3]
                # Sentence 3 (32116):
                #  - 32014 = [2351, 80, 5]
                [
                    32120,
                    32115, 275, 430, 5, 32120,
                    32119, 100, 32057, 9, 7142, 5, 32120,
                    32116, 32014, 32120,
                    1,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ],
                # Sentence 1 (32107):
                #  No corruption
                # Sentence 2 (32100):
                #  - 32004 = [9]
                # Sentence 3 (32102):
                #  - 32041 = [600]
                # Sentence 4 (32110):
                #  - 32023 = [2018, 4409, 55]
                [
                    32120,
                    32107, 27, 183, 22045, 5, 32120,
                    32100, 94, 19, 19, 3, 32004, 5565, 12, 942, 25, 5, 32120,
                    32102, 27, 183, 3, 9, 32041, 32120,
                    32110, 32023, 32120, 1,
                ]
            ],
            constants.UnitTestConstants.EXPECTED_MODIFIED_LABEL_IDS: [
                [
                    32120,
                    32100, 32063, 1485, 7142, 5, 32120,
                    32117, 32120,
                    32115, 32120,
                    32101, 32010, 5, 32120,
                    -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                ],
                [
                    32120,
                    32119, 32057, 19, 3, 32120,
                    32116, 32014, 2351, 80, 5, 32120,
                    32115, 32120,
                    -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                ],
                [
                    32120,
                    32110, 32023, 2018, 4409, 55, 32120,
                    32107, 32120,
                    32100, 32004, 9, 32120,
                    32102, 32041, 600,
                    -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                ],
            ],
            constants.UnitTestConstants.EXPECTED_TARGET_IDS: [
                [
                    0, 32120,
                    32100, 32063, 1485, 7142, 5, 32120,
                    32117, 32120, 32115, 32120,
                    32101, 32010, 5, 32120,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ],
                [
                    0, 32120,
                    32119, 32057, 19, 3, 32120,
                    32116, 32014, 2351, 80, 5, 32120,
                    32115, 32120,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ],
                [
                    0, 32120,
                    32110, 32023, 2018, 4409, 55, 32120,
                    32107, 32120,
                    32100, 32004, 9, 32120,
                    32102, 32041, 600,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ],
            ],
            constants.UnitTestConstants.EXPECTED_ENCODER_SELF_ATTENTION_MASK: [
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sent 1
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sent 2
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sent 3
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], # sent 4
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # pad
                ],
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sent 1
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sent 2
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sent 3
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # pad
                ],
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sent 1
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sent 2
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], # sent 3
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], # sent 4
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ],
            ],
            constants.UnitTestConstants.EXPECTED_DECODER_SELF_ATTENTION_MASK: [
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sent 1
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sent 2
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sent 3
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sent 4
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # pad
                ],
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sent 1
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sent 2
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sent 3
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0 ,0 ,0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # pad
                ],
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sent 1
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sent 2
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sent 3
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sent 4
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # pad
                ],
            ],
            constants.UnitTestConstants.EXPECTED_CROSS_ATTENTION_MASK: [
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # eosen
                    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sent 1
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sent 2
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sent 3
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sent 4
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ],
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # eosen
                    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sent 1
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sent 2
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sent 3
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # eosen
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # pad
                ],
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # eosen
                    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # sent 1
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # sent 2
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # sent 3
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # sent 4
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # pad
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # pad
                ],
            ],
            constants.UnitTestConstants.EXPECTED_LENGTH: 114,
            constants.UnitTestConstants.EXPECTED_IS_SHUFFLED: True,
        }
    )
    def test_corrupt_for_depth(
            self,
            examples: typing.List[str],
            noise_density: float,
            mean_noise_span_length: int,
            do_shuffle: bool,
            expected_input_ids: typing.List[typing.List[int]],
            expected_modified_input_ids: typing.List[typing.List[int]],
            expected_modified_label_ids: typing.List[typing.List[int]],
            expected_target_ids: typing.List[typing.List[int]],
            expected_encoder_self_attention_mask: typing.List[typing.List[typing.List[int]]],
            expected_decoder_self_attention_mask: typing.List[typing.List[typing.List[int]]],
            expected_cross_attention_mask: typing.List[typing.List[typing.List[int]]],
            expected_length: int,
            expected_is_shuffled: bool,
            input_length: int = 20,
            target_length: int = 20,
            seed: int = 42,
    ):
        """
        Test the corrupt_for_depth method from the corruption module.

        Corruption involves applying T5-esque span corruption, as well as shuffling sentences in the input.
        Note that Depth's corruption scheme involves creating designed "sentence" and "end of sentence" tokens.
        Sentence tokens in the input attend strictly to tokens in their own sentence. In the decoder,
        sentence tokens can only attend to previous sentence tokens (i.e., those that appeared in the encoder, or
        earlier in the decoder).

        :param examples: A list of string examples to tokenize and corrupt.
        :param noise_density: The density of noise to add to the input. This is a float between 0 and 1, where 0 means
            no noise and 1 means all noise.
        :param mean_noise_span_length: The mean length of noise spans to add to the input.
        :param do_shuffle: Whether to shuffle the input.
        :param expected_input_ids: The expected input IDs after tokenization, and before corruption.
        :param expected_modified_input_ids: The expected input IDs after corruption.
        :param expected_modified_label_ids: The expected label IDs after corruption.
        :param expected_target_ids: The expected target IDs after corruption.
        :param expected_encoder_self_attention_mask: The expected encoder self-attention mask after span corruption and
            shuffling.
        :param expected_decoder_self_attention_mask: The expected decoder self-attention mask after span corruption and
            shuffling.
        :param expected_cross_attention_mask: The expected cross-attention mask after corruption.
        :param expected_length: The expected length of the corrupted batch. This accounts for both the input and target
            lengths, not including padding.
        :param expected_is_shuffled: Whether the corrupted batch is expected to be shuffled.
        :param input_length: The maximum length of the input.
        :param target_length: The maximum length of the target.
        :param seed: The random seed to use for reproducibility.
        """
        self.set_seed(seed)
        tokenized_examples = self.tokenizer(
            examples,
            max_length=input_length,
            padding=True,
            truncation=True,
            return_tensors='np',
        )
        np.testing.assert_array_equal(
            tokenized_examples[constants.TokenizerConstants.INPUT_IDS],
            np.array(expected_input_ids),
            err_msg=f'Input IDs are not as expected.\n'
                    f'Expected: {expected_input_ids}\n'
                    f'Actual: {tokenized_examples[constants.TokenizerConstants.INPUT_IDS]}'
        )
        np_batch_encoding = transformers.BatchEncoding(
            {
                constants.TokenizerConstants.INPUT_IDS: np.array(
                    tokenized_examples[constants.TokenizerConstants.INPUT_IDS]
                ),
                constants.TokenizerConstants.TOKEN_TYPE_IDS: np.array(
                    tokenized_examples[constants.TokenizerConstants.TOKEN_TYPE_IDS]
                ),
            }
        )
        corrupted_batch = corruption_lib.corrupt_for_depth(
            examples=np_batch_encoding,
            tokenizer=self.tokenizer,
            pad_token_id=self.tokenizer.pad_token_id,
            decoder_start_token_id=self.tokenizer.pad_token_id,
            noise_density=noise_density,
            mean_noise_span_length=mean_noise_span_length,
            do_shuffle=do_shuffle,
            input_length=input_length,
            target_length=target_length,
        )
        input_ids = corrupted_batch[constants.TokenizerConstants.INPUT_IDS]
        label_ids = corrupted_batch[constants.TokenizerConstants.LABELS]
        target_ids = corrupted_batch[constants.DepthDataCollatorConstants.TARGET_IDS]
        encoder_self_attention_mask = corrupted_batch[constants.DepthDataCollatorConstants.ENCODER_ATTENTION_MASK]
        decoder_self_attention_mask = corrupted_batch[constants.DepthDataCollatorConstants.DECODER_ATTENTION_MASK]
        cross_attention_mask = corrupted_batch[constants.DepthDataCollatorConstants.CROSS_ATTENTION_MASK]

        np.testing.assert_array_equal(
            input_ids,
            np.array(expected_modified_input_ids),
            err_msg=f'Corrupted input IDs are not as expected.\n'
                    f'Expected: {expected_modified_input_ids}\n'
                    f'Actual: {input_ids.tolist()}'
        )
        np.testing.assert_array_equal(
            label_ids,
            np.array(expected_modified_label_ids),
            err_msg=f'Corrupted target IDs are not as expected.\n'
                    f'Expected: {expected_modified_label_ids}\n'
                    f'Actual: {label_ids.tolist()}'
        )
        np.testing.assert_array_equal(
            target_ids,
            np.array(expected_target_ids),
            err_msg=f'Target IDs are not as expected.\n'
                    f'Expected: {expected_target_ids}\n'
                    f'Actual: {target_ids.tolist()}'
        )
        np.testing.assert_array_equal(
            encoder_self_attention_mask,
            np.array(expected_encoder_self_attention_mask),
            err_msg=f'Encoder self-attention mask is not as expected.\n'
                    f'Expected: {expected_encoder_self_attention_mask}\n'
                    f'Actual: {encoder_self_attention_mask.tolist()}'
        )
        np.testing.assert_array_equal(
            decoder_self_attention_mask,
            np.array(expected_decoder_self_attention_mask),
            err_msg=f'Decoder self-attention mask is not as expected.\n'
                    f'Expected: {expected_decoder_self_attention_mask}\n'
                    f'Actual: {decoder_self_attention_mask.tolist()}'
        )
        np.testing.assert_array_equal(
            cross_attention_mask,
            np.array(expected_cross_attention_mask),
            err_msg=f'Cross-attention mask is not as expected.\n'
                    f'Expected: {expected_cross_attention_mask}\n'
                    f'Actual: {cross_attention_mask.tolist()}'
        )
        # Length has shape (batch_size, 1)
        self.assertEqual(
            corrupted_batch[constants.DepthDataCollatorConstants.LENGTH][0][0],
            expected_length,
            f'Length is not as expected.\n'
            f'Expected: {expected_length}\n'
            f'Actual: {corrupted_batch[constants.DepthDataCollatorConstants.LENGTH][0][0]}'
        )
        # Is_shuffled has shape (batch_size, 1)
        self.assertEqual(
            corrupted_batch[constants.DepthDataCollatorConstants.IS_SHUFFLED][0][0],
            expected_is_shuffled,
            f'Is shuffled is not as expected.\n'
            f'Expected: {expected_is_shuffled}\n'
            f'Actual: {corrupted_batch[constants.DepthDataCollatorConstants.IS_SHUFFLED][0][0]}'
        )


if __name__ == '__main__':
    unittest.main()

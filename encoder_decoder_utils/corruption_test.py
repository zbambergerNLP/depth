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
        np.random.seed(seed)
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
         constants.UnitTestConstants.EXPECTED_MODIFIED_INPUT_IDS: np.array(
             [
                 [
                     32120, 32100, 32010, 32120,
                     32117, 32085, 1036, 12, 32049, 14145, 8585, 7, 5, 32120,
                     32115, 3963, 25, 214, 32071, 33, 32087, 1633, 58, 32120,
                     0, 0, 0, 0, 0, 0,
                 ],
                 [
                     32120, 32119, 32038, 47, 32072, 13006, 32056, 32120,
                     32116, 32038, 31, 17, 199, 68, 103, 776, 326, 5, 32120,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 ],
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
        sentence_tokens = list(
            filter(
                lambda token: f'<{constants.DEPTHTokenizerConstants.SENT}' in token,   # Function
                self.depth_tokenizer.all_special_tokens                                # Iterable
            )
        )
        sentence_tokens.append(constants.DEPTHTokenizerConstants.END_OF_SENTENCE_TOKEN)
        sentence_token_ids = self.depth_tokenizer.convert_tokens_to_ids(sentence_tokens)

        # Ensure mask is only applied to non-special tokens.
        augmented_input_span_mask = np.where(np.isin(input_ids, special_tokens, invert=True), span_mask, False)

        # Create a sentinel mask, where 0s indicate a lack of mask, positive values indicate the start of a masked span,
        #  and -1 indicates the continuation of a masked span.
        input_ids_sentinel = corruption_lib.create_sentinel_ids(
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

    def set_seed(self, seed: int):
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
            constants.UnitTestConstants.TESTCASE_NAME: "TestSingleSentenceNoPadding",
            constants.TokenizerConstants.INPUT_IDS: [
                ["<eosen>", "<sent_17>", "hello", "<extra_id_82>", "<eosen>",]
            ],
            constants.DepthDataCollatorConstants.INPUT_TOKEN_TYPE_IDS: [[1, 1, 1, 1, 1]],
            constants.DepthDataCollatorConstants.TARGET_IDS: [
                ["<eosen>", "<sent_17>", "<extra_id_82>", "world", "<eosen>",]
            ],
            constants.DepthDataCollatorConstants.TARGET_TOKEN_TYPE_IDS: [[1, 1, 1, 1, 1]],
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
                    [0, 1, 0, 0, 0],  # sentence token in target
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                ],
            ],
            constants.UnitTestConstants.EXPECTED_CROSS_ATTENTION_MASK: [
                [
                    [1, 1, 1, 1, 1],
                    [0, 1, 0, 0, 0],  # sentence token in target
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                ],
            ],
        },
        {
            constants.UnitTestConstants.TESTCASE_NAME: "TestMultipleSentencesNoPadding",
            constants.TokenizerConstants.INPUT_IDS: [
                [
                    "<eosen>", "<sent_17>", "hello", "<extra_id_82>", "<eosen>",
                    "<sent_9>", "<extra_id_13>", "are", "you?", "<eosen>",
                ]
            ],
            constants.DepthDataCollatorConstants.INPUT_TOKEN_TYPE_IDS: [[1, 1, 1, 1, 1, 2, 2, 2, 2, 2]],
            constants.DepthDataCollatorConstants.TARGET_IDS: [
                [
                    "<eosen>", "<sent_17>", "<extra_id_82>", "world", "<eosen>",
                    "<sent_9>", "<extra_id_13>", "how", "<eosen>",
                ]
            ],
            constants.DepthDataCollatorConstants.TARGET_TOKEN_TYPE_IDS: [[1, 1, 1, 1, 1, 2, 2, 2, 2]],
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
                    [1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0],  # sentence token #1
                    [1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0, 0, 0],  # sentence token #2
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
                    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # sentence token #1
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # sentence token #2
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ],
            ],
        },
        {
            constants.UnitTestConstants.TESTCASE_NAME: "TestSingleSentenceWithPadding",
            constants.TokenizerConstants.INPUT_IDS: [
                [
                    "<eosen>", "<sent_17>", "hello", "<extra_id_82>", "<eosen>",
                    "<pad>", "<pad>", "<pad>", "<pad>", "<pad>",
                ]
            ],
            constants.DepthDataCollatorConstants.INPUT_TOKEN_TYPE_IDS: [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]],
            constants.DepthDataCollatorConstants.TARGET_IDS: [
                [
                    "<eosen>", "<sent_17>", "<extra_id_82>", "world", "<eosen>",
                    "<pad>", "<pad>", "<pad>", "<pad>",
                ]
            ],
            constants.DepthDataCollatorConstants.TARGET_TOKEN_TYPE_IDS: [[1, 1, 1, 1, 1, 0, 0, 0, 0]],
            # 10 x 10 attention mask corresponding to the 10 tokens in the input sequence
            constants.UnitTestConstants.EXPECTED_ENCODER_SELF_ATTENTION_MASK: [
                [
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Pad
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Pad
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # Pad
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Pad
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Pad
                ],
            ],
            # 9 x 9 attention mask corresponding to the 9 (of which 4 are padding) tokens in the target sequence
            constants.UnitTestConstants.EXPECTED_DECODER_SELF_ATTENTION_MASK: [
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0],  # Pad
                    [0, 0, 0, 0, 0, 0, 1, 0, 0],  # Pad
                    [0, 0, 0, 0, 0, 0, 0, 1, 0],  # Pad
                    [0, 0, 0, 0, 0, 0, 0, 0, 1],  # Pad
                 ],
            ],
            # 9 x 10 attention mask corresponding to the 9 tokens in the target sequence attending to 10 tokens in the
            # input sequence
            constants.UnitTestConstants.EXPECTED_CROSS_ATTENTION_MASK: [
                [
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
            ],
        },
        {
            constants.UnitTestConstants.TESTCASE_NAME: "TestMultipleSentencesWithPadding",
            constants.TokenizerConstants.INPUT_IDS: [
                [
                    "<eosen>", "<sent_17>", "hello", "<extra_id_82>", "<eosen>",
                "<sent_9>", "<extra_id_13>", "are", "you?", "<eosen>",
                "<pad>", "<pad>", "<pad>", "<pad>", "<pad>",
                ]
            ],
            # Of length 10 + 5 = 15
            constants.DepthDataCollatorConstants.INPUT_TOKEN_TYPE_IDS: [
                [
                    1, 1, 1, 1, 1,
                    2, 2, 2, 2, 2,
                    0, 0, 0, 0, 0,
                ],
            ],
            constants.DepthDataCollatorConstants.TARGET_IDS: [
                [
                    "<eosen>", "<sent_17>", "<extra_id_82>", "world", "<eosen>",
                    "<sent_9>", "<extra_id_13>", "hello", "<eosen>",
                    "<pad>", "<pad>", "<pad>", "<pad>",
                ],
            ],
            # Of length 9 + 4 = 13
            constants.DepthDataCollatorConstants.TARGET_TOKEN_TYPE_IDS: [
                [
                    1, 1, 1, 1, 1,
                    2, 2, 2, 2,
                    0, 0, 0, 0,
                ],
            ],
            # 15 x 15 attention mask corresponding to the 15 tokens in the input sequence (including padding) attending
            # to 15 tokens in the input sequence (including padding).
            constants.UnitTestConstants.EXPECTED_ENCODER_SELF_ATTENTION_MASK: [
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Pad
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Pad
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # Pad
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Pad
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Pad
                ],
            ],
            # 13 x 15 attention mask corresponding to 13 tokens in the target sequence attending to 15 tokens in the
            # input sequence
            constants.UnitTestConstants.EXPECTED_CROSS_ATTENTION_MASK: [
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Pad
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Pad
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Pad
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Pad
                ],
            ],
            # 13 x 13 attention mask corresponding to 13 tokens in the target sequence attending to 13 tokens in the
            # target sequence
            constants.UnitTestConstants.EXPECTED_DECODER_SELF_ATTENTION_MASK: [
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Pad
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # Pad
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Pad
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Pad
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
        """Test the `create_attention_mask` function with various inputs.
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
            corruption_lib.create_attention_mask(
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
            err_msg=f'Expected encoder self attention mask to be {expected_encoder_self_attention_mask} '
                    f'but got {encoder_self_attention_mask}'
        )
        np.testing.assert_almost_equal(
            actual=cross_attention_mask,
            desired=np.array(expected_cross_attention_mask),
            err_msg=f'Expected cross attention mask to be {expected_cross_attention_mask} '
                    f'but got {cross_attention_mask}'
        )
        np.testing.assert_almost_equal(
            actual=decoder_self_attention_mask,
            desired=np.array(expected_decoder_self_attention_mask),
            err_msg=f'Expected decoder self attention mask to be {expected_decoder_self_attention_mask} '
                    f'but got {decoder_self_attention_mask}'
        )


if __name__ == '__main__':
    unittest.main()

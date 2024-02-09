import random
import typing
import unittest
from typing import List

import numpy as np

from encoder_decoder_utils import tokenizer_utils
from encoder_decoder_utils import constants
from absl.testing import parameterized
import transformers

class MyTestCase(parameterized.TestCase):

    def setUp(self):
        self.tokenizer = tokenizer_utils.DepthTokenizer.from_pretrained(
            pretrained_model_name_or_path=constants.ModelHuggingFaceName.T5_BASE.value,
            max_num_sentences_in_text=constants.DEPTHTokenizerConstants.NUM_SENT_TOKENS,
        )

    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)

    @parameterized.named_parameters(
        {
            constants.UnitTestConstants.TESTCASE_NAME: 'No punctuation',
            constants.UnitTestConstants.TEXT: 'Hello world',
            constants.UnitTestConstants.SEED: 42,
            constants.UnitTestConstants.EXPECTED_RESULT: f'{constants.DEPTHTokenizerConstants.END_OF_SENTENCE_TOKEN}'
                                                         f'<sent_0>Hello world'
                                                         f'{constants.DEPTHTokenizerConstants.END_OF_SENTENCE_TOKEN}',
        },
        {
            constants.UnitTestConstants.TESTCASE_NAME: 'Single period',
            constants.UnitTestConstants.TEXT: 'Hello world.',
            constants.UnitTestConstants.SEED: 42,
            constants.UnitTestConstants.EXPECTED_RESULT: f'{constants.DEPTHTokenizerConstants.END_OF_SENTENCE_TOKEN}'
                                                         '<sent_0>Hello world.'
                                                         f'{constants.DEPTHTokenizerConstants.END_OF_SENTENCE_TOKEN}',
        },
        {
            constants.UnitTestConstants.TESTCASE_NAME: 'Multiple sentences (three periods)',
            constants.UnitTestConstants.TEXT: 'first sentence. second sentence. third sentence.',
            constants.UnitTestConstants.SEED: 42,
            constants.UnitTestConstants.EXPECTED_RESULT: f'{constants.DEPTHTokenizerConstants.END_OF_SENTENCE_TOKEN}'
                                                         f'<sent_0>first sentence.'
                                                         f'{constants.DEPTHTokenizerConstants.END_OF_SENTENCE_TOKEN}'
                             f'<sent_17>second sentence.{constants.DEPTHTokenizerConstants.END_OF_SENTENCE_TOKEN}'
                             f'<sent_15>third sentence.{constants.DEPTHTokenizerConstants.END_OF_SENTENCE_TOKEN}'
        },
        {
            constants.UnitTestConstants.TESTCASE_NAME: 'Multiple sentences (multiple punctuation types)',
            constants.UnitTestConstants.TEXT: 'Hello world! I am learning to use tokenizers. '
                                              'Did you know they are this cool?',
            constants.UnitTestConstants.SEED: 42,
            constants.UnitTestConstants.EXPECTED_RESULT: f'{constants.DEPTHTokenizerConstants.END_OF_SENTENCE_TOKEN}'
                                                         f'<sent_0>Hello world!'
                                                         f'{constants.DEPTHTokenizerConstants.END_OF_SENTENCE_TOKEN}'
                                                         f'<sent_17>I am learning to use tokenizers.'
                                                         f'{constants.DEPTHTokenizerConstants.END_OF_SENTENCE_TOKEN}'
                                                         f'<sent_15>Did you know they are this cool?'
                                                         f'{constants.DEPTHTokenizerConstants.END_OF_SENTENCE_TOKEN}'
        },
        {
            constants.UnitTestConstants.TESTCASE_NAME: 'Periods used as part of words',
            constants.UnitTestConstants.TEXT: 'You can call me Dr. Bamberger',
            constants.UnitTestConstants.SEED: 42,
            constants.UnitTestConstants.EXPECTED_RESULT: f'{constants.DEPTHTokenizerConstants.END_OF_SENTENCE_TOKEN}'
                                                         f'<sent_0>You can call me Dr. Bamberger'
                                                         f'{constants.DEPTHTokenizerConstants.END_OF_SENTENCE_TOKEN}',
        },
        {
            constants.UnitTestConstants.TESTCASE_NAME: 'Multi-character punctuations',
            constants.UnitTestConstants.TEXT: 'His lecture was so boring... I couldn\'t help but doze off.',
            constants.UnitTestConstants.SEED: 42,
            constants.UnitTestConstants.EXPECTED_RESULT: f'{constants.DEPTHTokenizerConstants.END_OF_SENTENCE_TOKEN}'
                                                         f'<sent_0>His lecture was so boring...'
                                                         f'{constants.DEPTHTokenizerConstants.END_OF_SENTENCE_TOKEN}'
                                                         f'<sent_17>I couldn\'t help but doze off.'
                                                         f'{constants.DEPTHTokenizerConstants.END_OF_SENTENCE_TOKEN}',
        }
    )
    def test_add_sentence_tokens_to_text(
            self,
            text: str,
            seed: int,
            expected_result: str,
    ):
        self.set_seed(seed)
        segmented_text = self.tokenizer.add_sentence_tokens_to_text(text)
        self.assertEqual(segmented_text, expected_result)

    @parameterized.named_parameters(
        {
            constants.UnitTestConstants.TESTCASE_NAME: 'Batch of size 1, no punctuation',
            constants.UnitTestConstants.BATCH_OF_TEXT: ['Here is a first sentence'],
            constants.UnitTestConstants.SEED: 42,
            constants.UnitTestConstants.EXPECTED_RESULT: [
                [f'{constants.DEPTHTokenizerConstants.END_OF_SENTENCE_TOKEN}',
                 '<sent_0>', f'{constants.T5TokenizerConstants.SENTENCE_PIECE_UNDERSCORE}Here',
                 f'{constants.T5TokenizerConstants.SENTENCE_PIECE_UNDERSCORE}is',
                 f'{constants.T5TokenizerConstants.SENTENCE_PIECE_UNDERSCORE}',
                 'a',
                 f'{constants.T5TokenizerConstants.SENTENCE_PIECE_UNDERSCORE}first',
                 f'{constants.T5TokenizerConstants.SENTENCE_PIECE_UNDERSCORE}sentence',
                 f'{constants.DEPTHTokenizerConstants.END_OF_SENTENCE_TOKEN}'],
            ],
        }
    )
    def test_tokenize_text(
            self,
            batch_of_text: typing.List[str],
            seed: int,
            expected_result: typing.List[typing.List[str]],
    ):
        self.set_seed(seed)
        tokenized_sentences = self.tokenizer.tokenize_text(batch_of_text)
        self.assertListEqual(tokenized_sentences, expected_result)

    @parameterized.named_parameters([
        {
            constants.UnitTestConstants.TESTCASE_NAME: 'Batch of size 1',
            constants.UnitTestConstants.BATCH_OF_TEXT: 'Here is a first sentence! This is a second. What about a third? Four is enough!',
            constants.UnitTestConstants.SEED: 42,
            constants.UnitTestConstants.EXPECTED_RESULT: {
                constants.DEPTHTokenizerConstants.INPUT_IDS: np.array(
                    [[32120, 32100, 947, 19, 3, 9, 166, 7142, 55,
                      32120, 32117, 100, 19, 3, 9, 511, 5, 32120,
                      32115, 363, 81, 3, 9, 1025, 58, 32120, 32101,
                      5933, 19, 631]]
                ),
                constants.DEPTHTokenizerConstants.TOKEN_TYPE_IDS: np.array(
                    [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3,
                      3, 3, 3, 3, 3, 4, 4, 4, 4]]
                ),
                constants.DEPTHTokenizerConstants.SPECIAL_TOKENS_MASK: np.array(
                    [[1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
                      0, 0, 0, 1, 1, 0, 0, 0]]
                ),
                constants.DEPTHTokenizerConstants.ATTENTION_MASK: np.array(
                    [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      1, 1, 1, 1, 1, 1]]
                ),
                constants.DEPTHTokenizerConstants.INPUT_LENGTH: np.array([30]),
                constants.DEPTHTokenizerConstants.NUM_TRUNCATED_TOKENS: 1,
            }
        },
        {
            constants.UnitTestConstants.TESTCASE_NAME: 'Batch of size 2',
            constants.UnitTestConstants.BATCH_OF_TEXT: [
                'Here is a first sentence! This is a second. What about a third? Four is enough!',
                'Here is a first sentence! This is a second. What about a third? Four is enough!',
            ],
            constants.UnitTestConstants.SEED: 42,
            constants.UnitTestConstants.EXPECTED_RESULT: {
                constants.DEPTHTokenizerConstants.INPUT_IDS: np.array(
                    [[32120, 32100, 947, 19, 3, 9, 166, 7142, 55, 32120,
                      32117, 100, 19, 3, 9, 511, 5, 32120,
                      32115, 363, 81, 3, 9, 1025, 58, 32120, 32101,
                      5933, 19, 631],
                     [32120, 32119, 947, 19, 3, 9, 166, 7142, 55,
                      32120, 32116, 100, 19, 3, 9, 511, 5, 32120,
                      32115, 363, 81, 3, 9, 1025, 58, 32120, 32105,
                      5933, 19, 631]]
                ),
                constants.DEPTHTokenizerConstants.TOKEN_TYPE_IDS: np.array(
                    [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3,
                      3, 3, 3, 3, 3, 4, 4, 4, 4],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3,
                      3, 3, 3, 3, 3, 4, 4, 4, 4]],
                ),
                constants.DEPTHTokenizerConstants.INPUT_LENGTH: np.array([30, 30]),
            }
        }
    ])
    def test_tokenize(
            self,
            batch_of_text: List[str],
            seed: int,
            expected_result: typing.Dict[str, np.ndarray],
    ):
        self.set_seed(seed)
        batch_encodings = self.tokenizer(
            text=batch_of_text,
            max_length=30,
            return_tensors='pt',
            truncation=transformers.tokenization_utils_base.TruncationStrategy.ONLY_FIRST,
            return_length=True,
        )
        np.testing.assert_array_equal(
            batch_encodings[constants.DEPTHTokenizerConstants.INPUT_IDS],
            expected_result[constants.DEPTHTokenizerConstants.INPUT_IDS],
            err_msg=f'expected: {expected_result[constants.DEPTHTokenizerConstants.INPUT_IDS]}\n'
                    f'actual: {batch_encodings[constants.DEPTHTokenizerConstants.INPUT_IDS]}'
        )
        np.testing.assert_array_equal(
            batch_encodings[constants.DEPTHTokenizerConstants.TOKEN_TYPE_IDS],
            expected_result[constants.DEPTHTokenizerConstants.TOKEN_TYPE_IDS],
            err_msg=f'expected: {expected_result[constants.DEPTHTokenizerConstants.TOKEN_TYPE_IDS]}\n'
                    f'actual: {batch_encodings[constants.DEPTHTokenizerConstants.TOKEN_TYPE_IDS]}'
        )
        np.testing.assert_array_equal(
            batch_encodings[constants.DEPTHTokenizerConstants.INPUT_LENGTH],
            expected_result[constants.DEPTHTokenizerConstants.INPUT_LENGTH],
            err_msg=f'expected: {expected_result[constants.DEPTHTokenizerConstants.INPUT_LENGTH]}\n'
                    f'actual: {batch_encodings[constants.DEPTHTokenizerConstants.INPUT_LENGTH]}'
        )

if __name__ == '__main__':
    unittest.main()

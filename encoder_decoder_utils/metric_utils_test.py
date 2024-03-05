import typing
import unittest
import numpy as np
import transformers

from absl.testing import parameterized
from encoder_decoder_utils import constants
from encoder_decoder_utils import metric_utils
from encoder_decoder_utils import tokenizer_utils


EXAMPLE_BATCH_1 = {
    'input_ids': [
        [32120, 32100, 37, 151, 28, 32080, 54, 2085, 16, 2756, 659, 406, 136, 8565, 38, 168, 5, 32120, 32111, 100],
        [32120, 32101, 29952, 21, 96, 32052, 1379, 772, 5, 32120, 32117, 1336, 7, 1720, 9, 41, 254, 8419, 52, 9],
        [32120, 32105, 32045, 612, 269, 32023, 785, 233, 8, 1710, 6, 32074, 6, 3026, 11, 3365, 5, 32120, 32116, 363],
        [32120, 32113, 10625, 7912, 16422, 49, 180, 11345, 7, 30, 884, 5044, 32097, 24, 25, 33, 1095, 28, 69, 5347],
        [32120, 32108, 37, 29, 34, 2992, 3, 9, 11530, 106, 32036, 5050, 139, 3, 9, 786, 19744, 32005, 3, 89],
    ],
    'labels:': [
        [32120, 32106, 32035, 22516, 549, 4170, 11300, 549, 32005, 5055, 4369, 32088, 3, 11425, 3, 23262, 445, 19114,
         134, 32042],
        [32120, 32117, 32021, 9, 3987, 9, 61, 32049, 3849, 7, 32093, 261, 32099, 13768, 32085, 4262, 151, 7, 5, 32120],
        [32120, 32101, 32080, 14038, 16773, 32015, 21, 32120, 32117, 32034, 3652, 28, 14038, 16773, 32120, 32115, 32008,
         366, 11, 213],
        [32120, 32119, 32094, 10625, 7912, 32012, 200, 32062, 3038, 312, 32120, 32117, 32098, 148, 31, 195, 174, 32016,
         69, 10312],
        [32120, 32112, 32043, 94, 22, 7, 8, 414, 13, 8, 296, 32120, 32104, 32014, 105, 32080, 17, 22, 32005, 5]
    ],
    'predictions': [
        # Errors at indices:
        # - 6 (4175 instead of 4170)
        # - 12 (32090 instead of 32088)
        # - 13 (4329 instead of 3)
        [32120, 32106, 32035, 22516, 549, 4175, 11300, 549, 32005, 5055, 4369, 32090, 4329, 11425, 3, 23262, 445, 19114,
         134, 32042],
        # Errors at indices:
        # - 3 (21 instead of 9)
        # - 4 (9 instead of 3987)
        [32120, 32117, 32021, 21, 9, 9, 61, 32049, 3849, 7, 32093, 261, 32099, 13768, 32085, 4262, 151, 7, 5, 32120],
        # No errors
        [32120, 32101, 32080, 14038, 16773, 32015, 21, 32120, 32117, 32034, 3652, 28, 14038, 16773, 32120, 32115,
         32008, 366, 11, 213],
        # No errors
        [32120, 32119, 32094, 10625, 7912, 32012, 200, 32062, 3038, 312, 32120, 32117, 32098, 148, 31, 195, 174, 32016,
         69, 10312],
        # Errors at indices:
        # - 10 (300 instead of 296)
        [32120, 32112, 32043, 94, 22, 7, 8, 414, 13, 8, 300, 32120, 32104, 32014, 105, 32080, 17, 22, 32005, 5]
    ]
}


class MetricsTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.t5_tokenizer = transformers.T5Tokenizer.from_pretrained(
            constants.ModelHuggingFaceName.T5_BASE.value,
        )
        self.depth_tokenizer = tokenizer_utils.DepthTokenizer.from_pretrained(
            constants.ModelHuggingFaceName.T5_BASE.value,
            max_num_sentences_in_text=20,
        )

    @parameterized.named_parameters(
        {
            constants.UnitTestConstants.TESTCASE_NAME: 'test_t5_eval_metrics',
            'predictions': EXAMPLE_BATCH_1['predictions'],
            constants.TokenizerConstants.LABEL_IDS: EXAMPLE_BATCH_1['labels:'],
            constants.TokenizerConstants.INPUT_IDS: EXAMPLE_BATCH_1['input_ids'],
            'expected_accuracy': 0.75,
        }
    )
    def test_t5_eval_metrics(
            self,
            predictions: typing.List[typing.List[float]],
            label_ids: typing.List[typing.List[int]],
            input_ids: typing.List[typing.List[int]],
            expected_accuracy: float,
    ):
        metrics = metric_utils.compute_metrics(
            eval_preds=transformers.EvalPrediction(
                predictions=np.array(predictions),
                label_ids=np.array(label_ids),
                inputs=np.array(input_ids),
            ),
            tokenizer=self.t5_tokenizer,
        )
        print(f'Computed metrics: {metrics}')

if __name__ == '__main__':
    unittest.main()
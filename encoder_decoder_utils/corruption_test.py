import typing
import unittest
import numpy as np
import transformers.models.t5
import corruption as corruption_lib
from absl.testing import parameterized
import random
import nltk
from encoder_decoder_utils.constants import (
    DEPTHTokenizerConstants,
)
from encoder_decoder_utils.tokenizer_utils import (
    create_discourse_tokenizer,
    batch_encode_plus,
)


# Fields for parameterized tests
TESTCASE_NAME = 'testcase_name'
SENTENCES = 'sentences'
INPUT_LENGTH = 'input_length'
NOISE_DENSITY = 'noise_density'
MEAN_NOISE_SPAN_LENGTH = 'mean_noise_span_length'
SEEDS = 'seeds'
EXPECTED_SPAN_MASKS = 'expected_span_masks'
EXPECTED_INPUT_IDS_SENTINEL = 'expected_input_ids_sentinel'
EXPECTED_LABEL_IDS_SENTINEL = 'expected_label_ids_sentinel'

EXPECTED_MODIFIED_INPUT_IDS = 'expected_modified_input_ids'
EXPECTED_MODIFIED_LABEL_IDS = 'expected_modified_label_ids'
EXPECTED_TOKEN_TYPE_IDS = 'expected_token_type_ids'
EXPECTED_LABEL_TOKEN_TYPE_IDS = 'expected_label_token_type_ids'

# Add special tokens for the test
TOKEN_TYPE_IDS = 'token_type_ids'
PADDING_TOKEN_IDS = 'padding_token_id'
EXPECTED_SHUFFLED_SENTENCE_ORDER = 'expected_shuffled_sentence_order'
EXPECTED_SHUFFLED_SENTENCE_LENGTHS = 'expected_shuffled_sentence_lengths'
EXPECTED_SHUFFLED_SENTENCE_START_INDICES = 'expected_shuffled_sentence_start_indices'

# Test inputs
EXAMPLE_1 = 'Hello world! I am learning to use tokenizers. Did you know they are this cool?'
EXAMPLE_2 = 'His lecture was so boring... I couldn\'t help but doze off.'
EXAMPLE_3 = 'Here is a first sentence! This is a second. What about a third? Four is enough!'

EXAMPLES = 'examples'
PMI_VOCAB = 'pmi_vocab'
TOKENIZER = 'tokenizer'
INPUT_TOKENS = 'input_tokens'
MAX_PREDICTIONS = 'max_predictions'
MLM_PROBABILITY = 'mlm_probability'
EXPECTED_MASK_LABELS = 'expected_mask_labels'
PMI_DEMO_VOCAB = {'1950 and 1951',
                  'muttering to himself',
                  'in an interview',
                  'looked back at',
                  'united states',
                  'biological and geological',
                  'sergeant at arms',
                  'greenland and iceland',
                  'plan of action',
                  'wrote several books',
                  'propaganda minister joseph',
                  "none of your damn",
                  "first woman to win",
                  "commanded by a lieutenant",
                  "tells the true story",
                  "everything that is happening",
                  "i have to tell",
                  "from 1987 to 1990",
                  "hard as a rock",
                  "journal has a 2015",
                  "job as a waitress",
                  "turn into the straight",
                  "sat at the bar",
                  "london brighton and south",
                  "ask me a question",
                  "comes to his senses",
                  "mother of two children",
                  "or by any information",
                  "school district officials did",
                  "some explaining to do",
                  "pi beta phi",
                  "jew and gentile",
                  "central business district",
                  "butter and jelly",
                  "pupil total expenditures",
                  "stained glass windows"
                  }

DEMO_TEXTS = ['The united states is a country in North America. looking back at 1950 and 1951,'
              ' the president said in an interview that he was muttering to himself.',
              'I am a sergeant at arms. I wrote several books. I am the propaganda minister joseph.',
              'My plan of action is to go to greenland and iceland. biological and geological.',
              "None of your damn business, but I have to tell you about the hard-as-a-rock butter and jelly sandwich "
              "I had for lunch.",
              "The first woman to win the prestigious award sat at the bar, sipping her drink, surrounded by stained "
              "glass windows.",
              "Commanded by a lieutenant, the military unit turned into the straight path, ready for the mission ahead.",
              "London, Brighton, and South—locations explored by the journalist in the 2015 journal, tell the true "
              "story of diverse experiences.",
              "As a waitress, I have to tell you about the time a customer at the bar asked me a question that left "
              "me puzzled.",
              "From 1987 to 1990, the school district officials did some explaining to do regarding pupil total "
              "expenditures.",
              "Journal has a 2015 entry about the central business district, where I worked a job as a waitress.",
              "Pi Beta Phi hosted an event, and everyone, from the mother of two children to the jew and gentile "
              "attendees, enjoyed it.",
              "Everything that is happening around the world makes me wonder if people will ever come to their senses.",
              "The stained glass windows in the church depicted the turn-of-the-century scenes, including the London, "
              "Brighton, and South railway.",
              "A hard-as-a-rock butter and jelly sandwich was my go-to snack during the years from 1987 to 1990.",
              "The waitress at the bar, a mother of two children, juggled her job and school district responsibilities.",
              "None of your damn excuses could justify the actions of the lieutenant who commanded the ill-fated "
              "mission.",
              "The true story of the central business district development unfolds in the 2015 journal entry.",
              "From 1987 to 1990, I attended school district events and occasionally worked as a waitress during "
              "weekends.",
              "The first woman to win the championship sat at the bar, surrounded by stained glass windows depicting "
              "her achievements.",
              "The jew and gentile communities collaborated on a project that transformed the central business "
              "district.",
              "I have to tell you about the hard-as-a-rock bread I bought at the local bakery, where I also worked a "
              "job as a waitress.",
              "Everything that is happening in the world requires individuals to come to their senses and take action.",
              "Pi Beta Phi hosted an event, and the mother of two children volunteered to help with the preparations.",
              "London, Brighton, and South were the settings for the true story I read in the 2015 journal about a "
              "waitress's journey.",
              "Ask me a question about my experiences from 1987 to 1990, and I'll gladly share the highlights of "
              "school district life.",
              "None of your damn complaints will change the fact that the lieutenant commanded the military unit with "
              "precision.",
              "The job as a waitress allowed me to meet people from different backgrounds, including jew and gentile "
              "customers.",
              "Turning into the straight path, the military unit commanded by a lieutenant embarked on a challenging "
              "mission.",
              "The stained glass windows in the church told the true story of the London, Brighton, and South "
              "railway's history.",
              "Pupil total expenditures in the school district increased during the years from 1987 to 1990.",
              "I have to tell you about the delicious butter and jelly combination that I enjoyed at the bar last "
              "night.",
              "Everything that is happening in the central business district reflects the economic changes of the "
              "past decade.",
              "Pi Beta Phi organized an event, and the first woman to win a prestigious award was a guest of honor.",
              "A mother of two children, working a job as a waitress, shared her experiences in the 2015 journal.",
              "The hard-as-a-rock bread I bought from the bakery turned into the straight talk of the town.",
              "None of your damn opinions can change the fact that the school district officials did some explaining "
              "to do.",
              "London, Brighton, and South—locations explored by the journalist in the 2015 journal entry—inspired my "
              "travel plans.",
              "Commanded by a lieutenant, the military unit's journey turned into the straight path of historical "
              "significance.",
              "The jew and gentile communities collaborated on a project that transformed the central business "
              "district landscape.",
              "I have to tell you about the hard-as-a-rock sandwich I made for lunch, inspired by my job as a waitress.",
              "Everything that is happening in the world makes the stained glass windows of our experiences more "
              "colorful.",
              "Pupil total expenditures in the school district increased during the years from 1987 to 1990, "
              "impacting education.",
              "Ask me a question about the first woman to win the award, and I'll gladly share the inspiring story.",
              "None of your damn excuses can justify the actions of the military unit commanded by a reckless "
              "lieutenant.",
              "The job as a waitress allowed me to connect with people from diverse backgrounds, including jew and "
              "gentile customers.",
              "Turning into the straight path, the military unit commanded by a lieutenant faced unexpected challenges.",
              "The true story of the central business district's growth unfolds in the 2015 journal entry.",
              "Pi Beta Phi hosted an event, and the mother of two children actively participated in organizing the "
              "activities.",
              "I have to tell you about the delicious butter and jelly combination that I enjoyed at the bar with "
              "stained glass windows.",
              "Everything that is happening in the central business district reflects the economic changes from 1987 "
              "to 1990.",
              "London, Brighton, and South were the settings for the true story I read in the 2015 journal, "
              "where I worked a job as a waitress.",
              "Ask me a question about the hard-as-a-rock experiences during my school district years, and I'll share "
              "the lessons learned.",
              "None of your damn complaints will change the fact that the lieutenant commanded the military unit with "
              "precision and expertise."]


################
##### DEPTH ####
################
SENT = 'sent'
SENT_TOKEN = '<sent>'
END_OF_SENTENCE_TOKEN = '<eosen>'
CLS = 'cls'
CLS_TOKEN = '<cls>'
ADDITIONAL_SPECIAL_TOKENS = 'additional_special_tokens'
nltk.download('punkt')


class CorruptionTest(parameterized.TestCase):

    def setUp(self):
        self.tokenizer = transformers.T5Tokenizer.from_pretrained('google/t5-v1_1-small')
        random.seed(42)

    @parameterized.named_parameters(
        {TESTCASE_NAME: 'short_sequence_length',
         INPUT_LENGTH: 5,
         NOISE_DENSITY: 0.5,
         MEAN_NOISE_SPAN_LENGTH: 2,
         SEEDS: [[42, 43], [44, 45]],
         EXPECTED_SPAN_MASKS: [False, True, True, False, False]
         },
        {TESTCASE_NAME: 'medium_sequence_length',
         INPUT_LENGTH: 10,
         NOISE_DENSITY: 0.5,
         MEAN_NOISE_SPAN_LENGTH: 3,
         SEEDS: [[42, 43], [44, 45]],
         EXPECTED_SPAN_MASKS: [True, False, False, False, True, True, True, False, False, True],
         },
        {TESTCASE_NAME: 'medium_sequence_length_different_seed',
         INPUT_LENGTH: 10,
         NOISE_DENSITY: 0.5,
         MEAN_NOISE_SPAN_LENGTH: 3,
         SEEDS: [[100, 101], [102, 103]],
         EXPECTED_SPAN_MASKS: [False, False, False, True, True, True, True, False, False, True],
         },
        {TESTCASE_NAME: 'medium_sequence_length_lower_density',
         INPUT_LENGTH: 10,
         NOISE_DENSITY: 0.3,
         MEAN_NOISE_SPAN_LENGTH: 3,
         SEEDS: [[42, 43], [44, 45]],
         EXPECTED_SPAN_MASKS: [False, False, True, True, True, False, False, False, False, False],
         },
        {TESTCASE_NAME: 'medium_sequence_length_higher_density',
         INPUT_LENGTH: 10,
         NOISE_DENSITY: 0.9,
         MEAN_NOISE_SPAN_LENGTH: 9,
         SEEDS: [[42, 43], [44, 45]],
         EXPECTED_SPAN_MASKS: [True, False, True, True, True, True, True, True, True, True],
         },
    )
    def test_random_spans_noise_mask(
            self,
            input_length: int,
            noise_density: float,
            mean_noise_span_length: int,
            seeds: typing.List[typing.List[int]],
            expected_span_masks: typing.List[typing.List[bool]]):
        np.random.seed(seeds[0][0])
        span_masks = corruption_lib.random_spans_noise_mask(
            sequence_length=input_length,
            maximum_length=input_length,
            noise_density=noise_density,
            mean_noise_span_length=mean_noise_span_length,
        )
        np.testing.assert_array_equal(span_masks, expected_span_masks)

    @parameterized.named_parameters(
        {TESTCASE_NAME: 'short_sequence_length',
         INPUT_LENGTH: 5,
         NOISE_DENSITY: 0.5,
         MEAN_NOISE_SPAN_LENGTH: 2,
         SEEDS: [[42, 43], [44, 45]],
         EXPECTED_INPUT_IDS_SENTINEL: [[0, 32099, -1, 0, 0]],
         },
        {TESTCASE_NAME: 'medium_sequence_length',
         INPUT_LENGTH: 10,
         NOISE_DENSITY: 0.5,
         MEAN_NOISE_SPAN_LENGTH: 3,
         SEEDS: [[42, 43], [44, 45]],
         EXPECTED_INPUT_IDS_SENTINEL: [[32099, 0, 0, 0, 32098, -1, -1, 0, 0, 32097]],
         },
        {TESTCASE_NAME: 'medium_sequence_length_different_seed',
         INPUT_LENGTH: 10,
         NOISE_DENSITY: 0.5,
         MEAN_NOISE_SPAN_LENGTH: 3,
         SEEDS: [[100, 101], [102, 103]],
         EXPECTED_INPUT_IDS_SENTINEL: [[0, 0, 0, 32099, -1, -1, -1, 0, 0, 32098]],
         },
        {TESTCASE_NAME: 'medium_sequence_length_lower_density',
         INPUT_LENGTH: 10,
         NOISE_DENSITY: 0.3,
         MEAN_NOISE_SPAN_LENGTH: 3,
         SEEDS: [[42, 43], [44, 45]],
         EXPECTED_INPUT_IDS_SENTINEL: [[0, 0, 32099, -1, -1, 0, 0, 0, 0, 0]],
         },
        {TESTCASE_NAME: 'medium_sequence_length_higher_density',
         INPUT_LENGTH: 10,
         NOISE_DENSITY: 0.9,
         MEAN_NOISE_SPAN_LENGTH: 9,
         SEEDS: [[42, 43], [44, 45]],
         EXPECTED_INPUT_IDS_SENTINEL: [[32099, 0, 32098, -1, -1, -1, -1, -1, -1, -1]],
         },
    )
    def test_create_sentinel_ids(
            self,
            input_length: int,
            noise_density: float,
            mean_noise_span_length: int,
            seeds: typing.List[typing.List[int]],
            expected_input_ids_sentinel: typing.List[typing.List[int]],
    ):
        np.random.seed(seeds[0][0])
        tokenizer = transformers.T5Tokenizer.from_pretrained('t5-small')
        span_masks = corruption_lib.random_spans_noise_mask(
            sequence_length=input_length,
            noise_density=noise_density,
            mean_noise_span_length=mean_noise_span_length,
            maximum_length=input_length,

        )
        print(f'span masks are: {span_masks}')
        input_ids_sentinel = corruption_lib.create_sentinel_ids_for_t5(
            np.expand_dims(span_masks, axis=0).astype(np.int8),
            vocab_size=len(tokenizer),
        )
        print(f'input_ids_sentinel are: {input_ids_sentinel}')
        np.testing.assert_array_equal(input_ids_sentinel, expected_input_ids_sentinel)

    @parameterized.named_parameters(
        {
            'testcase_name': 'basic_test',
            'examples': [
                'Hello world!',
                'Here is an example with a longer sentence',
                'An example with multiple sentences? Might be tricky... Worth testing!',
            ],
            'input_length': 20,
            'target_length': 10,
            EXPECTED_MODIFIED_INPUT_IDS: np.array(
                [[32099, 296, 55, 1,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [947, 32099, 46, 677, 28, 3, 9, 1200, 32098, 1,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [32099, 28, 1317, 16513, 32098, 16114, 233, 16990, 2505, 32097, 1,
                  0, 0, 0, 0, 0, 0, 0, 0, 0]],
            ),
            # Recall that in the target of T5, padding tokens (usually 0) are replaced with -100.
            EXPECTED_MODIFIED_LABEL_IDS: np.array(
                [[32099, 8774, -100, -100, -100, -100, -100, -100, -100, -100],
                 [32099, 19, 32098, 7142, -100, -100, -100, -100, -100, -100],
                 [32099, 389, 677, 32098, 58, 23840, 36, 32097, 55, -100]],
            ),
        },
        {
            'testcase_name': 'truncated targets',
            'examples': [EXAMPLE_1, EXAMPLE_2, EXAMPLE_3],
            'input_length': 20,
            'target_length': 10,
            EXPECTED_MODIFIED_INPUT_IDS: np.array(
                [[32099, 296, 32098, 183, 32097, 169, 14145, 8585, 7, 5, 3963, 25, 214, 32096, 1,
                  0, 0, 0, 0, 0],
                 [32099, 47, 32098, 27, 2654, 31, 17, 199, 32097, 776, 326, 5, 1,
                  0, 0, 0, 0, 0, 0, 0],
                 [947, 19, 32099, 100, 19, 3, 9, 511, 5, 32098, 81, 32097, 1,
                  0, 0, 0, 0, 0, 0, 0]]
            ),
            EXPECTED_MODIFIED_LABEL_IDS: np.array(

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
            TESTCASE_NAME: "No n-grams in PMI Vocab",
            INPUT_TOKENS: "Ofek went to Taub.",
            EXPECTED_MASK_LABELS: [1, 1, 1, 1, 0, 0, 0, 0],
            MAX_PREDICTIONS: 512,
            MLM_PROBABILITY: 0.5,
        },
        {
            TESTCASE_NAME: "Gibberish",
            INPUT_TOKENS: "asdvbdsasd asdvewasdf",
            EXPECTED_MASK_LABELS: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            MAX_PREDICTIONS: 512,
            MLM_PROBABILITY: 0.5,
        },
        {
            TESTCASE_NAME: "Some n-grams in PMI Vocab",
            INPUT_TOKENS: "I have to tell everything that is happening, but what happens after that, i don't know",
            EXPECTED_MASK_LABELS: [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            MAX_PREDICTIONS: 512,
            MLM_PROBABILITY: 0.5,
        },
        {
            TESTCASE_NAME: "extra punctuation",
            INPUT_TOKENS: "I want to tell you, maybe ask you? maybe yell! maybe scream & yell - then tell you.",
            EXPECTED_MASK_LABELS: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0,
                                   0, 1],
            MAX_PREDICTIONS: 512,
            MLM_PROBABILITY: 0.5,
        },
        {
            TESTCASE_NAME: "Testing high mlm_probability",
            INPUT_TOKENS: "butter and jelly pupil total expenditures stained glass windows",
            EXPECTED_MASK_LABELS: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            MAX_PREDICTIONS: 512,
            MLM_PROBABILITY: 1,
        },
        {
            TESTCASE_NAME: "Testing low mlm_probability",
            INPUT_TOKENS: "butter and jelly pupil total expenditures stained glass windows",
            EXPECTED_MASK_LABELS: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            MAX_PREDICTIONS: 512,
            MLM_PROBABILITY: 0,
        },

    )
    def test_pmi_mask_word(self,
                           input_tokens,
                           expected_mask_labels,
                           max_predictions,
                           mlm_probability,
                           seed=42,
                           ):
        """
        Test different use cases of the method "pmi_word_mask"
        :param input_tokens: input tokens to test
        :param max_predictions: max predictions to test
        :param mlm_probability: mlm probability to test
        :return: None
        """
        random.seed(seed)
        input_tokens = self.tokenizer(
            input_tokens,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
        )['input_ids']
        input_tokens = input_tokens.squeeze()
        ref_tokens = []
        for input_id in input_tokens:
            token = self.tokenizer._convert_id_to_token(input_id.item())
            ref_tokens.append(token)
        mask_labels_for_sample = corruption_lib.pmi_word_mask(
            ref_tokens,
            PMI_DEMO_VOCAB,
            max_predictions,
            mlm_probability,
        )
        self.assertIsNotNone(mask_labels_for_sample)
        self.assertListEqual(mask_labels_for_sample, expected_mask_labels)
        self.assertEquals(len(mask_labels_for_sample), len(ref_tokens))

    @parameterized.named_parameters(
        {
            TESTCASE_NAME: f"test stand use of pmi_noise_mask",
            EXAMPLES: DEMO_TEXTS,
            PMI_VOCAB: PMI_DEMO_VOCAB,
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
        tokenized_examples = self.tokenizer(
            examples,
            # return_tensors="pt",
            return_tensors="np",
            add_special_tokens=False,
            truncation=True,
            padding=True,
        )
        predicted_mask_labels = corruption_lib.pmi_noise_mask(
            tokenized_examples,
            pmi_vocab,
            self.tokenizer,
        )
        # TODO: check the correctness of the output relative to the input. If you set the seed initially, you should
        #  get the same output for the same input.
        self.assertIsNotNone(predicted_mask_labels)

    @parameterized.named_parameters(
        {
            'testcase_name': 'shuffle_multi_sentence',
            'input_ids': np.array(
                [[32128, 31999, 55, 32145, 27, 31998, 169, 14145, 8585, 31997,
                  5, 32143, 3963, 25, 31996, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [32137, 978, 31999, 233, 32134, 27, 2654, 31, 31998, 68,
                  103, 31997, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [32143, 31999, 3, 31998, 7142, 55, 32142, 31997, 19, 3,
                  9, 511, 5, 32137, 363, 31996, 32130, 5933, 19, 31995,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
            ),
            'token_type_ids': np.array(
                [[1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0]]
            ),
            'expected_input_ids': np.array([
                [32128, 31999, 55, 32145, 27, 31998, 169, 14145, 8585,
                 31997, 5, 32143, 3963, 25, 31996, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [32134, 27, 2654, 31, 31998, 68, 103, 31997, 32137,
                 978, 31999, 233, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [32143, 31999, 3, 31998, 7142, 55, 32142, 31997, 19, 3, 9, 511,
                 5, 32130, 5933, 19, 31995, 32137, 363, 31996, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0]
            ]),
            'expected_token_type_ids': np.array([
                [1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0],
                [2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 3, 3, 3, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0]
            ]),
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


class TestCreateAttentionMask(parameterized.TestCase):

    @parameterized.named_parameters(
        {
            "testcase_name": "TestSingleSentenceNoPadding",
            "input_ids": ["<eosen>", "<sent_17>", "hello", "<extra_id_82>", "<eosen>"],
            "input_token_type_ids": [[1, 1, 1, 1, 1]],
            "target_ids": ["<eosen>", "<sent_17>", "<extra_id_82>", "world", "<eosen>"],
            "target_token_type_ids": [[1, 1, 1, 1, 1]],
            "expected_encoder_self_attention_mask": [[
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],  # sentence token in the input
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ]],
            "expected_decoder_self_attention_mask": [[
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],  # sentence token in target
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
            ]],
            "expected_cross_attention_mask": [[
                [1, 1, 1, 1, 1],
                [0, 1, 0, 0, 0],  # sentence token in target
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ]],
        },
        {
            "testcase_name": "TestMultipleSentencesNoPadding",
            "input_ids": [
                "<eosen>", "<sent_17>", "hello", "<extra_id_82>", "<eosen>", "<sent_9>", "<extra_id_13>", "are",
                "you?", "<eosen>"],
            "input_token_type_ids": [[1, 1, 1, 1, 1, 2, 2, 2, 2, 2]],
            "target_ids": [
                "<eosen>", "<sent_17>", "<extra_id_82>", "world", "<eosen>", "<sent_9>", "<extra_id_13>", "how",
                "<eosen>",
            ],
            "target_token_type_ids": [[1, 1, 1, 1, 1, 2, 2, 2, 2]],
            # 10 x 10 attention mask corresponding to the 10 tokens in the input sequence
            "expected_encoder_self_attention_mask": [[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                                   [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                                   [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                                                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]],
            # 9 x 9 attention mask corresponding to the 9 tokens in the target sequence
            "expected_decoder_self_attention_mask": [[[1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                   [0, 1, 0, 0, 0, 0, 0, 0, 0],  # sentence token #1
                                                                   [1, 1, 1, 0, 0, 0, 0, 0, 0],
                                                                   [1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                                   [1, 1, 1, 1, 1, 0, 0, 0, 0],
                                                                   [0, 1, 0, 0, 0, 1, 0, 0, 0],  # sentence token #2
                                                                   [1, 1, 1, 1, 1, 1, 1, 0, 0],
                                                                   [1, 1, 1, 1, 1, 1, 1, 1, 0],
                                                                   [1, 1, 1, 1, 1, 1, 1, 1, 1]]],
            # 9 x 10 attention mask corresponding to the 9 tokens in the target sequence attending to 10 tokens in the
            # input sequence
            "expected_cross_attention_mask": [[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # sentence token #1
                                                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # sentence token #2
                                                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]],

        },
        {
            "testcase_name": "TestSingleSentenceWithPadding",
            "input_ids": ["<eosen>", "<sent_17>", "hello", "<extra_id_82>", "<eosen>"],
            "input_id_padding": 5,
            "input_token_type_ids": [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]],
            "target_ids": ["<eosen>", "<sent_17>", "<extra_id_82>", "world", "<eosen>"],
            "target_id_padding": 4,
            "target_token_type_ids": [[1, 1, 1, 1, 1, 0, 0, 0, 0]],
            # 10 x 10 attention mask corresponding to the 10 tokens in the input sequence
            "expected_encoder_self_attention_mask": [[[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                   [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                   [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                   [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                   [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Pad
                                                   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Pad
                                                   [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # Pad
                                                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Pad
                                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Pad
                                                   ]],
            # 9 x 9 attention mask corresponding to the 9 (of which 4 are padding) tokens in the target sequence
            "expected_decoder_self_attention_mask": [[[1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 1, 0, 0, 0, 0, 0, 0, 0],
                                                    [1, 1, 1, 0, 0, 0, 0, 0, 0],
                                                    [1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                    [1, 1, 1, 1, 1, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 1, 0, 0, 0],  # Pad
                                                    [0, 0, 0, 0, 0, 0, 1, 0, 0],  # Pad
                                                    [0, 0, 0, 0, 0, 0, 0, 1, 0],  # Pad
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 1],  # Pad
                                                    ]],
            # 9 x 10 attention mask corresponding to the 9 tokens in the target sequence attending to 10 tokens in the
            # input sequence
            "expected_cross_attention_mask": [[[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]],
        },
        {
            "testcase_name": "TestMultipleSentencesWithPadding",
            "input_ids": ["<eosen>", "<sent_17>", "hello", "<extra_id_82>", "<eosen>", "<sent_9>", "<extra_id_13>",
                     "are", "you?", "<eosen>"],
            "input_id_padding": 5,
            "input_token_type_ids": [[1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0]],  # Of length 10 + 5 = 15
            "target_ids": ["<eosen>", "<sent_17>", "<extra_id_82>", "world", "<eosen>", "<sent_9>", "<extra_id_13>",
                     "hello", "<eosen>"],
            "target_id_padding": 4,
            "target_token_type_ids": [[1, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0]],  # Of length 9 + 4 = 13
            # 15 x 15 attention mask corresponding to the 15 tokens in the input sequence (including padding) attending
            # to 15 tokens in the input sequence (including padding).
            "expected_encoder_self_attention_mask": [[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
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
                                                                   ]],
            # 13 x 15 attention mask corresponding to 13 tokens in the target sequence attending to 15 tokens in the
            # input sequence
            "expected_cross_attention_mask": [[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
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
                                                            ]],
            # 13 x 13 attention mask corresponding to 13 tokens in the target sequence attending to 13 tokens in the
            # target sequence
            "expected_decoder_self_attention_mask": [[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
                                                                   ]],
        },
    )
    def test_create_attention_mask(
            self,
            input_ids,
            input_token_type_ids,
            target_ids,
            target_token_type_ids,
            expected_encoder_self_attention_mask,
            expected_cross_attention_mask,
            expected_decoder_self_attention_mask,
            num_sent_tokens: int = 20,
            input_id_padding: int = 0,
            target_id_padding: int = 0,
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
        # Create the tokenizer
        tokenizer = transformers.models.t5.T5Tokenizer.from_pretrained('t5-small')
        additional_special_tokens = (
            tokenizer.special_tokens_map[ADDITIONAL_SPECIAL_TOKENS])
        sent_tokens = [f'<{SENT}_{i}>' for i in range(num_sent_tokens)]
        additional_special_tokens.extend(sent_tokens)
        additional_special_tokens.append(END_OF_SENTENCE_TOKEN)
        special_tokens_dict = {ADDITIONAL_SPECIAL_TOKENS: additional_special_tokens}
        tokenizer.add_special_tokens(special_tokens_dict=special_tokens_dict)

        input_ids = np.array(
            [tokenizer.encode(input_ids, add_special_tokens=False) + [tokenizer.pad_token_id] * input_id_padding]
        )
        target_ids = np.array(
            [tokenizer.encode(target_ids, add_special_tokens=False) + [tokenizer.pad_token_id] * target_id_padding]
        )
        input_token_type_ids = np.array(input_token_type_ids)
        target_token_type_ids = np.array(target_token_type_ids)
        encoder_self_attention_mask, cross_attention_mask, decoder_self_attention_mask = (
            corruption_lib.create_attention_mask(
                input_ids,
                target_ids,
                input_token_type_ids,
                target_token_type_ids,
                tokenizer,
            )
        )
        np.testing.assert_almost_equal(actual=encoder_self_attention_mask, desired=expected_encoder_self_attention_mask)
        np.testing.assert_almost_equal(actual=cross_attention_mask, desired=expected_cross_attention_mask)
        np.testing.assert_almost_equal(actual=decoder_self_attention_mask, desired=expected_decoder_self_attention_mask)

    @parameterized.named_parameters(
        {TESTCASE_NAME: 'multiple_inputs_consisting_of_a_single_sentence',
         SENTENCES: [
             'hello world!',
             'I am a machine learning researcher.',
             'She was born in New York, but not in the city.',
         ],
         NOISE_DENSITY: 0.5,
         SEEDS: [[42, 43], [44, 45]],
         MEAN_NOISE_SPAN_LENGTH: 3,
         EXPECTED_MODIFIED_INPUT_IDS: np.array(
             [[32148, 32128, 21820, 296, 32078, 32148,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [32148, 32137, 32091, 183, 3, 32098, 18658, 5, 32148,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [32148, 32143, 32000, 2170, 32052, 368, 1060, 6, 68, 59, 32060, 8, 690, 32018, 32148,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
         ),
         EXPECTED_TOKEN_TYPE_IDS: np.array(
             [[1, 1, 1, 1, 1, 1,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 1,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
         ),
         EXPECTED_MODIFIED_LABEL_IDS: np.array(
             [[32148, 32128, 32078, 55, 32148,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [32148, 32137, 32091, 27, 32098, 9, 1437, 1036, 32148,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [32148, 32143, 32000, 451, 47, 32052, 16, 32060, 16, 32018, 5, 32148,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
         ),
         EXPECTED_LABEL_TOKEN_TYPE_IDS: np.array(
             [[1, 1, 1, 1, 1,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 1,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
         )
         },
        {TESTCASE_NAME: 'multiple_inputs_consisting_of_a_multiple_sentences',
         SENTENCES: [EXAMPLE_1, EXAMPLE_2, EXAMPLE_3],
         NOISE_DENSITY: 0.5,
         SEEDS: [[42, 43], [44, 45]],
         MEAN_NOISE_SPAN_LENGTH: 3,
         EXPECTED_MODIFIED_INPUT_IDS: np.array(
             [[32148, 32128, 32066, 32148,
               32145, 27, 32052, 12, 169, 14145, 32038, 32148,
               32143, 3963, 32076, 33, 32059, 1633, 58, 32148,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [32148, 32137, 32099, 13006, 233, 32148,
               32134, 27, 2654, 32095, 103, 32028, 326, 5, 32148,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [32148, 32143, 947, 19, 3, 9, 166, 7142, 55, 32148,
               32142, 32091, 19, 32085, 32148,
               32137, 32004, 9, 32006, 58, 32148,
               32130, 5933, 32064,
               0, 0, 0, 0, 0, 0]]
         ),
         EXPECTED_TOKEN_TYPE_IDS: np.array(
             [[1, 1, 1, 1,
               2, 2, 2, 2, 2, 2, 2, 2,
               3, 3, 3, 3, 3, 3, 3, 3,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1,
               2, 2, 2, 2, 2, 2, 2, 2, 2,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               2, 2, 2, 2, 2,
               3, 3, 3, 3, 3, 3,
               4, 4, 4,
               0, 0, 0, 0, 0, 0]]
         ),
         EXPECTED_MODIFIED_LABEL_IDS: np.array(
             [[32148, 32128, 32066, 8774, 296, 55, 32148,
               32145, 32052, 183, 1036, 32038, 8585, 7, 5, 32148,
               32143, 32076, 25, 214, 79, 32059, 48, 32148,
               0, 0, 0, 0, 0, 0],
              [32148, 32137, 32099, 978, 7177, 47, 78, 32148,
               32134, 32095, 31, 17, 199, 68, 32028, 776, 32148,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [32148, 32143, 32148,
               32142, 32091, 100, 32085, 3, 9, 511, 5, 32148,
               32137, 32004, 363, 81, 3, 32006, 1025, 32148,
               32130, 32064, 19, 631,
               0, 0, 0, 0, 0, 0]]
         ),
         EXPECTED_LABEL_TOKEN_TYPE_IDS: np.array(
             [[1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
               0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0],
              [1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4,
               0, 0, 0, 0, 0, 0]]
         )
    })
    def test_create_inputs_and_targets(
        self,
        sentences: typing.List[str],
        noise_density: float,
        seeds: typing.List[typing.List[int]],
        mean_noise_span_length: int,
        expected_modified_input_ids: typing.List[typing.List[int]],
        expected_token_type_ids: typing.List[typing.List[int]],
        expected_modified_label_ids: typing.List[typing.List[int]],
        expected_label_token_type_ids: typing.List[typing.List[int]],
        sequence_length: int = 30,
    ):
        tokenizer, _ = create_discourse_tokenizer(model_name='google/ul2', use_fast=True)
        batch_encodings = batch_encode_plus(
            batch_text=sentences,
            tokenizer=tokenizer,
            seed=seeds[0][0],
            max_length=sequence_length,
            return_tensors=None,
        )

        np.random.seed(seeds[0][0])
        input_ids = np.array(batch_encodings[DEPTHTokenizerConstants.INPUT_IDS])
        token_type_ids = np.array(batch_encodings[DEPTHTokenizerConstants.TOKEN_TYPE_IDS])
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
        special_tokens = tokenizer.all_special_ids
        sentence_tokens = list(
            filter(lambda token: f'<{DEPTHTokenizerConstants.SENT}' in token, tokenizer.all_special_tokens))
        sentence_tokens.append(DEPTHTokenizerConstants.END_OF_SENTENCE_TOKEN)
        sentence_token_ids = tokenizer.convert_tokens_to_ids(sentence_tokens)

        # Ensure mask is only applied to non-special tokens.
        augmented_input_span_mask = np.where(np.isin(input_ids, special_tokens, invert=True), span_mask, False)

        # Create a sentinel mask, where 0s indicate a lack of mask, positive values indicate the start of a masked span,
        #  and -1 indicates the continuation of a masked span.
        input_ids_sentinel = corruption_lib.create_sentinel_ids(tokenizer, augmented_input_span_mask.astype(np.int8))

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

        np.testing.assert_array_equal(modified_input_ids, expected_modified_input_ids)
        np.testing.assert_array_equal(modified_input_token_type_ids, expected_token_type_ids)
        np.testing.assert_array_equal(modified_label_ids, expected_modified_label_ids)
        np.testing.assert_array_equal(modified_label_token_type_ids, expected_label_token_type_ids)


if __name__ == '__main__':
    unittest.main()

import dataclasses
import enum
import typing


TRUSTED_DATASETS = {
    'c4',
    'wikipedia',
    'bookcorupus',
    'wikitext',
    'togethercomputer/RedPajama-Data-V2',
}


class MonitoringPlatform(enum.Enum):
    """Monitoring platform."""
    WANDB = "wandb"
    NEPTUNE = "neptune"


class Device(enum.Enum):
    """Device to train on."""
    CPU = "cpu"
    GPU = "gpu_cluster"


class TrainingPhase(enum.Enum):
    """Training phase."""
    FT = "ft"
    PT = "pt"


class ReturnTensor(enum.Enum):
    """Return tensor."""
    PT = "pt"
    NP = "np"


class NumericalPrecision(enum.Enum):
    """Numerical precision."""
    FP32 = "fp32"
    BF16 = "bf16"


class PaddingConstants(enum.Enum):
    """Padding constants."""
    MAX_LENGTH = "max_length"
    LONGEST = "longest"


class ModelImplementation(enum.Enum):
    """Model implementations."""
    LOCAL_T5 = "local_t5"
    HUGGINGFACE_T5 = "hf_t5"
    DEPTH = "depth"


class ModelHuggingFaceName(enum.Enum):
    """HuggingFace model names."""
    T5_SMALL = "google/t5-v1_1-small"  # 60M params
    T5_BASE = "google/t5-v1_1-base"  # 220M params
    T5_LARGE = "google/t5-v1_1-large"  # 770M params
    T5_3B = "google/t5-v1_1-xl"  # 3B params
    T5_11B = "google/t5-v1_1-xxl"  # 11B params
    UL2 = "google/ul2"  # 20B params


class EnvironmentVariable(enum.Enum):
    """Environment variables."""
    SLURM_JOB_ID = "SLURM_JOB_ID"


class DatasetSplit(enum.Enum):
    """Dataset split."""
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"


class Optimizer(enum.Enum):
    """Optimizer constants."""
    ADAMW: str = 'adamw'
    ADAMWSCALE: str = 'adamwscale'
    ADAFACTOR: str = 'adafactor'
    ADAMW_HF: str = 'adamw_hf'
    ADAMW_TORCH: str = 'adamw_torch'


class Scheduler(enum.Enum):
    """Scheduler constants."""
    CONSTANT: str = 'constant'
    CONSTANT_WITH_WARMUP: str = 'constant_with_warmup'
    COSINE: str = 'cosine'
    LEGACY: str = 'legacy'  # The legacy scheduler from the original T5 paper.
    LINEAR: str = 'linear'
    INVERSE_SQRT: str = 'inverse_sqrt'


class TagCategory(enum.Enum):
    """Tag categories for logging."""
    MODEL = "model"
    DATASET = "dataset"
    BATCH_SIZE = "batch_size"
    BASE_LR = "base_lr"
    LR_SCHEDULER = "lr_scheduler"
    EPOCHS = "epochs"
    STEPS = "steps"
    IMPLEMENTATION = "implementation"
    PRECISION = "precision"
    NUM_PROCESSES = "num_processes"
    NUM_GPUS = "num_gpus"


class Metric(enum.Enum):
    """Metrics."""
    # Classification metrics.
    ACCURACY: str = 'accuracy'
    LOSS: str = 'loss'
    PRECISION: str = 'precision'
    RECALL: str = 'recall'
    F1: str = 'f1'
    MCC: str = 'mcc'
    SPEARMAN: str = 'spearman'
    PEARSON: str = 'pearson'

    # Generative metrics.
    ROUGE: str = 'rouge'
    ROUGE_L: str = 'rougeL'

    # Optimization metrics.
    GRAD_L2: str = 'grad_l2'
    WEIGHTS_L2: str = 'weights_l2'
    LR: str = 'lr'
    SECONDS_PER_STEP: str = 'seconds_per_step'
    TIME: str = 'time'

    # Data/Tokenization metrics.
    AVERAGE_NON_PADDING_TOKENS_PER_EXAMPLE_INPUT: str = 'average_non_padding_tokens_per_example_input'
    VARIANCE_NON_PADDING_TOKENS_PER_EXAMPLE_INPUT: str = 'variance_non_padding_tokens_per_example_input'
    AVERAGE_NON_PADDING_TOKENS_PER_EXAMPLE_LABEL: str = 'average_non_padding_tokens_per_example_label'
    VARIANCE_NON_PADDING_TOKENS_PER_EXAMPLE_LABEL: str = 'variance_non_padding_tokens_per_example_label'
    NUM_NON_PADDING_TOKENS_IN_BATCH_INPUT: str = 'num_non_padding_tokens_in_batch_input'
    NUM_NON_PADDING_TOKENS_IN_BATCH_LABEL: str = 'num_non_padding_tokens_in_batch_label'
    NUM_SENTENCE_TOKENS: str = 'num_sentence_tokens'

    # Example-level metric names
    EXAMPLE_ACCURACY: str = 'example_accuracy'
    EXAMPLE_F1: str = 'example_f1'
    EXAMPLE_PRECISION: str = 'example_precision'
    EXAMPLE_RECALL: str = 'example_recall'
    EXAMPLE_MCC: str = 'example_mcc'
    EXAMPLE_SPEARMAN: str = 'example_spearman'
    EXAMPLE_PEARSON: str = 'example_pearson'

    # Token-level metric names
    TOKEN_ACCURACY: str = 'token_accuracy'
    TOKEN_F1: str = 'token_f1'
    TOKEN_PRECISION: str = 'token_precision'
    Token_RECAll: str = 'token_recall'
    TOKEN_MCC: str = 'token_mcc'
    TOKEN_SPEARMAN: str = 'token_spearman'
    TOKEN_PEARSON: str = 'token_pearson'


    # Depth metrics
    AVERAGE_LOSS_ON_SENTENCE_TOKENS: str = 'average_loss_on_sentence_tokens'
    VARIANCE_LOSS_ON_SENTENCE_TOKENS: str = 'variance_loss_on_sentence_tokens'
    AVERAGE_LOSS_ON_NON_SENTENCE_TOKENS: str = 'average_loss_on_non_sentence_tokens'
    VARIANCE_LOSS_ON_NON_SENTENCE_TOKENS: str = 'variance_loss_on_non_sentence_tokens'

    # Pre-Training metrics
    NUM_SENTINEL_TOKENS_IN_LABELS: str = 'num_sentinel_tokens_in_labels'
    NUM_SENTINEL_TOKENS_IN_INPUTS: str = 'num_sentinel_tokens_in_inputs'
    PADDING_TOKENS_IN_LABELS: str = 'padding_tokens_in_labels'
    PADDING_TOKENS_IN_INPUTS: str = 'padding_tokens_in_inputs'
    NON_PADDING_TOKENS_IN_LABELS: str = 'non_padding_tokens_in_labels'
    NON_PADDING_TOKENS_IN_INPUTS: str = 'non_padding_tokens_in_inputs'
    SPAN_LENGTH: str = 'span_length'

    # Pre-Training Accuracy
    SENTENCE_ACCURACY: str = 'sentence_accuracy'
    RECONSTRUCTION_ACCURACY: str = 'reconstruction_accuracy'


class DownstreamDataset(enum.Enum):
    """Downstream datasets."""
    GLUE: str = 'glue'
    SUPERGLUE: str = 'superglue'
    SQUAD: str = 'squad'
    DISCO_EVAL: str = 'OfekGlick/DiscoEval'


@dataclasses.dataclass
class RawTrainingExampleConstants:
    PREPROCESSED_COLUMN_NAMES: typing.Tuple[str] = ('idx', 'processed_inputs', 'processed_outputs')
    TEXT_COLUMN_NAME: str = 'text'


@dataclasses.dataclass
class TokenizerConstants:
    """
    Tokenizer constants. These are used to tokenize the inputs and outputs of the T5 model.
    """
    TOKENIZER: str = 'tokenizer'
    IN_LENGTH: str = 'in_length'
    OUT_LENGTH: str = 'out_length'
    EXAMPLES: str = 'examples'
    INPUT_IDS: str = 'input_ids'
    ATTENTION_MASK: str = 'attention_mask'
    LABEL_IDS: str = 'label_ids'
    LABEL: str = 'label'
    LABELS: str = 'labels'
    DECODER_INPUT_IDS: str = 'decoder_input_ids'
    TOKEN_TYPE_IDS: str = 'token_type_ids'
    SPECIAL_TOKENS_MASK: str = 'special_tokens_mask'
    INPUT_LENGTH: str = 'input_length'
    TARGET_LENGTH: str = 'target_length'
    NUM_TRUNCATED_TOKENS: str = 'num_truncated_tokens'

@dataclasses.dataclass
class T5DataCollatorConstants:

    # Inputs
    PAD_TO_MAX_LENGTH: str = 'pad_to_max_length'
    MAX_LENGTH: str = 'max_length'
    TRUNCATION: str = 'truncation'
    MEAN_NOISE_SPAN_LENGTH: str = 'mean_noise_span_length'
    NOISE_DENSITY: str = 'noise_density'
    PMI_VOCAB: str = 'pmi_vocab'

    # Outputs
    INPUT_IDS: str = 'input_ids'
    TARGET_IDS: str = 'target_ids'
    LABELS: str = 'labels'
    LENGTH: str = 'length'

    ENCODER_ATTENTION_MASK: str = 'encoder_attention_mask'
    DECODER_ATTENTION_MASK: str = 'decoder_attention_mask'
    CROSS_ATTENTION_MASK: str = 'cross_attention_mask'

@dataclasses.dataclass
class DepthDataCollatorConstants(T5DataCollatorConstants):

    # Inputs
    INPUT_TOKEN_TYPE_IDS: str = 'input_token_type_ids'
    TARGET_TOKEN_TYPE_IDS: str = 'target_token_type_ids'
    LABEL_TOKEN_TYPE_IDS: str = 'label_token_type_ids'

    SENTENCE_SHUFFLING_PROBABILITY: str = 'sentence_shuffling_probability'

    LOSS_WEIGHTS: str = 'loss_weights'
    SENTENCE_LOSS_COEFFICIENT: str = 'sentence_loss_coefficient'

    # Outputs
    IS_SHUFFLED: str = 'is_shuffled'


@dataclasses.dataclass
class T5TokenizerConstants(TokenizerConstants):
    START_TOKEN: str = '</s>'
    PAD_TOKEN: str = '<pad>'
    SPACE_TOKEN: str = "▁"
    PAD_TOKEN_ID: int = -100
    TEXT_COLUMN_NAME: str = 'text'
    SENTENCE_PIECE_UNDERSCORE: str = '▁'
    OVERFLOWING_TOKENS: str = 'overflowing_tokens'
    ADDITIONAL_SPECIAL_TOKENS: str = 'additional_special_tokens'


# TODO: Add a UL2 tokenizer class.
class DEPTHTokenizerConstants(T5TokenizerConstants):
    SENTENCE_TOKEN: str = "SENT"
    SENT: str = "sent"
    SENT_TOKEN: str = '<sent>'
    END_OF_SENTENCE_TOKEN: str = "<eosen>"
    NUM_SENT_TOKENS: int = 20


@dataclasses.dataclass
class DataLoaderConstants:
    BATCH_SIZE: str = 'batch_size'
    NUM_WORKERS: str = 'num_workers'
    PIN_MEMORY: str = 'pin_memory'
    DROP_LAST: str = 'drop_last'
    SHUFFLE: str = 'shuffle'
    COLLATE_FN: str = 'collate_fn'
    SAMPLER: str = 'sampler'
    PERSISTENT_WORKERS: str = 'persistent_workers'


@dataclasses.dataclass(frozen=True)
class OptimizerConstants:
    """
    Optimizer constants.
    """
    PARAMS: str = 'params'
    WEIGHT_DECAY: str = 'weight_decay'
    STEP: str = 'step'
    EXP_AVG: str = 'exp_avg'
    EXP_AVG_SQ: str = 'exp_avg_sq'
    LR: str = 'lr'
    CORRECT_BIAS: str = 'correct_bias'
    EPS: str = 'eps'


@dataclasses.dataclass(frozen=True)
class SchedulerConstants:
    NO_DECAY: typing.Tuple[str] = ("bias", "LayerNorm", "layernorm", "layer_norm", "ln")


@dataclasses.dataclass(frozen=True)
class GLUEConstants:
    ID = 'id'
    LABEL = 'label'

    # Dataset names
    COLA: str = 'cola'
    MNLI: str = 'mnli'
    MRPC: str = 'mrpc'
    QNLI: str = 'qnli'
    QQP: str = 'qqp'
    RTE: str = 'rte'
    SST2: str = 'sst2'
    STS_B: str = 'stsb'
    WNLI: str = 'wnli'
    AX: str = 'ax'

    # Types of MNLI datasets
    MATCHED: str = 'matched'
    MISMATCHED: str = 'mismatched'


@dataclasses.dataclass(frozen=True)
class UnitTestConstants:

    ########################
    ### Unit test inputs ###
    ########################

    TESTCASE_NAME: str = 'testcase_name'
    TEXT: str = 'text'
    BATCH_OF_TEXT: str = 'batch_of_text'
    EXAMPLES: str = 'examples'
    SEED: str = 'seed'
    DO_SHUFFLE: str = 'do_shuffle'
    MODEL_IMPLEMENTATION: str = 'model_implementation'

    ###################################
    ### Unit test expected results  ###
    ###################################

    EXPECTED_RESULT: str = 'expected_result'

    # Span-Masking
    EXPECTED_SPAN_MASKS: str = 'expected_span_masks'
    EXPECTED_INPUT_IDS: str = 'expected_input_ids'
    EXPECTED_TARGET_IDS: str = 'expected_target_ids'

    EXPECTED_INPUT_IDS_SENTINEL: str = 'expected_input_ids_sentinel'
    EXPECTED_MODIFIED_INPUT_IDS: str = 'expected_modified_input_ids'
    EXPECTED_MODIFIED_LABEL_IDS: str = 'expected_modified_label_ids'

    EXPECTED_LENGTH: str = 'expected_length'

    # Sentence shuffling
    EXPECTED_TOKEN_TYPE_IDS: str = 'expected_token_type_ids'
    EXPECTED_LABEL_TOKEN_TYPE_IDS: str = 'expected_label_token_type_ids'
    EXPECTED_IS_SHUFFLED: str = 'expected_is_shuffled'

    # Attention masks
    EXPECTED_ENCODER_SELF_ATTENTION_MASK: str = 'expected_encoder_self_attention_mask'
    EXPECTED_DECODER_SELF_ATTENTION_MASK: str = 'expected_decoder_self_attention_mask'
    EXPECTED_CROSS_ATTENTION_MASK: str = 'expected_cross_attention_mask'

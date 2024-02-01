import dataclasses
import enum
import typing


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


class NumericalPrecision(enum.Enum):
    """Numerical precision."""
    FP32 = "fp32"
    BF16 = "bf16"


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


class Scheduler(enum.Enum):
    """Scheduler constants."""
    CONSTANT: str = 'constant'
    COSINE: str = 'cosine'
    LEGACY: str = 'legacy'  # The legacy scheduler from the original T5 paper.
    LINEAR: str = 'linear'


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

    # Example-level metrics
    EXAMPLE_ACCURACY: str = 'example_accuracy'
    EXAMPLE_F1: str = 'example_f1'
    EXAMPLE_PRECISION: str = 'example_precision'
    EXAMPLE_RECALL: str = 'example_recall'
    EXAMPLE_MCC: str = 'example_mcc'
    EXAMPLE_SPEARMAN: str = 'example_spearman'
    EXAMPLE_PEARSON: str = 'example_pearson'


class DepthMetric(enum.Enum):
    AVERAGE_LOSS_ON_SENTENCE_TOKENS: str = 'average_loss_on_sentence_tokens'
    VARIANCE_LOSS_ON_SENTENCE_TOKENS: str = 'variance_loss_on_sentence_tokens'
    AVERAGE_LOSS_ON_NON_SENTENCE_TOKENS: str = 'average_loss_on_non_sentence_tokens'
    VARIANCE_LOSS_ON_NON_SENTENCE_TOKENS: str = 'variance_loss_on_non_sentence_tokens'


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
    NUM_TRUNCATED_TOKENS: str = 'num_truncated_tokens'


@dataclasses.dataclass
class T5TokenizerConstants(TokenizerConstants):
    START_TOKEN: str = '</s>'
    PAD_TOKEN: str = '<pad>'
    SPACE_TOKEN: str = "▁"
    PAD_TOKEN_ID: int = -100
    TEXT_COLUMN_NAME: str = 'text'
    SENTENCE_PIECE_UNDERSCORE: str = '▁'


# TODO: Add a UL2 tokenizer class.

class DEPTHTokenizerConstants(T5TokenizerConstants):
    SENTENCE_TOKEN: str = "SENT"
    SENT: str = "sent"
    SENT_TOKEN: str = '<sent>'
    END_OF_SENTENCE_TOKEN: str = "<eosen>"
    ADDITIONAL_SPECIAL_TOKENS: str = 'additional_special_tokens'
    NUM_SENT_TOKENS: int = 20


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

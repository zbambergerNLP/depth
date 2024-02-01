import typing

import torch
import datasets
from torch.utils.data import DataLoader
from omegaconf import open_dict
from datasets.iterable_dataset import IterableDataset
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    AutoConfig,
)
import omegaconf
import transformers

from encoder_decoder_utils.data_utils import (
    tokenize_function,
    tokenizer_function_t5_pre_training,
    tokenizer_function_depth_pre_training
)

from encoder_decoder_utils.t5_model import (
    MyT5,
    # Depth,
    DepthForConditionalGeneration
)

from encoder_decoder_utils.constants import (
    ModelImplementation,
    TrainingPhase,
    DatasetSplit,
    T5TokenizerConstants,
    ModelHuggingFaceName, DEPTHTokenizerConstants,
)

from encoder_decoder_utils.data_collator_utils import (
    T5DataCollator,
    DEPTHDataCollator,
)

from encoder_decoder_utils.logging_utils import Logger

from encoder_decoder_utils.tokenizer_utils import DepthTokenizer


def get_model(
        args: omegaconf.DictConfig,
        config: transformers.AutoConfig,
        logger: Logger,
) -> torch.nn.Module:
    """
    Either create or load a T5 model for conditional generation.

    The T5 model we use can be either a HuggingFace T5 model or a locally implemented T5 model.
    Furthermore, we support loading a model from a checkpoint, randomly initializing a model, or loading a model from
    a pretrained checkpoint (e.g., the standard T5-base weights on Huggingface).

    We also save the number of parameters in the model to the args.

    :param args: The omegaconf configuration used to generate the model.
    :param config: The model configuration. See `get_config` for more details.
    :param logger: The logger. See `logging_utils.py` for more details.
    :return: A T5 model for conditional generation.
    """

    logger.log_message(f'Loading {args.model.model_implementation} model')
    # TODO: Review the following code. We may want to customize the hydra and omegaconf code to make this cleaner.
    #  Furthermore, we want to support more than just a T5 architecture (e.g., support DEPTH and UL2 in additional to
    #  the basic T5 architecture).
    model_implementation: torch.nn.Module = {
        ModelImplementation.HUGGINGFACE_T5.value: transformers.T5ForConditionalGeneration,  # HuggingFace T5
        ModelImplementation.LOCAL_T5.value: MyT5,  # TODO: Consider using Megatron LM for this.
        ModelImplementation.DEPTH.value: DepthForConditionalGeneration,
    }[args.model.model_implementation]

    # Randomly initialize the model
    if args.model.random_init:
        logger.log_message('Randomly initializing model')
        model = model_implementation(config)

    # Load the model from a pretrained checkpoint (e.g., the standard T5-base weights on Huggingface)
    else:

        # If the model is DEPTH, load a T5 model from HuggingFace, and then load the weights into DEPTH
        if (
                args.model.model_implementation == ModelImplementation.DEPTH.value or
                args.model.model_implementation == ModelImplementation.LOCAL_T5.value
        ):
            logger.log_message(f'Loading model from pretrained: {args.model.name}')
            t5_model = transformers.T5ForConditionalGeneration.from_pretrained(
                args.model.name,
                config=config,
            )
            weights = t5_model.state_dict()
            model = model_implementation(config)
            model.load_state_dict(weights)

        # If the model is T5, load a T5 model from HuggingFace
        else:
            logger.log_message(f'Loading model from pretrained: {args.model.name}')
            model = model_implementation.from_pretrained(
                args.model.name,
                config=config,
            )

    # Save the number of parameters in the model to the args
    with omegaconf.open_dict(args):
        args.n_all_param = sum([parameter.nelement() for parameter in model.parameters()])
        logger.log_message(f'Number of parameters: {args.n_all_param.__format__("0,")}')

    return model

def get_config(
        args: omegaconf.DictConfig,
        logger: Logger,
) -> transformers.AutoConfig:
    """
    Get the model configuration, which is used to initialize the model.

    :param args: The omegaconf configuration used to generate the model's configuration.
    :param logger: The logger. See `logging_utils.py` for more details.
    :return: The model configuration.
    """
    logger.log_message('Loading model config')

    config = transformers.AutoConfig.from_pretrained(
        args.model.name,
    )

    if hasattr(args.model, 'overwrite'):
        for k, v in args.model.overwrite.items():
            assert hasattr(config, k), f'config does not have attribute {k}'
            setattr(config, k, v)

    if hasattr(args.model, 'add_config'):
        for k, v in args.model.add_config.items():
            assert not hasattr(config, k), f'config already has attribute {k}'
            setattr(config, k, v)

    return config

def get_tokenizer(
        args: omegaconf.DictConfig,
        logger: Logger,
) -> transformers.AutoTokenizer:
    """
    Get the tokenizer. This is used to tokenize the input data.
    :param args: The omegaconf configuration used to generate the tokenizer.
    :param logger: The logger. See `logging_utils.py` for more details.
    :return: The tokenizer.
    """
    # TODO: Enable custom tokenizer
    logger.log_message(f'Loading {args.model.tokenizer} tokenizer')

    if args.model.model_implementation == ModelImplementation.DEPTH.value:
        tokenizer = DepthTokenizer.from_pretrained(
            args.model.name,
            use_fast=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model.tokenizer,
            use_fast=True,
        )
    tokenizer.model_max_length = int(1e9)

    return tokenizer


def load_dataset_splits(
        args: omegaconf.DictConfig,
        logger: Logger,
) -> typing.Dict[str, datasets.Dataset]:
    """
    Load the splits of the dataset (e.g., train, test, validation).

    :param args: The omegaconf configuration used to generate the dataset splits.
    :param logger: A logging_utils.Logger object. See `logging_utils.py` for more details.
    :return: A dictionary of the dataset splits.
    """
    logger.log_message(f'Loading dataset {args.dataset.name} from {args.dataset.path}')

    if args.mode == TrainingPhase.PT.value:

        # TODO: Enable loading multiple datasets and interweaving them.
        dataset = datasets.load_dataset(
            path=args.dataset.path,
            name=args.dataset.name,
            streaming=args.dataset.streaming,
        )

        dataset = dataset.remove_columns(
            args.dataset.columns_to_remove
        )

        training_set = dataset[DatasetSplit.TRAIN.value]
        validation_set = dataset[DatasetSplit.VALIDATION.value]

        # If specified, take a subset of the training and validation sets
        if args.dataset.training_set.num_examples > -1:
            logger.log_message(f'Only using {args.dataset.training_set.num_examples} examples from the training set.')
            training_set = training_set.take(args.dataset.training_set.num_examples)

        if args.dataset.validation_set.num_examples > -1:
            logger.log_message(
                f'Only using {args.dataset.validation_set.num_examples} examples from the validation set.')
            validation_set = validation_set.take(args.dataset.validation_set.num_examples)

        # We want to use the validation set as the test set
        dataset_splits = {
            DatasetSplit.TRAIN.value: training_set,
            DatasetSplit.TEST.value: validation_set,
        }

        assert (
            dataset[DatasetSplit.TRAIN.value].n_shards == args.dataset.num_shards
        ), "We want to have many shards for efficient processing with num_workes in PyTorch dataloader"

    # TODO: Add support for fine-tuning tasks on GLUE, SuperGLUE, DiscoEval, etc...
    elif args.mode == TrainingPhase.FT.value:
        raise NotImplementedError(f'Fine-tuning not implemented for {args.dataset.path}')

    else:
        raise NotImplementedError(f'Unknown mode: {args.mode}')

    return dataset_splits


def process_dataset(
        dataset_splits: typing.Dict[str, datasets.Dataset],
        args: omegaconf.DictConfig,
        tokenizer: transformers.AutoTokenizer,
        logger: Logger,
) -> typing.Dict[str, datasets.Dataset]:
    """
    Process the dataset splits (e.g., tokenize the inputs and outputs).

    :param dataset_splits: A dictionary of the dataset splits. The keys are the split names (e.g., train, test,
        validation) and the values are the dataset splits (i.e., a HuggingFace Dataset object).
    :param args: The omegaconf configuration used to process the dataset splits.
    :param tokenizer: The tokenizer used to tokenize the inputs and outputs.
    :param logger: A logging_utils.Logger object. See `logging_utils.py` for more details.
    :return: A dictionary of the processed dataset splits.
    """
    logger.log_message('Processing dataset splits')

    if args.mode == TrainingPhase.PT.value:
        logger.log_message('Pre-processing for pre-training')
        final_datasets = {}

        for split, dataset_split in dataset_splits.items():

            # We increase the input_length, because instead of masking tokens T5 replaces
            # masked spans with a single token, therefore to avoid padding we need to have
            # longer sequences at the start, before masking
            # before_mask_input_length, target_length = compute_input_and_target_lengths(
            #     inputs_length=args.data.input_length,
            #     noise_density=args.data.mlm_probability,
            #     mean_noise_span_length=args.data.mean_noise_span_length,
            # )
            # logger.log_message(
            #     f'Input_length: {args.data.input_length}\n'
            #     f'Before Mask Input Length: {before_mask_input_length}\n'
            #     f'Target Length: {target_length}'
            # )
            # with open_dict(args):
            #     args.data.before_mask_input_length = before_mask_input_length
            #     args.data.target_length = target_length

            # Merge multiple examples into each input of the language model
            # TODO: Pass in the name of the text column in the dataset. It may not always be 'text'
            if args.dataset.merge_examples:
                logger.log_message('Tokenizing for T5 with merging examples')
                dataset_split = dataset_split.map(
                    tokenize_function,
                    batched=True,
                    fn_kwargs={
                        T5TokenizerConstants.TOKENIZER: tokenizer,
                        T5TokenizerConstants.IN_LENGTH: args.data.input_length,
                    },
                    remove_columns=[args.dataset.text_column]
                )

            # TODO: Add support for DEPTH tokenizer
            elif args.model.model_implementation == ModelImplementation.DEPTH.value:
                logger.log_message('Tokenizing for DEPTH without merging examples')
                dataset_split = dataset_split.map(
                    tokenizer_function_depth_pre_training,
                    batched=True,
                    fn_kwargs={
                        T5TokenizerConstants.TOKENIZER: tokenizer,
                        T5TokenizerConstants.IN_LENGTH: args.data.input_length,
                    },
                    remove_columns=[args.dataset.text_column],
                )

            # Each example corresponds with an input of the language model
            else:
                logger.log_message('Tokenizing for T5 without merging examples')
                dataset_split = dataset_split.map(
                    function=tokenizer_function_t5_pre_training,
                    batched=True,
                    fn_kwargs={
                        T5TokenizerConstants.TOKENIZER: tokenizer,
                        T5TokenizerConstants.IN_LENGTH: args.data.input_length,
                    },
                    remove_columns=[args.dataset.text_column],
                )

            dataset_split = dataset_split.shuffle(buffer_size=args.dataset.buffer_size, seed=args.seed)
            final_datasets[split] = dataset_split

    elif args.mode == TrainingPhase.FT.value:
        logger.log_message('Pre-processing for fine-tuning')

        # TODO: Add support for fine-tuning tasks on GLUE, SuperGLUE, DiscoEval, etc...
        if args.dataset.path == 'glue':
           raise NotImplementedError(f'Fine-tuning not implemented for {args.dataset.path}')
        elif args.dataset.path == 'discoeval':
            raise NotImplementedError(f'Fine-tuning not implemented for {args.dataset.path}')
        else:
            raise NotImplementedError(f'Fine-tuning not implemented for {args.dataset.path}')
    else:
        raise NotImplementedError

    return final_datasets


def get_data_collator(
        tokenizer: transformers.AutoTokenizer,
        config: transformers.AutoConfig,
        args: omegaconf.DictConfig,
        logger: Logger,
) -> typing.Union[T5DataCollator, DEPTHDataCollator]:
    """
    Get the data collator. This is used to collate the data into batches.

    :param tokenizer: The tokenizer used to tokenize the inputs and outputs.
    :param config: The model configuration. See `get_config` for more details.
    :param args: The omegaconf configuration used to generate the data collator.
    :param logger: The logger. See `logging_utils.py` for more details.
    :return: The data collator.
    """
    if args.mode == TrainingPhase.PT.value:
        if args.data.data_collator == 'custom_t5':
            logger.log_message('Using custom T5 data collator')
            data_collator = T5DataCollator(
                tokenizer=tokenizer,
                noise_density=args.data.mlm_probability,
                mean_noise_span_length=args.data.mean_noise_span_length,
                input_length=args.data.input_length,
                target_length=args.data.target_length,
                pad_token_id=config.pad_token_id,
                decoder_start_token_id=config.decoder_start_token_id,
            )
        elif args.data.data_collator == 'depth':
            logger.log_message('Using custom DEPTH data collator')
            data_collator = DEPTHDataCollator(
                tokenizer=tokenizer,
                noise_density=args.data.mlm_probability,
                mean_noise_span_length=args.data.mean_noise_span_length,
                input_length=args.data.input_length,
                target_length=args.data.target_length,
                pad_token_id=config.pad_token_id,
                decoder_start_token_id=config.decoder_start_token_id,
            )
        else:
            raise NotImplementedError(f'Unknown data collator: {args.data.data_collator}')

    # TODO: Add support for fine-tuning tasks on GLUE, SuperGLUE, DiscoEval, etc...
    elif args.mode == TrainingPhase.FT.value:
        raise NotImplementedError(f'Fine-tuning not implemented for {args.dataset.path}')

    else:
        raise NotImplementedError(f'Unknown mode: {args.mode}')

    return data_collator

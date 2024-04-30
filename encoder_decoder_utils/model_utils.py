import typing
import datasets
from transformers import (
    AutoTokenizer,
)
import omegaconf
import transformers

from encoder_decoder_utils import constants
from encoder_decoder_utils import data_utils
from encoder_decoder_utils import t5_model
from encoder_decoder_utils import data_collator_utils

from encoder_decoder_utils.logging_utils import Logger
from encoder_decoder_utils.tokenizer_utils import DepthTokenizer

from fine_tune_constants.glue_constants import GlueConstants, TaskConfigOneInput, TaskConfigTwoInput
from fine_tune_constants.disco_eval_constants import DiscoEvalConstants, DiscoEvalTaskConfig


def get_model(
        args: omegaconf.DictConfig,
        config: transformers.PretrainedConfig,
        logger: Logger,
        tokenizer: transformers.PreTrainedTokenizer,
        last_checkpoint: typing.Optional[str] = None,
) -> typing.Union[
    transformers.T5ForConditionalGeneration,
    t5_model.DepthForConditionalGeneration,
]:
    """
    Either create or load a T5 model for conditional generation.

    The T5 model we use can be either a HuggingFace T5 model or a locally implemented T5 model.
    Furthermore, we support loading a model from a checkpoint, randomly initializing a model, or loading a model from
    a pretrained checkpoint (e.g., the standard T5-base weights on Huggingface).

    We also save the number of parameters in the model to the args.

    :param args: The omegaconf configuration used to generate the model.
    :param config: The model configuration. See `get_config` for more details.
    :param logger: The logger. See `logging_utils.py` for more details.
    :param tokenizer: The tokenizer used to tokenize the inputs and outputs.
    :param last_checkpoint: The path to the last checkpoint. If specified, the model will be loaded from the last
    :return: A T5 model for conditional generation.
    """

    logger.log_message(f'Loading {args.model.model_implementation} model')
    # TODO: Review the following code. We may want to customize the hydra and omegaconf code to make this cleaner.
    #  Furthermore, we want to support more than just a T5 architecture (e.g., support DEPTH and UL2 in additional to
    #  the basic T5 architecture).
    model_implementation = {
        constants.ModelImplementation.HUGGINGFACE_T5.value: transformers.T5ForConditionalGeneration,  # HuggingFace T5
        constants.ModelImplementation.DEPTH.value: t5_model.DepthForConditionalGeneration,
    }[args.model.model_implementation]

    logger.log_message(f'config is:\n{config}')

    # Randomly initialize the model
    if args.model.random_init:
        logger.log_message('Randomly initializing model')
        model = model_implementation(config)

    # Load the model from a pretrained checkpoint (e.g., the standard T5-base weights on Huggingface)
    else:

        # If the model is DEPTH, load a T5 model from HuggingFace, and then load the weights into DEPTH
        if args.model.model_implementation == constants.ModelImplementation.DEPTH.value:
            logger.log_message(f'Loading model from pretrained: {args.model.name}')

            if last_checkpoint is not None:
                logger.log_message(f'Loading model from checkpoint: {last_checkpoint}')
                model = t5_model.DepthForConditionalGeneration.from_pretrained(
                    pretrained_model_name_or_path=last_checkpoint,
                    config=config,
                    ignore_mismatched_sizes=True,
                )

            else:
                logger.log_message(f'Loading model from pretrained: {args.model.name}')
                base_model = transformers.T5ForConditionalGeneration.from_pretrained(
                    args.model.name,
                    config=config,
                    ignore_mismatched_sizes=True,
                )
                weights = base_model.state_dict()
                model = model_implementation(config)
                model.load_state_dict(weights)

        # Load the model from a local checkpoint
        elif last_checkpoint is not None:
            logger.log_message(f'Loading model from checkpoint: {last_checkpoint}')
            model = model_implementation.from_pretrained(
                last_checkpoint,
                config=config,
                ignore_mismatched_sizes=True,
            )

        # If the model is from HuggingFace, load it from the hub by name.
        else:
            logger.log_message(f'Loading model from pretrained: {args.model.name}')
            model = model_implementation.from_pretrained(
                args.model.name,
                config=config,
            )

    # Resize the model's input and output embeddings if the tokenizer's vocabulary size is different from the model's
    if len(tokenizer) != model.config.vocab_size:
        logger.log_message(
            f'Resizing model embeddings (from {model.config.vocab_size}) to match tokenizer vocabulary size: '
            f'{len(tokenizer)}')
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    else:
        logger.log_message(f'Model embeddings match tokenizer vocabulary size: {len(tokenizer)}')

    # Save the number of parameters in the model to the args
    with omegaconf.open_dict(args):
        args.n_all_param = sum([parameter.nelement() for parameter in model.parameters()])
        logger.log_message(f'Number of parameters: {args.n_all_param.__format__("0,")}')

    return model


def get_config(
        args: omegaconf.DictConfig,
        logger: Logger,
        last_checkpoint: typing.Optional[str] = None,
) -> transformers.PretrainedConfig:
    """
    Get the model configuration, which is used to initialize the model.

    :param args: The omegaconf configuration used to generate the model's configuration.
    :param logger: The logger. See `logging_utils.py` for more details.
    :param last_checkpoint: The path to the last checkpoint. If specified, the model configuration will be loaded from
        the last checkpoint.
    :return: The model configuration.
    """
    logger.log_message('Loading model config')

    config = transformers.AutoConfig.from_pretrained(
        pretrained_model_name_or_path=args.model.name if last_checkpoint is None else last_checkpoint,
    )

    if hasattr(args.model, 'overwrite'):
        for k, v in args.model.overwrite.items():
            if hasattr(config, k):
                setattr(config, k, v)
            else:
                logger.log_message(f'config does not have attribute {k}')

    if hasattr(args.model, 'add_config'):
        for k, v in args.model.add_config.items():
            if hasattr(config, k):
                logger.log_message(f'config already has attribute {k}')
            else:
                setattr(config, k, v)

    return config


def get_tokenizer(
        args: omegaconf.DictConfig,
        logger: Logger,
) -> transformers.PreTrainedTokenizer:
    """
    Get the tokenizer. This is used to tokenize the input data.
    :param args: The omegaconf configuration used to generate the tokenizer.
    :param logger: The logger. See `logging_utils.py` for more details.
    :return: The tokenizer.
    """
    # TODO: Enable custom tokenizer
    logger.log_message(f'Loading {args.model.tokenizer} tokenizer')

    if args.model.model_implementation == constants.ModelImplementation.DEPTH.value:
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
    if args.mode == constants.TrainingPhase.PT.value:
        logger.log_message(f'Loading dataset {args.dataset.name} from {args.dataset.path}')

        # TODO: Enable loading multiple datasets and interweaving them.
        dataset = datasets.load_dataset(
            path=args.dataset.path,
            name=args.dataset.name,
            streaming=args.dataset.streaming,
            trust_remote_code=args.dataset.path in constants.TRUSTED_DATASETS,
        )

        dataset = dataset.remove_columns(
            args.dataset.columns_to_remove
        )

        training_set = dataset[constants.DatasetSplit.TRAIN.value]
        validation_set = dataset[constants.DatasetSplit.VALIDATION.value]

        logger.log_message(f'Loaded dataset splits: {training_set}')
        logger.log_message(f'Loaded dataset splits: {validation_set}')
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
            constants.DatasetSplit.TRAIN.value: training_set,
            constants.DatasetSplit.TEST.value: validation_set,
        }
        logger.log_message(f'Loaded dataset splits: {list(dataset_splits.keys())}')
        logger.log_message(f'Loaded dataset splits: {list(dataset_splits.values())}')
        assert (
                dataset[constants.DatasetSplit.TRAIN.value].n_shards == args.dataset.num_shards
        ), "We want to have many shards for efficient processing with num_workes in PyTorch dataloader"

    # TODO: Add support for fine-tuning tasks on GLUE, SuperGLUE, DiscoEval, etc...
    elif args.mode == constants.TrainingPhase.FT.value:
        logger.log_message(
            f'Loading dataset {args.downstream.benchmark_dataset} from {args.downstream.benchmark_constants}'
        )
        benchmark_name = args.downstream.benchmark_constants
        dataset_name = args.downstream.benchmark_dataset
        if benchmark_name == constants.DownstreamDataset.GLUE.value:
            dataset = datasets.load_dataset(
                benchmark_name,
                dataset_name,
                streaming=args.dataset.streaming,
            )
        elif benchmark_name == constants.DownstreamDataset.DISCO_EVAL.value:
            dataset = datasets.load_dataset(
                benchmark_name,
                dataset_name,
                streaming=args.dataset.streaming,
            )
        elif benchmark_name == "ni":
            dataset = datasets.load_dataset(
                './encoder_decoder_utils/ni_dataset.py',
                data_dir='./data/splits/default',
                task_dir='./data/tasks',
                max_num_instances_per_task=100,
                max_num_instances_per_eval_task=100
            )
            return dataset

        else:
            raise NotImplementedError(f'Unknown benchmark name: {benchmark_name}')

        # TODO: Use constants instead of literal string
        if benchmark_name == constants.DownstreamDataset.GLUE.value:
            if dataset_name == constants.GLUEConstants.MNLI:
                training_set = dataset[constants.DatasetSplit.TRAIN.value]
                if args.downstream.mnli_sub_dir == constants.GLUEConstants.MISMATCHED:
                    validation_set = dataset['validation_mismatched']
                    test_set = dataset['test_mismatched']
                else:
                    validation_set = dataset['validation_matched']
                    test_set = dataset['test_matched']
            else:
                training_set = dataset[constants.DatasetSplit.TRAIN.value]
                validation_set = dataset[constants.DatasetSplit.VALIDATION.value]
                test_set = dataset[constants.DatasetSplit.TEST.value]
        else:
            training_set = dataset[constants.DatasetSplit.TRAIN.value]
            validation_set = dataset[constants.DatasetSplit.VALIDATION.value]
            test_set = dataset[constants.DatasetSplit.TEST.value]
        dataset_splits = {
            constants.DatasetSplit.TRAIN.value: training_set,
            constants.DatasetSplit.VALIDATION.value: validation_set,
            constants.DatasetSplit.TEST.value: test_set,
        }
    else:
        raise NotImplementedError(f'Unknown mode: {args.mode}')

    return dataset_splits


def process_dataset(
        dataset_splits: typing.Dict[str, datasets.Dataset],
        args: omegaconf.DictConfig,
        tokenizer: transformers.PreTrainedTokenizer,
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
    if args.mode == constants.TrainingPhase.PT.value:
        logger.log_message('Pre-processing for pre-training')
        final_datasets = {}

        for split, dataset_split in dataset_splits.items():
            # Merge multiple examples into each input of the language model
            # TODO: Pass in the name of the text column in the dataset. It may not always be 'text'
            if args.dataset.merge_examples:
                logger.log_message('Tokenizing for T5 with merging examples')
                if args.dataset.streaming:
                    dataset_split = dataset_split.map(
                        data_utils.tokenize_function,
                        batched=True,
                        fn_kwargs={
                            constants.T5TokenizerConstants.TOKENIZER: tokenizer,
                            constants.T5TokenizerConstants.IN_LENGTH: args.data.input_length,
                        },
                        remove_columns=[args.dataset.text_column],
                    )
                else:
                    dataset_split = dataset_split.map(
                        data_utils.tokenize_function,
                        batched=True,
                        fn_kwargs={
                            constants.T5TokenizerConstants.TOKENIZER: tokenizer,
                            constants.T5TokenizerConstants.IN_LENGTH: args.data.input_length,
                        },
                        remove_columns=[args.dataset.text_column],
                        num_proc=args.data.num_workers,
                        desc=f'Tokenizing {split}',
                    )

            # TODO: Add support for DEPTH tokenizer
            elif args.model.model_implementation == constants.ModelImplementation.DEPTH.value:
                logger.log_message('Tokenizing for DEPTH without merging examples')

                if args.dataset.streaming:
                    dataset_split = dataset_split.map(
                        data_utils.tokenizer_function_depth_pre_training,
                        batched=True,
                        batch_size=args.optim.batch_size,
                        fn_kwargs={
                            constants.T5TokenizerConstants.TOKENIZER: tokenizer,
                            constants.T5TokenizerConstants.IN_LENGTH: args.data.input_length,
                        },
                        remove_columns=[args.dataset.text_column],
                    )
                else:
                    dataset_split = dataset_split.map(
                        data_utils.tokenizer_function_depth_pre_training,
                        batched=True,
                        batch_size=args.optim.batch_size,
                        fn_kwargs={
                            constants.T5TokenizerConstants.TOKENIZER: tokenizer,
                            constants.T5TokenizerConstants.IN_LENGTH: args.data.input_length,
                        },
                        remove_columns=[args.dataset.text_column],
                        num_proc=args.data.num_workers,
                        desc=f'Tokenizing {split}',
                    )

            # Each example corresponds with an input of the language model
            else:
                logger.log_message('Tokenizing for T5 without merging examples')
                if args.dataset.streaming:
                    dataset_split = dataset_split.map(
                        function=data_utils.tokenizer_function_t5_pre_training,
                        batched=True,
                        batch_size=args.optim.batch_size,
                        fn_kwargs={
                            constants.T5TokenizerConstants.TOKENIZER: tokenizer,
                            constants.T5TokenizerConstants.IN_LENGTH: args.data.input_length,
                        },
                        remove_columns=[args.dataset.text_column],
                    )
                else:
                    dataset_split = dataset_split.map(
                        function=data_utils.tokenizer_function_t5_pre_training,
                        batched=True,
                        batch_size=args.optim.batch_size,
                        fn_kwargs={
                            constants.T5TokenizerConstants.TOKENIZER: tokenizer,
                            constants.T5TokenizerConstants.IN_LENGTH: args.data.input_length,
                        },
                        remove_columns=[args.dataset.text_column],
                        num_proc=args.data.num_workers,
                        desc=f'Tokenizing {split}',
                    )

            # Shuffle the dataset
            dataset_split = dataset_split.shuffle(buffer_size=args.dataset.buffer_size, seed=args.seed)
            final_datasets[split] = dataset_split

    elif args.mode == constants.TrainingPhase.FT.value:
        logger.log_message('Pre-processing for fine-tuning')
        if args.downstream.benchmark_constants == 'ni':
            final_datasets = dataset_splits
        else:
            ft_constants = GlueConstants() if args.downstream.benchmark_constants == 'glue' else DiscoEvalConstants()
            logger.log_message(f'Fine-tuning constants: {ft_constants}')
            # TODO: Add support for fine-tuning tasks on GLUE, SuperGLUE, DiscoEval, etc...
            final_datasets = {}
            dataset_name = args.downstream.benchmark_dataset
            for split, dataset_split in dataset_splits.items():
                logger.log_message('Tokenizing for T5 without merging examples')
                preprocessing_function = data_utils.create_preprocess_function(
                    dataset_info=ft_constants[dataset_name],
                    dataset_name=dataset_name,
                    tokenizer=tokenizer,
                    args=args,
                    logger=logger,
                )
                logger.log_message(f'preprocessing function: {preprocessing_function}')
                if isinstance(ft_constants[dataset_name], TaskConfigOneInput):
                    remove_columns = [
                        ft_constants[dataset_name].TEXT_COLUMN_NAME,
                        ft_constants[dataset_name].LABEL_COLUMN_NAME,
                    ] + ['idx']
                elif isinstance(ft_constants[dataset_name], TaskConfigTwoInput):
                    remove_columns = [
                        ft_constants[dataset_name].TEXT_COLUMN_NAME_1,
                        ft_constants[dataset_name].TEXT_COLUMN_NAME_2,
                        ft_constants[dataset_name].LABEL_COLUMN_NAME,
                    ] + ['idx']
                elif isinstance(ft_constants[dataset_name], DiscoEvalTaskConfig):
                    remove_columns = [
                                         ft_constants.TEXT_COLUMN_NAMES[i] for i in
                                         range(ft_constants[dataset_name].TEXT_COLUMN_AMOUNT)
                                     ] + [ft_constants[dataset_name].LABEL_COLUMN_NAME]
                else:
                    raise NotImplementedError(f'Unknown task config: {ft_constants[dataset_name]}')

                if args.dataset.streaming:
                    dataset_split = dataset_split.map(
                        function=preprocessing_function,
                        batched=True,
                        batch_size=args.optim.batch_size,
                        remove_columns=remove_columns,
                    )
                else:
                    dataset_split = dataset_split.map(
                        function=preprocessing_function,
                        batched=True,
                        batch_size=args.optim.batch_size,
                        remove_columns=remove_columns,
                        num_proc=args.data.num_workers,
                        desc=f'Tokenizing {split}',
                    )
                # This is nessesary, in glue, the test set should not be shuffled.
                if (
                        args.downstream.benchmark_constants == constants.DownstreamDataset.GLUE and
                        not split == constants.DatasetSplit.TEST.value
                ):
                    if args.dataset.streaming:
                        dataset_split = dataset_split.shuffle(buffer_size=args.dataset.buffer_size, seed=args.seed)
                    else:
                        dataset_split = dataset_split.shuffle(seed=args.seed)
                final_datasets[split] = dataset_split

    else:
        raise NotImplementedError

    return final_datasets


def get_data_collator(
        tokenizer: typing.Union[transformers.PreTrainedTokenizer, DepthTokenizer],
        config: transformers.PretrainedConfig,
        args: omegaconf.DictConfig,
        logger: Logger,
) -> typing.Union[
    data_collator_utils.T5DataCollator,
    data_collator_utils.DEPTHDataCollator,
]:
    """
    Get the data collator. This is used to collate the data into batches.

    :param tokenizer: The tokenizer used to tokenize the inputs and outputs.
    :param config: The model configuration. See `get_config` for more details.
    :param args: The omegaconf configuration used to generate the data collator.
    :param logger: The logger. See `logging_utils.py` for more details.
    :return: The data collator.
    """
    input_length = args.data.input_length
    target_length = args.data.target_length

    if args.mode == constants.TrainingPhase.PT.value:
        if args.data.data_collator == 'custom_t5':  # TODO: Make this a constant
            logger.log_message('Using custom T5 data collator')
            data_collator = data_collator_utils.T5DataCollator(
                tokenizer=tokenizer,
                noise_density=args.data.mlm_probability,
                mean_noise_span_length=args.data.mean_noise_span_length,
                input_length=input_length,
                target_length=target_length,
                pad_token_id=tokenizer.pad_token_id,
                decoder_start_token_id=tokenizer.pad_token_id,
            )

        elif args.data.data_collator == constants.ModelImplementation.DEPTH.value:
            logger.log_message('Using custom DEPTH data collator')
            data_collator = data_collator_utils.DEPTHDataCollator(
                tokenizer=tokenizer,
                noise_density=args.data.mlm_probability,
                mean_noise_span_length=args.data.mean_noise_span_length,
                input_length=input_length,
                target_length=args.data.target_length,
                pad_token_id=target_length,
                decoder_start_token_id=tokenizer.pad_token_id,
                sentence_shuffling_probability=args.data.sentence_shuffling_probability,
            )
        else:
            raise NotImplementedError(f'Unknown data collator: {args.data.data_collator}')

    elif args.mode == constants.TrainingPhase.FT.value:
        logger.log_message('Using HuggingFace data collator')
        if args.model.model_implementation == constants.ModelImplementation.DEPTH.value:
            data_collator = data_collator_utils.DEPTHDataCollatorFineTuning(
                tokenizer=tokenizer,
                input_length=args.data.input_length,
                target_length=args.data.target_length,
                pad_token_id=tokenizer.pad_token_id,
                decoder_start_token_id=tokenizer.pad_token_id,
            )
        else:
            if args.downstream.benchmark_constants == 'ni':
                data_collator = data_collator_utils.DataCollatorForNI(
                    tokenizer=tokenizer,
                    padding="longest",
                    max_source_length=512,
                    max_target_length=128,
                    label_pad_token_id=-100,
                    pad_to_multiple_of=8,
                    add_task_name=False,
                    add_task_definition=True,
                    num_pos_examples=2,
                    num_neg_examples=0,
                    add_explanation=False,
                    tk_instruct=False,
                )
            else:
                data_collator = transformers.DataCollatorForSeq2Seq(
                    tokenizer=tokenizer,
                    label_pad_token_id=tokenizer.pad_token_id,
                )
    else:
        raise NotImplementedError(f'Unknown mode: {args.mode}')

    return data_collator

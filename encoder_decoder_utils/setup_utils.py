import time
import numpy as np
import collections
import typing
import omegaconf
import logging
import datasets
import transformers
import os
import neptune
import accelerate

from encoder_decoder_utils import constants
from encoder_decoder_utils.constants import (
    MonitoringPlatform,
)
import wandb
import torch


class Experiment:

    date_time = time.strftime("%Y-%m-%d_%H-%M")

    def __init__(
            self,
            dict_config: omegaconf.DictConfig
    ):
        self.output_dir = dict_config.checkpoint.output_dir
        self.training_phase = (
            "fine_tune" if dict_config.mode == constants.TrainingPhase.FT.value else "pre_train"
        )
        self.model_implementation = dict_config.model.model_implementation
        self.checkpoint_origin = self._get_checkpoint_origin(
            random_init=dict_config.model.random_init,
            mode=dict_config.mode,
            model_implementation=self.model_implementation,
            checkpoint_path=dict_config.checkpoint.checkpoint_path,
        )
        self.dataset = (
            f"{dict_config.dataset.path}_{dict_config.dataset.name}" if
            dict_config.mode == constants.TrainingPhase.PT.value else
            f"{dict_config.downstream.benchmark_constants}_{dict_config.downstream.benchmark_dataset}"
        )
        self.hparams = self._get_hparams(
            dict_config.optim.base_lr,
            dict_config.optim.lr_scheduler,
            dict_config.optim.batch_size,
            dict_config.data.sentence_shuffling_probability,
        )
        self.date_time = time.strftime("%Y-%m-%d_%H-%M")
        self.experiment_name = self._create_name()
        self.experiment_path = self._create_path()


    @staticmethod
    def _get_checkpoint_origin(
            random_init: bool,
            mode: str,
            model_implementation: str,
            checkpoint_path: str,
    ):
        if (
                checkpoint_path
        ):
            return "from_pretrained" if "from_pretrained" in checkpoint_path else "from_scratch"
        else:
            # If the model is fine-tuning and the checkpoint path is not provided, then the model is fine-tuning from
            # the baseline model (the checkpoint available via the huggingface model hub).
            if mode == constants.TrainingPhase.FT.value:
                # Only the baseline does not accept a checkpoint path during fine-tuning.
                return "baseline"
            else:
                # If fine-tuning, we can determine the checkpoint origin from the random_init flag.
                return "from_scratch" if random_init else "from_pretrained"


    @staticmethod
    def _get_hparams(
            learning_rate: float,
            scheduler: str,
            batch_size: int,
            shuffling_probability: float,
    ) -> str:
        return '_'.join(
            [
                f"lr_{str(learning_rate).replace('.', '_')}",
                scheduler,
                f"bsz_{str(batch_size)}",
                f"shuffle_p_{str(shuffling_probability).replace('.', '_')}"
            ]
        )

    def _create_path(self) -> str:
        """
        Get the path to the experiment folder.
        :return: The path to the experiment folder.
        """
        hparams = self.hparams
        return '/'.join([
            self.output_dir,
            self.training_phase,
            str(self.checkpoint_origin),
            self.model_implementation,
            self.dataset,
            hparams,
            self.date_time
        ])

    def _create_name(self) -> str:
        """
        Get the name of the experiment.
        :return: The name of the experiment.
        """
        hparams = self.hparams
        return '_'.join([
            self.training_phase,
            str(self.checkpoint_origin),
            self.model_implementation,
            self.dataset,
            hparams,
            self.date_time
        ])

    @property
    def path(self) -> str:
        return self.experiment_path

    @property
    def name(self) -> str:
        return self.experiment_name

    def set_path(self, path: str):
        self.experiment_path = path

    def set_name(self, name: str):
        self.experiment_name = name


class Logger:
    def __init__(
            self,
            args: omegaconf.DictConfig,
            accelerator: accelerate.Accelerator,
            experiment: Experiment,
    ):
        """
        Initialize the logger.

        Set up the logger for the main process and set the verbosity of the transformers and datasets libraries.

        :param args: The arguments for the run. A hydra config which contains the model, data, and training arguments.
            See the configs/default.yaml file for the default configuration.
        :param accelerator: The accelerator object which will be used to train the model.
        :param experiment: The experiment object which defines the experiment hyper-parameters. Use to construct the
            name of the run for logging.
        """
        self.experiment = experiment
        self.logger = accelerate.logging.get_logger('Main')

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        self.logger.info(accelerator.state, main_process_only=False)
        self.logger.info(f'Working directory is {os.getcwd()}')

        if accelerator.is_local_main_process:
            # Log everything on the main process
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            # Only log errors on the other processes
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()

        self.accelerator = accelerator

        # Setup the neptune and wandb loggers according to the user's preferences in the args
        self.neptune_logger =self.setup_neptune(args)
        self.wandb_logger = self.setup_wandb(args)

    def setup_neptune(
            self, args: omegaconf.DictConfig,
    ) -> typing.Union[neptune.Run, None]:
        """
        Setup the neptune logger.

        Initialize the neptune logger if the user wants to log to neptune. Otherwise, set the neptune logger to None.

        :param args: The arguments for the run. A hydra config which contains the model, data, and training arguments.
            See the configs/default.yaml file for the default configuration.
        """
        if args.logging.neptune:
            self.log_message('Initializing neptune')
            neptune_logger = neptune.init_run(
                project=args.logging.neptune_creds.project,
                api_token=args.logging.neptune_creds.api_token,
                tags=[str(item) for item in args.logging.neptune_creds.tags.split(",")],
            )
        else:
            return None

        self.neptune_logger = neptune_logger

        # Retrieve the neptune id of the run and add it to the args
        with omegaconf.open_dict(args):
            if neptune_logger is not None:
                args.neptune_id = neptune_logger["sys/id"].fetch()
        return neptune_logger


    def setup_wandb(self, args: omegaconf.DictConfig) -> typing.Union[wandb.sdk.wandb_run.Run, None]:
        """
        Setup the wandb logger.

        Initialize the wandb logger if the user wants to log to wandb. Otherwise, set the wandb logger to None.

        :param args: The arguments for the run. A hydra config which contains the model, data, and training arguments.
            See the configs/default.yaml file for the default configuration.
        """
        experiment_name = self.experiment.name
        if args.logging.wandb:
            self.accelerator.init_trackers(
                project_name=args.logging.wandb_creds.project,
                # Create a dictionary from args and pass it to wandb as a config
                config=dict(args),
                init_kwargs={
                    MonitoringPlatform.WANDB.value: {
                        'name': experiment_name,    # The name of the experiment run
                        'job_type': args.mode,      # Either 'pt' for pre-training or 'ft' for fine-tuning
                    }
                }
            )
            wandb_logger = self.accelerator.get_tracker(MonitoringPlatform.WANDB.value, unwrap=True)
        else:
            wandb_logger = None

        return wandb_logger

    def log_args(
            self,
            args: omegaconf.DictConfig,
    ):
        """
        Log the omegaconfig (arguments/hyperparameters) to neptune.

        :param args: The arguments for the run. A hydra config which contains the model, data, and training arguments.
            See the configs/default.yaml file for the default configuration.
        """
        all_args = {}
        logging_args = omegaconf.OmegaConf.to_container(args, resolve=True)

        # Flatten the nested dictionary
        def flatten(d, parent_key='', sep='_'):
            items = []
            for k, v in d.items():
                new_key = f'{parent_key}{sep}{k}' if parent_key else k
                if isinstance(v, collections.MutableMapping):
                    items.extend(flatten(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        for k, v in flatten(logging_args).items():
            all_args[k] = v

        if self.neptune_logger is not None:
            self.neptune_logger['args'] = all_args
        if self.wandb_logger and self.accelerator.is_local_main_process:
            self.wandb_logger.config.update(all_args)

    def log_stats(
            self,
            stats: typing.Dict[str, float],
            step: int,
            args: omegaconf.DictConfig,
            prefix: str ='',
    ):
        """
        Log the statistics to neptune.

        As part of logging the statistics, the statistics are averaged over the number of updates and the averaged
        statistics are logged to neptune. The averaged statistics are also logged to the logger.

        :param stats: A dictionary of statistics to log. This dictionary should be of the form
            {'stat_name': stat_value}, where stat_name is a string and stat_value is a float representing the value
            of the statistic (e.g. {'loss': 0.5}).
        :param step: The step at which the statistics were logged.
        :param args: The arguments for the run. A hydra config which contains the model, data, and training arguments.
            See the configs/default.yaml file for the default configuration.
        :param prefix: The prefix to add to the statistics when logging to neptune. Default is an empty string. For
            example, if the prefix is 'train_', then the statistics will be logged to neptune as 'train_{stat_name}'.
        """
        if self.neptune_logger is not None:
            for k, v in stats.items():
                self.neptune_logger[f'{prefix}{k}'].log(v, step=step)
        if self.wandb_logger and self.accelerator.is_local_main_process:
            wandb.log(stats, step=step)

    def log_message(self, msg: str):
        """
        Log a message to the logger.

        Create a lightweight printout of a message via the logger.

        :param msg: The message to log.
        """
        self.logger.info(msg)

    def finish(self):
        """
        Finish logging. Stop the neptune logger if it is not None.
        """
        if self.neptune_logger is not None:
            self.neptune_logger.stop()
        if self.wandb_logger is not None:
            self.wandb_logger.finish()


def set_seed(
        seed: int = 2137
):
    """
    Set the seed for all the random number generators.
    :param seed: The seed to set. If None, the seed will not be set.
    :return:
    :rtype:
    """
    transformers.set_seed(seed)
    transformers.enable_full_determinism(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    accelerate.utils.set_seed(seed)

def check_args_and_env(
        args: omegaconf.DictConfig
):
    """
    Check if the arguments and environment variables are valid.

    :param args: The hydra config which contains the model, data, and training arguments. See the
        configs/default.yaml file for the default configuration.
    """
    assert args.optim.batch_size % args.optim.grad_acc == 0, \
        'Batch size must be divisible by grad_acc\n' \
        f'Batch size: {args.optim.batch_size}, grad_acc: {args.optim.grad_acc}'

    # Train log must happen before eval log
    assert args.evaluate.every_steps % args.logging.every_steps == 0

    if args.device == constants.Device.GPU.value:
        assert (
            torch.cuda.is_available(),
            'You selected to use a GPU, but CUDA is not available on your machine.'
        )

    assert not (args.eval_only and args.predict_only), \
        'Cannot both only evaluate and only predict.'

    if args.predict_only:
        assert args.mode == constants.TrainingPhase.FT.value, \
            'Predict only works in fine-tuning mode, but the current mode is pre-training (pt)'


def opti_flags(
        args: omegaconf.DictConfig
):
    """
    Enable more effective cuda operations, and utilize bf16 precision if appropriate.

    :param args: The hydra config which contains the model, data, and training arguments. See the
        configs/default.yaml file for the default configuration.
    """
    # This lines reduce training step by 2.4x
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if (
            args.precision == constants.NumericalPrecision.BF16 and
            args.device == constants.Device.GPU.value and
            args.model.klass == constants.ModelImplementation.LOCAL_T5.value
    ):
        args.model.add_config.is_bf16 = True


def update_args_with_env_info(
        args: omegaconf.DictConfig,
):
    """
    Update the arguments with environment variables.

    :param args: The hydra config which contains the model, data, and training arguments. See the
        configs/default.yaml file for the default configuration.
    """
    with omegaconf.open_dict(args):
        slurm_id = os.getenv(constants.EnvironmentVariable.SLURM_JOB_ID.value)

        if slurm_id is not None:
            args.slurm_id = slurm_id
        else:
            args.slurm_id = 'none'

        args.working_dir = os.getcwd()


def setup_basics(
        accelerator: accelerate.Accelerator,
        args: omegaconf.DictConfig,
        experiment: Experiment,
) -> Logger:
    """
    Setup the logger and accelerator.
    :param accelerator: The accelerator object which will be used to train the model.
    :param args: The hydra config which contains the model, data, and training arguments. See the
        configs/default.yaml file for the default configuration.
    :param experiment: The experiment object which defines the experiment hyper-parameters. Use to construct the
        name of the run for logging.
    :return: The logger object which will be used to log the training and evaluation results.
    """
    check_args_and_env(args)
    update_args_with_env_info(args)
    # update_paths(args)
    opti_flags(args)

    if args.seed is not None:
        set_seed(args.seed)

    print(f'experiment is: {experiment.name}')
    logger = Logger(args=args, accelerator=accelerator, experiment=experiment)

    return logger

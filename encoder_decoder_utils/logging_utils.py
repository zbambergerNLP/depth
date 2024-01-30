import typing
from collections import defaultdict

import omegaconf
import logging
import datasets
import transformers
import os
import neptune
import accelerate
from encoder_decoder_utils.constants import (
    MonitoringPlatform,
    TagCategory,
)
import wandb
import torch

class Logger:
    def __init__(self, args: omegaconf.DictConfig, accelerator: accelerate.Accelerator):
        """
        Initialize the logger.

        Set up the logger for the main process and set the verbosity of the transformers and datasets libraries.

        :param args: The arguments for the run. A hydra config which contains the model, data, and training arguments.
            See the configs/default.yaml file for the default configuration.
        :param accelerator: The accelerator object which will be used to train the model.
        """
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
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
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
        # TODO: Create an experiment name based on the model name and learning rate and save it as the name of the
        #  wandb run
        if args.logging.wandb:
            tags = self._create_tags(args)
            self.accelerator.init_trackers(
                project_name=args.logging.wandb_creds.project,
                # Create a dictionary from args and pass it to wandb as a config
                config=dict(args),
                init_kwargs={
                    MonitoringPlatform.WANDB.value: {
                        'tags': tags,           # A list of tags to filter runs by on the wandb dashboard
                        'job_type': args.mode,  # Either 'pt' for pre-training or 'ft' for fine-tuning
                    }
                }
            )
            wandb_logger = self.accelerator.get_tracker(MonitoringPlatform.WANDB.value, unwrap=True)
        else:
            wandb_logger = None

        return wandb_logger

    def _create_tags(self, args: omegaconf.DictConfig):
        """
        Create tags the experiment run.

        Create tags on the basis of:
        - model name
        - dataset name
        - batch size
        - learning rate
        - number of epochs
        - number of steps
        - model implementation (e.g. huggingface, vs. Megatron LM, vs. local)
        - precision (e.g. fp32, fp16, etc.)
        - number of GPUs

        :param args: The arguments for the run. A hydra config which contains the model, data, and training arguments.
            See the configs/default.yaml file for the default configuration.
        :return: A list of string tags to add to the experiment run.
        """
        tags = [
            f'{TagCategory.MODEL.value}_{args.model.name}',
            f'{TagCategory.DATASET.value}_{args.dataset.path}',
            f'{TagCategory.BATCH_SIZE.value}_{args.optim.batch_size}',
            f'{TagCategory.BASE_LR.value}_{args.optim.base_lr}',
            f'{TagCategory.LR_SCHEDULER.value}_{args.optim.lr_scheduler}',
            f'{TagCategory.EPOCHS.value}_{args.optim.epochs}',
            f'{TagCategory.STEPS.value}_{args.optim.total_steps}',
            f'{TagCategory.IMPLEMENTATION.value}_{args.model.model_implementation}',
            f'{TagCategory.PRECISION.value}_{args.precision}',
            f'{TagCategory.NUM_PROCESSES.value}_{self.accelerator.state.num_processes}',
            f'{TagCategory.NUM_GPUS.value}_{torch.cuda.device_count()}',
        ]
        return tags

    def log_args(self, args: omegaconf.DictConfig):
        """
        Log the omegaconfig (arguments/hyperparameters) to neptune.

        :param args: The arguments for the run. A hydra config which contains the model, data, and training arguments.
            See the configs/default.yaml file for the default configuration.
        """

        logging_args = omegaconf.OmegaConf.to_container(args, resolve=True)
        if self.neptune_logger is not None:
            self.neptune_logger['args'] = logging_args

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
        # TODO: Keep track of total training time
        # TODO: Predict and visualize the expected time to completion
        if self.neptune_logger is not None:
            for k, v in stats.items():
                self.neptune_logger[f'{prefix}{k}'].log(v, step=step)
        if self.wandb_logger and self.accelerator.is_local_main_process:
            wandb.log(stats, step=step)

        # TODO: Create this message using a tqdm progress bar. The progress bar should be updated every time the
        #  statistics are logged. The progress bar should be updated with the current step and the total number of
        #  steps. The progress bar should also be updated with the current statistics (e.g. loss, learning rate, etc.)
        # msg_start = f'[{prefix[:-1]}] Step {step} out of {args.optim.total_steps}' + ' | '
        # dict_msg = ' | '.join([f'{k.capitalize()} --> {v:.3f}' for k, v in stats.items()]) + ' | '
        #
        # msg = msg_start + dict_msg
        #
        # self.log_message(msg)

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
            self.wandb_logger.close()


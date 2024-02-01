import torch
import os

import omegaconf
import hydra
import accelerate
import transformers
import numpy as np

# Local imports
from encoder_decoder_utils.constants import (
    TrainingPhase,
    Device,
    NumericalPrecision,
    ModelImplementation,
    EnvironmentVariable,
)
from encoder_decoder_utils.logging_utils import Logger


def set_seed(seed: int = 2137):
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

def check_args_and_env(args: omegaconf.DictConfig):
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

    if args.device == Device.GPU.value:
        assert (
            torch.cuda.is_available(),
            'You selected to use a GPU, but CUDA is not available on your machine.'
        )

    assert not (args.eval_only and args.predict_only), \
        'Cannot both only evaluate and only predict.'

    if args.predict_only:
        assert args.mode == TrainingPhase.FT.value, \
            'Predict only works in fine-tuning mode, but the current mode is pre-training (pt)'


def opti_flags(args: omegaconf.DictConfig):
    """
    Enable more effective cuda operations, and utilize bf16 precision if appropriate.

    :param args: The hydra config which contains the model, data, and training arguments. See the
        configs/default.yaml file for the default configuration.
    """
    # This lines reduce training step by 2.4x
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if (
            args.precision == NumericalPrecision.BF16 and
            args.device == Device.GPU.value and
            args.model.klass == ModelImplementation.LOCAL_T5.value
    ):
        args.model.add_config.is_bf16 = True


def update_args_with_env_info(args: omegaconf.DictConfig):
    """
    Update the arguments with environment variables.

    :param args: The hydra config which contains the model, data, and training arguments. See the
        configs/default.yaml file for the default configuration.
    """
    with omegaconf.open_dict(args):
        slurm_id = os.getenv(EnvironmentVariable.SLURM_JOB_ID.value)

        if slurm_id is not None:
            args.slurm_id = slurm_id
        else:
            args.slurm_id = 'none'

        args.working_dir = os.getcwd()


def setup_basics(accelerator: accelerate.Accelerator, args: omegaconf.DictConfig) -> Logger:
    """
    Setup the logger and accelerator.
    :param accelerator: The accelerator object which will be used to train the model.
    :param args: The hydra config which contains the model, data, and training arguments. See the
        configs/default.yaml file for the default configuration.
    :return: The logger object which will be used to log the training and evaluation results.
    """
    check_args_and_env(args)
    update_args_with_env_info(args)
    # update_paths(args)
    opti_flags(args)

    if args.seed is not None:
        set_seed(args.seed)

    logger = Logger(args=args, accelerator=accelerator)

    return logger

import os
import argparse
from huggingface_hub import HfApi

api = HfApi()

"""
Example run command:

# From-Scratch T5
python upload_to_hf_hub.py \
--model_name hf_t5 \
--model_path checkpoints/pre_train/from_scratch/hf_t5/c4_en/lr_0_0001_inverse_sqrt_bsz_200_shuffle_p_0_5/2024-03-18_21-25 \
--checkpoint_upper_bound 1000000 \
--checkpoint_lower_bound 0 \
--repo_name depth \
--user_name zbambergerNLP 

# From-Scratch Depth
python upload_to_hf_hub.py \
--model_name depth \
--model_path checkpoints/pre_train/from_scratch/depth/c4_en/lr_0_0001_linear_bsz_200_shuffle_p_0_5/2024-03-11_18-50 \
--checkpoint_upper_bound 1000000 \
--checkpoint_lower_bound 0 \
--repo_name depth \
--user_name zbambergerNLP 

# From-Pretrained T5
python upload_to_hf_hub.py \
--model_name hf_t5 \
--model_path checkpoints/pre_train/from_pretrained/hf_t5/allenai/c4_en/lr_0_0001_inverse_sqrt_bsz_200_shuffle_p_0_5/2024-04-03_20-31 \
--checkpoint_upper_bound 256000 \
--checkpoint_lower_bound 0 \
--repo_name depth \
--user_name zbambergerNLP

# From-Pretrained Depth
python upload_to_hf_hub.py \
--model_name depth \
--model_path checkpoints/pre_train/from_pretrained/depth/allenai_c4_en/lr_0_0001_linear_bsz_200_shuffle_p_0_5/2024-04-09_19-16 \
--checkpoint_upper_bound 256000 \
--checkpoint_lower_bound 0 \
--repo_name depth \
--user_name zbambergerNLP

"""

T5_PATH = os.path.join(
    'checkpoints',
    'pre_train',
    'from_scratch',
    'hf_t5',
    'c4_en',
    'lr_0_0001_linear_bsz_200_shuffle_p_0_5',
    '2024-03-12_13-26',
)

DEPTH_PATH = os.path.join(
    'checkpoints',
    'pre_train',
    'from_scratch',
    'depth',
    'c4_en',
    'lr_0_0001_linear_bsz_200_shuffle_p_0_5',
    '2024-03-11_18-50',
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    default='hf_t5',
    help="The name of the model to upload to the Hugging Face model hub.",
)
parser.add_argument(
    "--model_path",
    type=str,
    default=os.path.join(
        'checkpoints',
        'pre_train',
        'from_scratch',
        'hf_t5',
        'c4_en',
        'lr_0_0001_linear_bsz_200_shuffle_p_0_5',
        '2024-03-12_13-26',
    ),
    help="The path to the model checkpoint (saved locally).",
)
parser.add_argument(
    "--checkpoint_upper_bound",
    type=int,
    default=512_000,
    help="The latest checkpoint to upload to the Hugging Face model hub."
)
parser.add_argument(
    "--checkpoint_lower_bound",
    type=int,
    default=0,
    help="The earliest checkpoint to upload to the Hugging Face model hub."
)
parser.add_argument(
    "--repo_name",
    type=str,
    default='depth',
    help="The name of the repository to which the model should be uploaded.",
)
parser.add_argument(
    "--user_name",
    type=str,
    default='zbambergerNLP',
    help="The name of the user who owns the repository to which the model should be uploaded.",
)
parser.add_argument(
    "--private",
    action="store_true",
    help="Whether the model should be uploaded as private.",
)


def upload_to_hf_hub(
        model_name: str,
        model_path: str,
        user_name: str,
        repo_name: str,
        upper_bound: int = 512_000,
        lower_bound: int = 0,
):
    """
    Save a model to the Hugging Face model hub.
    :param model_name: The name of the model.
    :param model_path: The path to the model checkpoint (saved locally)
    :param user_name: The name of the user who owns the repository to which the model should be uploaded.
    :param repo_name: The name of the repository to which the model should be uploaded.
    :param upper_bound: The latest checkpoint to upload to the Hugging Face model hub.
    :param lower_bound: The earliest checkpoint to upload to the Hugging Face model hub.
    """
    print(f'model name is: {model_name}')
    print(f'model path is: {model_path}')

    # Identify all of the checkpoint directories under the specified model path
    checkpoint_dirs = [
        os.path.join(model_path, d)  # Join the model path with the directory name for the resulting paths
        for d in os.listdir(model_path)  # List all files and directories in the model path
        if (os.path.isdir(os.path.join(model_path, d)) and d.startswith('checkpoint'))  # Must be a checkpoint directory
    ]

    for checkpoint_path in checkpoint_dirs:

        if not (lower_bound <= int(checkpoint_path.rsplit('-')[-1]) <= upper_bound):
            print(f'Skipping checkpoint {checkpoint_path} (outside of range {lower_bound} to {upper_bound})')
            continue

        print(f'checkpoint_path: {checkpoint_path}')
        api.upload_folder(
            folder_path=checkpoint_path,
            repo_id=f'{user_name}/{repo_name}',
            path_in_repo=checkpoint_path,
            commit_message=f"Upload model ({model_name}) to HuggingFace Hub.\n\tcheckpoint {checkpoint_path}",
        )


if __name__ == '__main__':
    args = parser.parse_args()
    upload_to_hf_hub(
        model_name=args.model_name,
        model_path=args.model_path,
        repo_name=args.repo_name,
        user_name=args.user_name,
        upper_bound=args.checkpoint_upper_bound,
        lower_bound=args.checkpoint_lower_bound,
    )

import os
import csv
import typing

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.colors as mcolors


FILES_IN_RUN = [
    'eval_loss.csv',
    'train_loss.csv',
    'train_global_step.csv',
]

FILES_IN_DEPTH_RUN = [
    'eval_reconstruction_loss.csv',
    'eval_sentence_loss.csv',
]

LINESTYLES = ['-', '--', '-.', ':']

def format_label(label: str) -> str:
    """Capitalize the first letter of each word in the label."""
    return label.replace('_', ' ').title()


def process_run(
        run_path: str,
        min_step: int,
        max_step: int,
) -> typing.Dict[str, typing.Dict[int, float]]:
    """
    Load the files containing the training and validation metrics, and return a dictionary containing the metrics.
    :param run_path: The path to the run directory. Must contain the following files:
        - eval_loss.csv: The validation loss values for each evaluation step.
        - train_loss.csv: The training loss values for each training step.
        - eval_sentence_loss.csv: The sentence loss values for each evaluation step. Only applicable to the DEPTH model.
        - train_global_step.csv: A mapping from each training step to the global step.
        - train_loss.csv: The training loss values for each training step.
    :type run_path:
    :return: A dictionary containing the metrics. The keys are 'train_loss', 'eval_loss', 'eval_sentence_loss', and
        'eval_reconstruction_loss'. The values are dictionaries mapping the global step to the corresponding loss value.
    """
    results = {}
    # Get the mapping from training step to global step
    with open(os.path.join(run_path, 'train_global_step.csv'), 'r') as f:
        reader = csv.reader(f)
        train_global_step = list(reader)
        train_global_step = {int(row[0]): int(row[1]) for row in train_global_step[1:]}
    # print(f"train_global_step:\n{train_global_step}")

    # Map the global step to the training loss values
    with open(os.path.join(run_path, 'train_loss.csv'), 'r') as f:
        reader = csv.reader(f)
        train_loss = list(reader)
        train_loss = {train_global_step[int(row[0])]: float(row[1]) for row in train_loss[1:]}
        results['train_loss'] = train_loss
    # print(f"train_loss:\n{train_loss}")

    # Map the global step to the evaluation loss values
    with open(os.path.join(run_path, 'eval_loss.csv'), 'r') as f:
        reader = csv.reader(f)
        eval_loss = list(reader)
        eval_loss = {train_global_step[int(row[0])]: float(row[1]) for row in eval_loss[1:]}
        results['eval_loss'] = eval_loss
    # print(f"eval_loss:\n{eval_loss}")

    if 'depth' in run_path:
        # Map the global step to the sentence loss values
        with open(os.path.join(run_path, 'eval_sentence_loss.csv'), 'r') as f:
            reader = csv.reader(f)
            eval_sentence_loss = list(reader)
            eval_sentence_loss = {train_global_step[int(row[0])]: float(row[1]) for row in eval_sentence_loss[1:]}
            results['eval_sentence_loss'] = eval_sentence_loss
        # print(f"eval_sentence_loss:\n{eval_sentence_loss}")

        # Map the global step to the reconstruction loss values
        with open(os.path.join(run_path, 'eval_reconstruction_loss.csv'), 'r') as f:
            reader = csv.reader(f)
            eval_reconstruction_loss = list(reader)
            eval_reconstruction_loss = {
                train_global_step[int(row[0])]: float(row[1]) for row in eval_reconstruction_loss[1:]
            }
            results['eval_reconstruction_loss'] = eval_reconstruction_loss
        # print(f"eval_reconstruction_loss:\n{eval_reconstruction_loss}")

    # Ensure that the results are within the specified range
    for metric_name, metric_dict in results.items():
        results[metric_name] = {
            step: loss for step, loss in metric_dict.items() if min_step <= step <= max_step
        }
    return results


def process_and_merge_runs(
        run_files: typing.List[typing.Dict[str, typing.Union[str, int]]],
        interval: int = 1_000,
) -> typing.Dict[str, typing.Dict[int, float]]:

    # Initialize the results dictionary
    results = {
        "train_loss": {},
        "eval_loss": {},
    }
    if 'depth' in run_files[0]["run_path"]:
        results["eval_sentence_loss"] = {}
        results["eval_reconstruction_loss"] = {}

    # Process each run
    for run in run_files:
        run_results = process_run(
            run["run_path"],
            run["min_step"],
            run["max_step"],
        )

        # Filter steps within each metric, and merge the results into a unified dictionary
        for metric_name, metric_dict in run_results.items():
            for step, metric_value in metric_dict.items():
                if step % interval == 0:
                    results[metric_name][step] = metric_value
    return results


def plot_recreating_t5_experiments():

    T5_COLOR = mcolors.CSS4_COLORS.get('darkorange')
    T5_HIGH_LR_PACKING_COLOR = mcolors.CSS4_COLORS.get('mediumpurple')
    T5_HIGH_LR_NO_PACKING_COLOR = mcolors.CSS4_COLORS.get('green')

    # Legacy T5 model
    t5_lr_0_01_files = [
        {
            'run_path': 'hf_t5/lr_0_01_inverse_sqrt_bsz_128_adamw/0-42k',
            'min_step': 0,
            'max_step': 42_000
        }
    ]
    t5_lr_0_01_results = process_and_merge_runs(t5_lr_0_01_files)
    print(f"t5_lr_0_01_results:\n{t5_lr_0_01_results}")

    # Plot NanoT5 results
    nanot5_train_loss_path = "hf_t5/nanoT5_legacy/train_loss.csv"
    nanot5_eval_loss_path = "hf_t5/nanoT5_legacy/eval_loss.csv"
    nanot5_results = {
        "train_loss": {},
        "eval_loss": {},
    }
    with open(nanot5_train_loss_path, 'r') as f:
        reader = csv.reader(f)
        train_loss = list(reader)
        print(f'train_loss:\n{train_loss}')
        train_loss = {int(float(row[0])): float(row[2]) for row in train_loss}
        nanot5_results['train_loss'] = train_loss
    with open(nanot5_eval_loss_path, 'r') as f:
        reader = csv.reader(f)
        eval_loss = list(reader)
        print(f'eval_loss:\n{eval_loss}')
        eval_loss = {int(float(row[0])): float(row[2]) for row in eval_loss}
        nanot5_results['eval_loss'] = eval_loss

    # Plot T5 train and eval losses for lr=0.01 versus the baseline T5 model (lr=0.0001)
    plt.figure(figsize=(5, 5), layout='constrained')
    for metric_name, metric_dict in t5_from_scratch_results.items():
        plt.plot(
            list(metric_dict.keys()),
            list(metric_dict.values()),
            label=f'T5 (No Packing, low LR) {format_label(metric_name)}',
            color=T5_COLOR,
            linestyle=LINESTYLES[0] if 'train' in metric_name.lower() else LINESTYLES[1],
        )
    for metric_name, metric_dict in t5_lr_0_01_results.items():
        plt.plot(
            list(metric_dict.keys()),
            list(metric_dict.values()),
            label=f'T5 (No packing, high LR) {format_label(metric_name)}',
            color=T5_HIGH_LR_NO_PACKING_COLOR,
            linestyle=LINESTYLES[0] if 'train' in metric_name.lower() else LINESTYLES[1],
        )

    # plt.figure(figsize=(5, 5), layout='constrained')
    for metric_name, metric_dict in nanot5_results.items():
        print(f"metric_dict:\n{metric_dict}")
        plt.plot(
            list(metric_dict.keys()),
            list(metric_dict.values()),
            label=f'T5 (Packing, high LR) {format_label(metric_name)}',
            color=T5_HIGH_LR_PACKING_COLOR,
            linestyle=LINESTYLES[0] if 'train' in metric_name.lower() else LINESTYLES[1],
        )

    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(5_000))
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.xticks(rotation=45)
    plt.ylim(0.3, 5)
    plt.xlim(0, 42_000)
    plt.yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.5, 1, 2, 3, 4, 5])
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    image_file_name = "replicating_t5.pdf"
    image_path = os.path.join(current_working_directory, "figures", image_file_name)
    if os.path.exists(image_path):
        os.remove(image_path)
    plt.savefig(image_path)
    plt.show()


if __name__ == "__main__":

    depth_from_scratch_files = [
        {
            "run_path": "depth/lr_0_0001_bsz_200_linear_decay_adamw/0-48k",
            "min_step": 0,
            "max_step": 48_000,
        },
        {
            "run_path": "depth/lr_0_0001_bsz_200_linear_decay_adamw/48k-70k",
            "min_step": 48_000,
            "max_step": 70_000,
        },
        {
            "run_path": "depth/lr_0_0001_bsz_200_linear_decay_adamw/70k-251k",
            "min_step": 70_000,
            "max_step": 251_000,
        },
        {
            "run_path": "depth/lr_0_0001_bsz_200_linear_decay_adamw/260k-307k",
            "min_step": 260_000,
            "max_step": 307_000,
        },
        {
            "run_path": "depth/lr_0_0001_bsz_200_linear_decay_adamw/307k-654k",
            "min_step": 307_000,
            "max_step": 654_000,
        },
        {
            "run_path": "depth/lr_0_0001_bsz_200_linear_decay_adamw/654k-765k",
            "min_step": 654_000,
            "max_step": 765_000,
        },
        {
            "run_path": "depth/lr_0_0001_bsz_200_linear_decay_adamw/764k-878k",
            "min_step": 764_000,
            "max_step": 878_000,
        },
        {
            "run_path": "depth/lr_0_0001_bsz_200_linear_decay_adamw/878k-894k",
            "min_step": 878_000,
            "max_step": 894_000,
        },
        {
            "run_path": "depth/lr_0_0001_bsz_200_linear_decay_adamw/894k-918k",
            "min_step": 894_000,
            "max_step": 918_000,
        },
        {
            "run_path": "depth/lr_0_0001_bsz_200_linear_decay_adamw/918k-924k",
            "min_step": 918_000,
            "max_step": 924_000,
        },
        {
            "run_path": "depth/lr_0_0001_bsz_200_linear_decay_adamw/938k-1M",
            "min_step": 938_000,
            "max_step": 1_000_000,
        }
    ]

    t5_from_scratch_files = [
        {
            "run_path": "hf_t5/lr_0_0001_bsz_200_inverse_sqrt_decay_adamw/0-10k",
            "min_step": 0,
            "max_step": 10_000,
        },
        {
            "run_path": "hf_t5/lr_0_0001_bsz_200_inverse_sqrt_decay_adamw/10k-42k",
            "min_step": 10_000,
            "max_step": 42_000,
        },
        {
            "run_path": "hf_t5/lr_0_0001_bsz_200_inverse_sqrt_decay_adamw/42k-246k",
            "min_step": 42_000,
            "max_step": 246_000,
        },
        {
            "run_path": "hf_t5/lr_0_0001_bsz_200_inverse_sqrt_decay_adamw/246k-310k",
            "min_step": 246_000,
            "max_step": 310_000,
        },
        {
            "run_path": "hf_t5/lr_0_0001_bsz_200_inverse_sqrt_decay_adamw/310k-498k",
            "min_step": 310_000,
            "max_step": 498_000,
        },
        {
            "run_path": "hf_t5/lr_0_0001_bsz_200_inverse_sqrt_decay_adamw/498k-527k",
            "min_step": 498_000,
            "max_step": 527_000,
        },
        {
            "run_path": "hf_t5/lr_0_0001_bsz_200_inverse_sqrt_decay_adamw/520k-623k",
            "min_step": 527_000,
            "max_step": 623_000,
        },
        {
            "run_path": "hf_t5/lr_0_0001_bsz_200_inverse_sqrt_decay_adamw/623k-662k",
            "min_step": 623_000,
            "max_step": 662_000,
        },
        {
            "run_path": "hf_t5/lr_0_0001_bsz_200_inverse_sqrt_decay_adamw/662k-705k",
            "min_step": 662_000,
            "max_step": 705_000,
        },
        {
            "run_path": "hf_t5/lr_0_0001_bsz_200_inverse_sqrt_decay_adamw/705k-820k",
            "min_step": 705_000,
            "max_step": 820_000,
        },
        {
            "run_path": "hf_t5/lr_0_0001_bsz_200_inverse_sqrt_decay_adamw/820k-939k",
            "min_step": 820_000,
            "max_step": 939_000,
        },
        {
            "run_path": "hf_t5/lr_0_0001_bsz_200_inverse_sqrt_decay_adamw/938k-978k",
            "min_step": 939_000,
            "max_step": 978_000,
        },
        {
            "run_path": "hf_t5/lr_0_0001_bsz_200_inverse_sqrt_decay_adamw/976k-1M",
            "min_step": 978_000,
            "max_step": 1_000_000,
        }
    ]
    depth_from_scratch_results = process_and_merge_runs(depth_from_scratch_files)
    t5_from_scratch_results = process_and_merge_runs(t5_from_scratch_files)

    current_working_directory = os.getcwd()

    # # Print the results (for debugging)
    # for metric_name, metric_dict in depth_from_scratch_results.items():
    #     print(f"\n\nDEPTH {format_label(metric_name)}:\n{metric_dict}")
    #
    # for metric_name, metric_dict in t5_from_scratch_results.items():
    #     print(f"\n\nT5 {format_label(metric_name)}:\n{metric_dict}")
    #
    # Plot the results

    DEPTH_COLOR = mcolors.CSS4_COLORS.get('blue')
    T5_COLOR = mcolors.CSS4_COLORS.get('darkorange')

    # plot_recreating_t5_experiments()

    # T5 Eval Loss vs. DEPTH Eval Reconstruction Loss
    plt.figure(figsize=(5, 5), layout='constrained')
    plt.plot(
        list(t5_from_scratch_results['eval_loss'].keys()),
        list(t5_from_scratch_results['eval_loss'].values()),
        label=format_label('T5 Reconstruction Loss'),
        color=T5_COLOR
    )
    plt.plot(
        list(depth_from_scratch_results['eval_reconstruction_loss'].keys()),
        list(depth_from_scratch_results['eval_reconstruction_loss'].values()),
        label=format_label('DEPTH Reconstruction Loss'),
        color=DEPTH_COLOR
    )
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(100_000))
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.xlabel("Step")
    plt.ylabel("Reconstruction Loss (Validation)")
    plt.xlim(0, 1_000_000)
    plt.ylim(0.3, 5)
    plt.yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.5, 1, 2, 3, 4, 5])
    # Make the figure taller than the default
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.xticks(rotation=45)
    # plt.title("T5 vs. DEPTH Reconstruction Loss")
    plt.tick_params(axis='x', labelsize=8)
    x_formatter = ticker.FuncFormatter(lambda x, pos: '{:,}'.format(int(x)))
    plt.gca().xaxis.set_major_formatter(x_formatter)
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.legend()
    # Overwrite the image if it already exists
    image_file_name = os.path.join(current_working_directory, "figures", "t5_vs_depth_reconstruction_loss.png")
    if os.path.exists(image_file_name):
        os.remove(image_file_name)
    plt.savefig(image_file_name)
    plt.show()

    # Plot DEPTH losses (Reconstruction, Sentence, Eval) vs. Step in a single plot
    plt.figure(figsize=(5, 5), layout='constrained')
    for metric_name, metric_dict in depth_from_scratch_results.items():
        plt.plot(
            list(metric_dict.keys()),
            list(metric_dict.values()),
            label=format_label(metric_name)
        )
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(100_000))
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.xticks(rotation=45)
    plt.ylim(0.3, 5)
    plt.yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.5, 1, 2, 3, 4, 5])
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.gca().xaxis.set_major_formatter(x_formatter)
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    image_file_name = os.path.join(current_working_directory, "figures", "depth_losses.png")
    if os.path.exists(image_file_name):
        os.remove(image_file_name)
    plt.savefig(image_file_name)
    plt.show()


    # Plot from-pretrained results
    depth_from_pretrained_files = [
        {
            "run_path": "depth/from_pretrained/0-106k",
            "min_step": 0,
            "max_step": 106_000,
        },
        {
            "run_path": "depth/from_pretrained/106k-158k",
            "min_step": 106_000,
            "max_step": 158_000,
        },
        {
            "run_path": "depth/from_pretrained/158k-256k",
            "min_step": 158_000,
            "max_step": 256_000,
        }
    ]

    t5_from_pretrained_files = [
        {
            "run_path": "hf_t5/from_pretrained/0-64k",
            "min_step": 0,
            "max_step": 64_000,
        },
        {
            "run_path": "hf_t5/from_pretrained/64k-263k",
            "min_step": 64_000,
            "max_step": 263_000,
        },
    ]

    depth_from_pretrained_results = process_and_merge_runs(depth_from_pretrained_files)
    t5_from_pretrained_results = process_and_merge_runs(t5_from_pretrained_files)

    plt.figure(figsize=(5, 5), layout='constrained')

    # Plot only the reconstruction loss for DEPTH and the eval loss for T5
    plt.plot(
        list(t5_from_pretrained_results['eval_loss'].keys()),
        list(t5_from_pretrained_results['eval_loss'].values()),
        label=format_label('T5 Eval Loss'),
        color=T5_COLOR
    )
    plt.plot(
        list(depth_from_pretrained_results['eval_reconstruction_loss'].keys()),
        list(depth_from_pretrained_results['eval_reconstruction_loss'].values()),
        label=format_label('DEPTH Reconstruction Loss'),
        color=DEPTH_COLOR
    )
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(25_000))
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.xticks(rotation=45)
    plt.ylim(0.3, 5)
    plt.xlim(0, 256_000)
    plt.yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5])
    plt.xlabel("Step")
    plt.ylabel("Reconstruction Loss (Validation)")
    plt.legend()
    image_file_name = "from_pretrained_losses.png"
    image_path = os.path.join(current_working_directory, "figures", image_file_name)
    if os.path.exists(image_path):
        os.remove(image_path)
    plt.savefig(image_path)
    plt.show()

    # Plot all of the losses for DEPTH
    plt.figure(figsize=(5, 5), layout='constrained')
    for metric_name, metric_dict in depth_from_pretrained_results.items():
        plt.plot(
            list(metric_dict.keys()),
            list(metric_dict.values()),
            label=format_label(metric_name)
        )
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(25_000))
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.xticks(rotation=45)
    plt.ylim(0.3, 5)
    plt.xlim(0, 256_000)
    plt.yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.5, 1, 2, 3, 4, 5])
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    image_file_name = "from_pretrained_depth_losses.png"
    image_path = os.path.join(current_working_directory, "figures", image_file_name)
    if os.path.exists(image_path):
        os.remove(image_path)
    plt.savefig(image_path)
    plt.show()



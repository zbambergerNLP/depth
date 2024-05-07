# A script meant to analyze the dataset for the DEPTH and T5 pre-training tasks.
import torch
import transformers
import datasets
import accelerate

from encoder_decoder_utils.data_collator_utils import (
    DEPTHDataCollator,
    T5DataCollator
)
from encoder_decoder_utils.tokenizer_utils import DepthTokenizer
from encoder_decoder_utils.data_utils import (
    tokenizer_function_depth_pre_training,
    tokenizer_function_t5_pre_training
)


def main(
        num_proc: int = 64,
):
    # Load the C4 dataset in streaming mode
    dataset = datasets.load_dataset(
        path="allenai/c4",
        name="en",
        streaming=True,
        trust_remote_code=True,
    )
    dataset.remove_columns(column_names=["timestamp", "url"])

    # Load the T5 and DEPTH tokenizers
    t5_tokenizer = transformers.AutoTokenizer.from_pretrained(
        "google/t5-v1_1-base",
        use_fast=True,
    )
    depth_tokenizer = DepthTokenizer.from_pretrained(
        "google/t5-v1_1-base",
        use_fast=True,
    )

    # tokenize the dataset
    tokenized_depth_dataset = {}
    tokenized_t5_dataset = {}
    for split in dataset.keys():
        print(f'Processing split: {split}')
        tokenized_depth_dataset[split] = dataset[split].map(
            tokenizer_function_depth_pre_training,
            batched=True,
            batch_size=200,
            fn_kwargs={
                "tokenizer": depth_tokenizer,
                "in_length": 512,
            },
            remove_columns=["text", "timestamp", "url"]
        ).shuffle(
            buffer_size=1000,
            seed=42,
        )
        tokenized_t5_dataset[split] = dataset[split].map(
            tokenizer_function_t5_pre_training,
            batched=True,
            batch_size=200,
            fn_kwargs={
                "tokenizer": t5_tokenizer,
                "in_length": 512,
            },
            remove_columns=["text", "timestamp", "url"],
        ).shuffle(
            buffer_size=1000,
            seed=42,
        )

    # Create the data collators
    t5_data_collator = T5DataCollator(
        tokenizer=t5_tokenizer,
        noise_density=0.3,
        mean_noise_span_length=3,
        input_length=512,
        target_length=512,
        pad_token_id=0,
        decoder_start_token_id=0,
    )

    depth_data_collator = DEPTHDataCollator(
        tokenizer=depth_tokenizer,
        noise_density=0.3,
        mean_noise_span_length=3,
        input_length=512,
        target_length=512,
        pad_token_id=0,
        decoder_start_token_id=0,
        sentence_shuffling_probability=0.5,
    )

    # Create the data loaders
    t5_data_loader = torch.utils.data.DataLoader(
        dataset=tokenized_t5_dataset["train"],
        batch_size=200,
        collate_fn=t5_data_collator,
        num_workers=num_proc,
    )

    depth_data_loader = torch.utils.data.DataLoader(
        dataset=tokenized_depth_dataset["train"],
        batch_size=200,
        collate_fn=depth_data_collator,
        num_workers=num_proc,
    )

    # Create the accelerator and prepare the data loaders
    accelerator = accelerate.Accelerator(
        dispatch_batches=False,
    )
    t5_data_loader = accelerator.prepare(t5_data_loader)
    depth_data_loader = accelerator.prepare(depth_data_loader)

    # Iterate over the first 10,000 batches from the data loaders
    # for batch in t5_data_loader:
    results = {
        't5_num_non_padding_input_ids': 0,
        't5_num_non_padding_target_ids': 0,
        'depth_num_non_padding_input_ids': 0,
        'depth_num_non_padding_target_ids': 0,
    }

    for batch_idx, batch in enumerate(depth_data_loader):

        print(f'\n\nBatch Index: {batch_idx}')

        input_ids = batch["input_ids"]
        decoder_input_ids = batch["target_ids"]

        non_padding_input_ids = input_ids[input_ids != 0]
        non_padding_decoder_input_ids = decoder_input_ids[decoder_input_ids != 0]

        results['depth_num_non_padding_input_ids'] += len(non_padding_input_ids)
        results['depth_num_non_padding_target_ids'] += len(non_padding_decoder_input_ids)

        print(f'Depth Non-Padding Input IDs: {results["depth_num_non_padding_input_ids"]}')
        print(f'Depth Non-Padding Target IDs: {results["depth_num_non_padding_target_ids"]}')

        if batch_idx == 10000:
            break

    print(f"Results: {results}")


if __name__ == '__main__':
    main(num_proc=50)

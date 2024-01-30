import transformers
import torch

class EncoderDecoderTrainer(transformers.Seq2SeqTrainer):
    def __init__(
            self,
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)


    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        batch_size = self._train_batch_size

        # dataloaders[split] = DataLoader(
        return self.accelerator.prepare(
            torch.utils.data.DataLoader(
                train_dataset,
                collate_fn=data_collator,
                batch_size=batch_size,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                drop_last=False,
            )
        )

    def get_eval_dataloader(self, eval_dataset):
        data_collator = self.data_collator
        return self.accelerator.prepare(
            torch.utils.data.DataLoader(
                eval_dataset,
                collate_fn=data_collator,
                batch_size=self.args.eval_batch_size,
                # batch_size=batch_size,
                pin_memory=self.args.dataloader_pin_memory,
                drop_last=False,
                num_workers=self.args.dataloader_num_workers,
            )
        )

    def get_test_dataloader(self, test_dataset):
        data_collator = self.data_collator
        batch_size = self._train_batch_size
        return self.accelerator.prepare(
            torch.utils.data.DataLoader(
                test_dataset,
                # shuffle=False,
                collate_fn=data_collator,
                batch_size=self.args.eval_batch_size,
                pin_memory=self.args.dataloader_pin_memory,
                drop_last=False,
                num_workers=self.args.dataloader_num_workers,
                # num_workers=8,  # The maximum number of workers on the eval set
            )
        )

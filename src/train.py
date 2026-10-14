from pathlib import Path
import os

import torch
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from processor import build_processor
from dataset import build_datasets
from collator import DataCollatorCTC
from model import build_model
from metrics import compute_metrics


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"


class CTCTrainer(Trainer):
    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if state_dict is None:
            state_dict = self.model.state_dict()

        self.model.save_pretrained(output_dir, state_dict=state_dict, save_embedding_layers=False)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def train():
    processor = build_processor()
    train_dataset, test_dataset = build_datasets()
    collator = DataCollatorCTC(processor)
    model = build_model()

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    training_args = TrainingArguments(output_dir=str(OUTPUT_DIR), per_device_train_batch_size=4, per_device_eval_batch_size=4, gradient_accumulation_steps=4, learning_rate=1e-4, num_train_epochs=10, warmup_steps=0.1, eval_strategy="epoch", save_strategy="epoch", logging_steps=25, save_total_limit=3, load_best_model_at_end=True, metric_for_best_model="wer", greater_is_better=False, fp16=use_fp16, bf16=use_bf16, remove_unused_columns=False, label_names=["labels"], report_to="none")

    trainer = CTCTrainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset, data_collator=collator, compute_metrics=compute_metrics)

    last_checkpoint = get_last_checkpoint(str(OUTPUT_DIR)) if OUTPUT_DIR.exists() else None

    if last_checkpoint:
        print("Resuming from:", last_checkpoint)
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("Starting training from scratch")
        trainer.train()

    final_model_dir = OUTPUT_DIR / "final_model"

    trainer.save_model(str(final_model_dir))
    processor.save_pretrained(str(final_model_dir))

    print("Training finished.")
    print("Final model saved to:", final_model_dir)


if __name__ == "__main__":
    train()

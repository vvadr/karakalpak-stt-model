import csv
import os

import torch
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

import config
from collator import DataCollatorCTC
from dataset import load_split
from evaluate_model import evaluate_tests
from metrics import argmax_logits, make_compute_metrics
from model import build_model, gpu_supports_bf16
from processor import build_processor


class TrainingLoop(Trainer):
    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir, state_dict=state_dict, save_embedding_layers=False)
        self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def training_arguments():
    return TrainingArguments(
        output_dir=str(config.CHECKPOINT_DIR),
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION,
        learning_rate=config.LEARNING_RATE,
        warmup_steps=config.WARMUP_RATIO,
        bf16=gpu_supports_bf16(),
        gradient_checkpointing=config.GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        train_sampling_strategy="group_by_length",
        length_column_name="duration",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        logging_steps=50,
        dataloader_num_workers=config.NUM_WORKERS,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="none",
        seed=config.SEED,
    )


def latest_checkpoint():
    if not config.CHECKPOINT_DIR.exists():
        return None
    return get_last_checkpoint(str(config.CHECKPOINT_DIR))


def validation_history(log_history):
    history = []
    for log in log_history:
        if "eval_wer" in log:
            history.append({
                "epoch": round(log["epoch"]),
                "step": log["step"],
                "loss": round(log["eval_loss"], 4),
                "wer": round(log["eval_wer"], 4),
                "cer": round(log["eval_cer"], 4),
            })
    return history


def save_history(log_history):
    history = validation_history(log_history)
    with open(config.OUTPUT_DIR / "history.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "step", "loss", "wer", "cer"])
        writer.writeheader()
        writer.writerows(history)

    print("\nValidation WER/CER per epoch:")
    for row in history:
        print(f"  epoch {row['epoch']:>2}: WER {row['wer']:.2%}  CER {row['cer']:.2%}  loss {row['loss']}")


def save_final_model(trainer, processor):
    model = trainer.model.merge_and_unload()
    model.save_pretrained(config.FINAL_MODEL_DIR)
    processor.save_pretrained(config.FINAL_MODEL_DIR)
    print("Best model saved to:", config.FINAL_MODEL_DIR)


def main():
    processor = build_processor()
    model = build_model(processor)

    train_data = load_split(config.TRAIN_CSVS, config.TRAIN_SUBSET_SIZE)
    validation_data = load_split(config.VALIDATION_CSVS, config.EVAL_SUBSET_SIZE)
    print(f"Train: {len(train_data)} files ({sum(train_data['duration']) / 3600:.2f} h)   Validation: {len(validation_data)} files")

    train = TrainingLoop(
        model=model,
        args=training_arguments(),
        train_dataset=train_data,
        eval_dataset=validation_data,
        data_collator=DataCollatorCTC(processor),
        processing_class=processor,
        compute_metrics=make_compute_metrics(processor),
        preprocess_logits_for_metrics=argmax_logits,
    )

    checkpoint = latest_checkpoint()
    print("Resuming from:", checkpoint or "nothing, starting from scratch")
    train.train(resume_from_checkpoint=checkpoint)

    save_history(train.state.log_history)
    save_final_model(train, processor)
    evaluate_tests(config.FINAL_MODEL_DIR)


if __name__ == "__main__":
    main()

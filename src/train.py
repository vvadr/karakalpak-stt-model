import csv

import torch
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

import config
from collator import DataCollatorCTC
from dataset import load_split
from evaluate_model import evaluate_tests
from metrics import argmax_logits, make_compute_metrics
from model import build_model
from processor import build_processor


def save_history(log_history):
    rows = [{"epoch": round(log["epoch"]), "step": log["step"], "loss": round(log["eval_loss"], 4), "wer": round(log["eval_wer"], 4), "cer": round(log["eval_cer"], 4)} for log in log_history if "eval_wer" in log]

    with open(config.OUTPUT_DIR / "history.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "step", "loss", "wer", "cer"])
        writer.writeheader()
        writer.writerows(rows)

    print("\nValidation WER/CER per epoch:")
    for row in rows:
        print(f"  epoch {row['epoch']:>2}: WER {row['wer']:.2%}  CER {row['cer']:.2%}  loss {row['loss']}")


def main():
    processor = build_processor()
    model = build_model(processor)

    args = TrainingArguments(
        output_dir=str(config.CHECKPOINT_DIR),
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION,
        learning_rate=config.LEARNING_RATE,
        warmup_steps=config.WARMUP_RATIO,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
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

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=load_split(config.TRAIN_CSVS),
        eval_dataset=load_split(config.VALIDATION_CSVS),
        data_collator=DataCollatorCTC(processor),
        processing_class=processor,
        compute_metrics=make_compute_metrics(processor),
        preprocess_logits_for_metrics=argmax_logits,
    )

    last_checkpoint = get_last_checkpoint(str(config.CHECKPOINT_DIR)) if config.CHECKPOINT_DIR.exists() else None
    print("Resuming from:", last_checkpoint or "nothing, starting from scratch")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    save_history(trainer.state.log_history)

    model = trainer.model.merge_and_unload()
    model.save_pretrained(config.FINAL_MODEL_DIR)
    processor.save_pretrained(config.FINAL_MODEL_DIR)
    print("Best model saved to:", config.FINAL_MODEL_DIR)

    evaluate_tests(config.FINAL_MODEL_DIR)


if __name__ == "__main__":
    main()

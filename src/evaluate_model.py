import csv
import json
import sys

from transformers import Trainer, TrainingArguments

import config
from collator import DataCollatorCTC
from dataset import load_split
from metrics import argmax_logits, decode, make_compute_metrics
from model import gpu_supports_bf16, load_trained_model


def prediction_trainer(model, processor):
    args = TrainingArguments(
        output_dir=str(config.RESULTS_DIR),
        per_device_eval_batch_size=config.BATCH_SIZE,
        bf16=gpu_supports_bf16(),
        dataloader_num_workers=config.NUM_WORKERS,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="none",
    )
    return Trainer(
        model=model,
        args=args,
        data_collator=DataCollatorCTC(processor),
        compute_metrics=make_compute_metrics(processor),
        preprocess_logits_for_metrics=argmax_logits,
    )


def write_predictions(path, audio_paths, references, predictions):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_path", "reference", "prediction"])
        writer.writerows(zip(audio_paths, references, predictions))


def print_examples(references, predictions, count=3):
    for reference, prediction in list(zip(references, predictions))[:count]:
        print("  reference: ", reference)
        print("  prediction:", prediction)


def evaluate_test_set(trainer, processor, name, csv_name):
    data = load_split([csv_name], config.EVAL_SUBSET_SIZE)
    output = trainer.predict(data, metric_key_prefix=name)
    predictions, references = decode(processor, output.predictions, output.label_ids)
    write_predictions(config.RESULTS_DIR / f"predictions_{name}.csv", data["audio_path"], references, predictions)

    result = {
        "files": len(data),
        "wer": round(output.metrics[f"{name}_wer"], 4),
        "cer": round(output.metrics[f"{name}_cer"], 4),
    }
    print(f"\n{name} test: WER {result['wer']:.2%}  CER {result['cer']:.2%}  ({len(data)} files)")
    print_examples(references, predictions)
    return result


def evaluate_tests(model_dir=None):
    model, processor = load_trained_model(model_dir)
    trainer = prediction_trainer(model, processor)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {name: evaluate_test_set(trainer, processor, name, csv_name) for name, csv_name in config.TEST_CSVS.items()}

    (config.RESULTS_DIR / "test_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nSaved to:", config.RESULTS_DIR)
    return results


if __name__ == "__main__":
    evaluate_tests(sys.argv[1] if len(sys.argv) > 1 else None)

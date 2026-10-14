import csv
import json
import sys
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments, Wav2Vec2ForCTC, Wav2Vec2Processor

import config
from collator import DataCollatorCTC
from dataset import load_split
from metrics import argmax_logits, decode, make_compute_metrics


def evaluate_tests(model_dir=None):
    model_dir = Path(model_dir or config.FINAL_MODEL_DIR)
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(model_dir)

    args = TrainingArguments(
        output_dir=str(config.RESULTS_DIR),
        per_device_eval_batch_size=config.BATCH_SIZE,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        dataloader_num_workers=config.NUM_WORKERS,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="none",
    )
    trainer = Trainer(model=model, args=args, data_collator=DataCollatorCTC(processor), compute_metrics=make_compute_metrics(processor), preprocess_logits_for_metrics=argmax_logits)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for name, csv_name in config.TEST_CSVS.items():
        data = load_split([csv_name])
        output = trainer.predict(data, metric_key_prefix=name)
        predictions, references = decode(processor, output.predictions, output.label_ids)

        with open(config.RESULTS_DIR / f"predictions_{name}.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["audio_path", "reference", "prediction"])
            writer.writerows(zip(data["audio_path"], references, predictions))

        results[name] = {"files": len(data), "wer": round(output.metrics[f"{name}_wer"], 4), "cer": round(output.metrics[f"{name}_cer"], 4)}
        print(f"\n{name} test: WER {results[name]['wer']:.2%}  CER {results[name]['cer']:.2%}  ({len(data)} files)")

        for reference, prediction in list(zip(references, predictions))[:3]:
            print("  reference: ", reference)
            print("  prediction:", prediction)

    (config.RESULTS_DIR / "test_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nSaved to:", config.RESULTS_DIR)

    return results


if __name__ == "__main__":
    evaluate_tests(sys.argv[1] if len(sys.argv) > 1 else None)

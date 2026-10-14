import csv
import json
import multiprocessing
import sys
from pathlib import Path

import jiwer
import torch
from pyctcdecode import build_ctcdecoder
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

import config
from collator import DataCollatorCTC
from dataset import load_split


def get_logits(model, processor, data):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device).eval()
    loader = DataLoader(data, batch_size=config.BATCH_SIZE, collate_fn=DataCollatorCTC(processor), num_workers=config.NUM_WORKERS)
    all_logits = []

    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        for batch in loader:
            logits = model(batch["input_values"].to(device), attention_mask=batch["attention_mask"].to(device)).logits
            lengths = model._get_feat_extract_output_lengths(batch["attention_mask"].sum(-1))
            all_logits += [row[:length].float().cpu().numpy() for row, length in zip(logits, lengths)]

    return all_logits


def ctc_labels(processor):
    labels = [""] * len(processor.tokenizer)
    for token, index in processor.tokenizer.get_vocab().items():
        labels[index] = {"[PAD]": "", "|": " ", "[UNK]": "⁇"}.get(token, token)
    return labels


def greedy(processor, logits):
    return [processor.tokenizer.decode(row.argmax(-1)) for row in logits]


def beam(decoder, logits):
    with multiprocessing.get_context("fork").Pool(config.DECODE_PROCESSES) as pool:
        return decoder.decode_batch(pool, logits, beam_width=config.BEAM_WIDTH)


def score(references, predictions):
    return {"wer": round(jiwer.wer(references, predictions), 4), "cer": round(jiwer.cer(references, predictions), 4)}


def tune(decoder, logits, references):
    best = None
    for alpha in config.ALPHAS:
        for beta in config.BETAS:
            decoder.reset_params(alpha=alpha, beta=beta)
            wer = jiwer.wer(references, beam(decoder, logits))
            print(f"  alpha {alpha}  beta {beta}:  WER {wer:.2%}")
            if best is None or wer < best[0]:
                best = (wer, alpha, beta)

    decoder.reset_params(alpha=best[1], beta=best[2])
    return best[1], best[2]


def main(model_dir=None):
    model_dir = Path(model_dir or config.FINAL_MODEL_DIR)
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(model_dir)

    labels = ctc_labels(processor)
    beam_decoder = build_ctcdecoder(labels)
    kenlm_decoder = build_ctcdecoder(labels, kenlm_model_path=str(config.LM_PATH))

    sets = {"validation": load_split(config.VALIDATION_CSVS, config.EVAL_FILES)}
    sets.update({f"{name}_test": load_split([csv_name], config.EVAL_FILES) for name, csv_name in config.TEST_CSVS.items()})
    logits = {name: get_logits(model, processor, data) for name, data in sets.items()}
    references = {name: list(data["text"]) for name, data in sets.items()}

    print(f"Tuning KenLM alpha/beta on {min(config.TUNE_FILES, len(sets['validation']))} validation files")
    alpha, beta = tune(kenlm_decoder, logits["validation"][:config.TUNE_FILES], references["validation"][:config.TUNE_FILES])
    print(f"Best: alpha {alpha}  beta {beta}")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for name, data in sets.items():
        predictions = {"greedy": greedy(processor, logits[name]), "beam": beam(beam_decoder, logits[name]), "kenlm": beam(kenlm_decoder, logits[name])}
        results[name] = {decoder: score(references[name], texts) for decoder, texts in predictions.items()}

        with open(config.RESULTS_DIR / f"decoded_{name}.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["audio_path", "reference", *predictions])
            writer.writerows(zip(data["audio_path"], references[name], *predictions.values()))

    summary = {"beam_width": config.BEAM_WIDTH, "lm": str(config.LM_PATH), "alpha": alpha, "beta": beta, "results": results}
    (config.RESULTS_DIR / "decoder_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    for name, scores in results.items():
        for decoder, s in scores.items():
            print(f"{name:<12} {decoder:<7} WER {s['wer']:>7.2%}   CER {s['cer']:>7.2%}")
    print("\nSaved to:", config.RESULTS_DIR)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

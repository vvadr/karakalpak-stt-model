import csv
import json
import multiprocessing
import sys

import torch
from pyctcdecode import build_ctcdecoder
from torch.utils.data import DataLoader

import config
from collator import DataCollatorCTC
from dataset import load_split
from metrics import wer_and_cer
from model import load_trained_model

PYCTCDECODE_LABELS = {config.PAD_TOKEN: "", config.WORD_SEPARATOR: " ", config.UNKNOWN_TOKEN: "⁇"}


def best_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_logits(model, processor, data):
    device = best_device()
    model.to(device).eval()
    loader = DataLoader(data, batch_size=config.BATCH_SIZE, collate_fn=DataCollatorCTC(processor), num_workers=config.NUM_WORKERS)
    all_logits = []

    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        for batch in loader:
            logits = model(batch["input_values"].to(device), attention_mask=batch["attention_mask"].to(device)).logits
            frames = model._get_feat_extract_output_lengths(batch["attention_mask"].sum(-1))
            all_logits += [row[:length].float().cpu().numpy() for row, length in zip(logits, frames)]

    return all_logits


def ctc_labels(processor):
    labels = [""] * len(processor.tokenizer)
    for token, index in processor.tokenizer.get_vocab().items():
        labels[index] = PYCTCDECODE_LABELS.get(token, token)
    return labels


def greedy(processor, logits):
    return [processor.tokenizer.decode(row.argmax(-1)) for row in logits]


def beam(decoder, logits):
    with multiprocessing.get_context("fork").Pool(config.DECODE_PROCESSES) as pool:
        return decoder.decode_batch(pool, logits, beam_width=config.BEAM_WIDTH)


def decode_all_ways(processor, logits, beam_decoder, kenlm_decoder):
    return {"greedy": greedy(processor, logits), "beam": beam(beam_decoder, logits), "kenlm": beam(kenlm_decoder, logits)}


def score(references, predictions):
    return {name: round(value, 4) for name, value in wer_and_cer(references, predictions).items()}


def tune(decoder, logits, references):
    best = None
    for alpha in config.LM_WEIGHT_CANDIDATES:
        for beta in config.WORD_BONUS_CANDIDATES:
            decoder.reset_params(alpha=alpha, beta=beta)
            wer = wer_and_cer(references, beam(decoder, logits))["wer"]
            print(f"  alpha {alpha}  beta {beta}:  WER {wer:.2%}")
            if best is None or wer < best[0]:
                best = (wer, alpha, beta)

    _, alpha, beta = best
    decoder.reset_params(alpha=alpha, beta=beta)
    return alpha, beta


def evaluation_sets():
    sets = {"validation": load_split(config.VALIDATION_CSVS, config.EVAL_SUBSET_SIZE)}
    for name, csv_name in config.TEST_CSVS.items():
        sets[f"{name}_test"] = load_split([csv_name], config.EVAL_SUBSET_SIZE)
    return sets


def write_decoded(path, audio_paths, references, predictions):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_path", "reference", *predictions])
        writer.writerows(zip(audio_paths, references, *predictions.values()))


def print_scores(results):
    print()
    for name, scores in results.items():
        for decoder, s in scores.items():
            print(f"{name:<12} {decoder:<7} WER {s['wer']:>7.2%}   CER {s['cer']:>7.2%}")


def main(model_dir=None):
    model, processor = load_trained_model(model_dir)
    labels = ctc_labels(processor)
    beam_decoder = build_ctcdecoder(labels)
    kenlm_decoder = build_ctcdecoder(labels, kenlm_model_path=str(config.LM_PATH))

    sets = evaluation_sets()
    logits = {name: get_logits(model, processor, data) for name, data in sets.items()}
    references = {name: list(data["text"]) for name, data in sets.items()}

    tuning_files = min(config.TUNING_FILES, len(sets["validation"]))
    print(f"Tuning KenLM alpha/beta on {tuning_files} validation files")
    alpha, beta = tune(kenlm_decoder, logits["validation"][:tuning_files], references["validation"][:tuning_files])
    print(f"Best: alpha {alpha}  beta {beta}")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, data in sets.items():
        predictions = decode_all_ways(processor, logits[name], beam_decoder, kenlm_decoder)
        results[name] = {decoder: score(references[name], texts) for decoder, texts in predictions.items()}
        write_decoded(config.RESULTS_DIR / f"decoded_{name}.csv", data["audio_path"], references[name], predictions)

    summary = {"beam_width": config.BEAM_WIDTH, "lm": str(config.LM_PATH), "alpha": alpha, "beta": beta, "results": results}
    (config.RESULTS_DIR / "decoder_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_scores(results)
    print("\nSaved to:", config.RESULTS_DIR)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

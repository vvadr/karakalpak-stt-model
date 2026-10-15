import json
import sys
from pathlib import Path

from datasets import Dataset
from pyctcdecode import build_ctcdecoder

import config
from dataset import read_split_csv
from decode import ctc_labels, decode_all_ways, get_logits, print_scores, score, write_decoded
from model import load_trained_model


def tuned_alpha_and_beta():
    tuned = json.loads((config.RESULTS_DIR / "decoder_results.json").read_text(encoding="utf-8"))
    return tuned["alpha"], tuned["beta"]


def main(csv_path, name=None):
    csv_path = Path(csv_path).resolve()
    name = name or f"{csv_path.parent.name}_{csv_path.stem}"
    alpha, beta = tuned_alpha_and_beta()

    model, processor = load_trained_model()
    labels = ctc_labels(processor)
    beam_decoder = build_ctcdecoder(labels)
    kenlm_decoder = build_ctcdecoder(labels, kenlm_model_path=str(config.LM_PATH), alpha=alpha, beta=beta)

    data = Dataset.from_pandas(read_split_csv(csv_path), preserve_index=False)
    references = list(data["text"])
    predictions = decode_all_ways(processor, get_logits(model, processor, data), beam_decoder, kenlm_decoder)
    results = {decoder: score(references, texts) for decoder, texts in predictions.items()}

    config.BENCHMARK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_decoded(config.BENCHMARK_RESULTS_DIR / f"{name}.csv", data["audio_path"], references, predictions)
    hours = round(sum(data["duration"]) / 3600, 2)
    summary = {"benchmark": str(csv_path), "files": len(data), "hours": hours, "alpha": alpha, "beta": beta, "results": results}
    (config.BENCHMARK_RESULTS_DIR / f"{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{name}: {len(data)} files, {hours} h (alpha {alpha}, beta {beta} from our validation set)")
    print_scores({name: results})
    print("Saved to:", config.BENCHMARK_RESULTS_DIR)


if __name__ == "__main__":
    main(*sys.argv[1:])

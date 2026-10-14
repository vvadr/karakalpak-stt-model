import argparse
from pathlib import Path

import torch

from scripts.wav2vec_lora.config import Config
from scripts.wav2vec_lora.data import read_manifest, write_json
from scripts.wav2vec_lora.model import choose_device, load_export
from scripts.wav2vec_lora.train import evaluate, loader


def main():
    parser = argparse.ArgumentParser(description="Evaluate an exported wav2vec2 CTC model")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dataset", default="data/karakalpak-combined")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(6)
    args.output.mkdir(parents=True, exist_ok=True)
    device = choose_device()
    model, processor = load_export(args.model, device)
    config = Config(batch_size=args.batch_size, precision=args.precision)
    batches = loader(args.dataset, read_manifest(args.dataset, args.split), processor, config)
    metrics = evaluate(model, batches, processor, device, args.precision, args.output / "predictions.csv")
    write_json(args.output / "metrics.json", metrics)


if __name__ == "__main__":
    main()

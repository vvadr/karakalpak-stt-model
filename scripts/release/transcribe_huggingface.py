"""Transcribe 16 kHz audio with a local or published Karakalpak model bundle."""

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download
import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


MODEL_ID = "iamabdulaziz/wav2vec2-xls-r-300m-karakalpak"
INFERENCE_FILES = [
    "model.safetensors", "config.json", "vocab.json", "tokenizer_config.json",
    "processor_config.json", "preprocessor_config.json", "special_tokens_map.json",
    "added_tokens.json", "decoder_config.json", "lm/kaa_4gram.arpa",
]


def model_folder(identifier, revision):
    local = Path(identifier)
    if local.is_dir():
        return local
    return Path(snapshot_download(identifier, revision=revision, allow_patterns=INFERENCE_FILES))


def load_audio(path):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != 16000:
        raise ValueError(f"Expected 16000 Hz audio, received {rate} Hz. Resample the recording first.")
    audio = audio.mean(axis=1)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Audio must contain finite, non-empty samples.")
    return audio


def choose_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def transcribe(folder, audio, decoder_name, device):
    processor = Wav2Vec2Processor.from_pretrained(folder, local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(folder, local_files_only=True).to(device).eval()
    inputs = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits[0].float().cpu().numpy()
    if decoder_name == "greedy":
        return processor.tokenizer.decode(logits.argmax(-1))

    from pyctcdecode import build_ctcdecoder

    settings = json.loads((folder / "decoder_config.json").read_text(encoding="utf-8"))
    relative_lm = Path(settings["lm_path"])
    if relative_lm.is_absolute() or ".." in relative_lm.parts:
        raise ValueError("The release must use a relative KenLM path without parent traversal.")
    # Hub snapshots legitimately link files to the neighboring blobs cache.
    lm = folder / relative_lm
    if not lm.is_file():
        raise ValueError("The release's KenLM file is missing.")
    tokenizer = processor.tokenizer
    special = {tokenizer.pad_token: "", tokenizer.word_delimiter_token: " ", tokenizer.unk_token: "⁇"}
    labels = [""] * len(tokenizer)
    for token, index in tokenizer.get_vocab().items():
        labels[index] = special.get(token, token)
    decoder = build_ctcdecoder(labels, kenlm_model_path=str(lm), alpha=settings["alpha"], beta=settings["beta"])
    return decoder.decode(logits, beam_width=settings["beam_width"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", default=MODEL_ID, help="Hugging Face model ID or local release folder")
    parser.add_argument("--revision", default="main", help="pin a Hugging Face commit or tag")
    parser.add_argument("--decoder", choices=["kenlm", "greedy"], default="kenlm")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    folder = model_folder(args.model, args.revision)
    audio = load_audio(args.audio)
    device = choose_device(args.device)
    text = transcribe(folder, audio, args.decoder, device)
    if args.json:
        print(json.dumps({"text": text, "decoder": args.decoder, "device": device,
                          "audio_seconds": len(audio) / 16000, "model": args.model,
                          "revision": args.revision}, ensure_ascii=False))
    else:
        print(text)


if __name__ == "__main__":
    main()

import sys

import soundfile as sf
import torch
from pyctcdecode import build_ctcdecoder

import config
from benchmark import tuned_alpha_and_beta
from decode import best_device, ctc_labels
from metrics import wer_and_cer
from model import load_trained_model


def load_audio(path):
    audio, sampling_rate = sf.read(path, dtype="float32", always_2d=True)
    if sampling_rate != config.SAMPLING_RATE:
        raise SystemExit(f"{path} is {sampling_rate} Hz, expected {config.SAMPLING_RATE} Hz")
    return audio.mean(axis=1)


def get_logits(model, processor, audio):
    device = best_device()
    model.to(device).eval()
    inputs = processor.feature_extractor(audio, sampling_rate=config.SAMPLING_RATE, return_tensors="pt")
    with torch.no_grad():
        return model(inputs.input_values.to(device)).logits[0].float().cpu().numpy()


def transcribe_all_ways(processor, logits):
    labels = ctc_labels(processor)
    alpha, beta = tuned_alpha_and_beta()
    kenlm_decoder = build_ctcdecoder(labels, kenlm_model_path=str(config.LM_PATH), alpha=alpha, beta=beta)
    return {
        "greedy": processor.tokenizer.decode(logits.argmax(-1)),
        "beam": build_ctcdecoder(labels).decode(logits, beam_width=config.BEAM_WIDTH),
        "kenlm": kenlm_decoder.decode(logits, beam_width=config.BEAM_WIDTH),
    }


def main(audio_path, reference=None):
    model, processor = load_trained_model()
    audio = load_audio(audio_path)
    texts = transcribe_all_ways(processor, get_logits(model, processor, audio))

    print(f"\n{audio_path} ({len(audio) / config.SAMPLING_RATE:.1f} s)")
    if reference:
        print(f"  {'reference':<9} {reference}")
    for decoder, text in texts.items():
        scores = wer_and_cer([reference], [text]) if reference else None
        suffix = f"   (WER {scores['wer']:.2%}  CER {scores['cer']:.2%})" if scores else ""
        print(f"  {decoder:<9} {text}{suffix}")


if __name__ == "__main__":
    main(*sys.argv[1:])

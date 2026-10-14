import argparse

import librosa
import torch

from scripts.wav2vec_lora.model import autocast, choose_device, load_export


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio with an exported wav2vec2 CTC model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    args = parser.parse_args()
    torch.set_num_threads(6)
    device = choose_device()
    model, processor = load_export(args.model, device)
    audio, _ = librosa.load(args.audio, sr=16000, mono=True)
    inputs = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt",
                                         return_attention_mask=True).to(device)
    with torch.inference_mode(), autocast(device, args.precision):
        ids = model(**inputs).logits.argmax(-1)[0].cpu().tolist()
    print(processor.tokenizer.decode(ids, group_tokens=True, skip_special_tokens=True))


if __name__ == "__main__":
    main()

import torch

from jiwer import cer, wer
from pyctcdecode import build_ctcdecoder
from torch.utils.data import DataLoader

from config import (
    TEST_CSV,
    KENLM_PATH,
    BEAM_WIDTH,
    LM_ALPHA,
    LM_BETA,
)

from data import SpeechDataset, DataCollatorCTC
from model import load_trained_model
from processor import load_processor


def create_decoder(processor):
    vocab = processor.tokenizer.get_vocab()

    labels = [
        ""
        for _ in range(len(processor.tokenizer))
    ]

    for token, index in vocab.items():
        labels[index] = token

    for index, token in enumerate(labels):
        if token == "|":
            labels[index] = " "

        elif token == "[PAD]":
            labels[index] = ""

    decoder = build_ctcdecoder(
        labels=labels,
        kenlm_model_path=str(KENLM_PATH),
        alpha=LM_ALPHA,
        beta=LM_BETA,
    )

    return decoder


def evaluate(
    model,
    processor,
    decoder,
    device,
):
    dataset = SpeechDataset(
        TEST_CSV,
        processor,
    )

    collator = DataCollatorCTC(
        processor
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collator,
    )

    references = []
    predictions = []

    total_loss = 0

    model.eval()

    with torch.inference_mode():

        for step, batch in enumerate(loader):
            input_values = batch[
                "input_values"
            ].to(device)

            labels = batch[
                "labels"
            ].to(device)

            attention_mask = batch.get(
                "attention_mask"
            )

            if attention_mask is not None:
                attention_mask = attention_mask.to(
                    device
                )

            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += outputs.loss.item()

            logits = (
                outputs.logits[0]
                .float()
                .cpu()
                .numpy()
            )

            prediction = decoder.decode(
                logits,
                beam_width=BEAM_WIDTH,
            ).strip()

            labels_for_decode = labels.clone()

            labels_for_decode[
                labels_for_decode == -100
            ] = processor.tokenizer.pad_token_id

            reference = processor.batch_decode(
                labels_for_decode,
                group_tokens=False,
            )[0].strip()

            predictions.append(
                prediction
            )

            references.append(
                reference
            )

            if step % 100 == 0:
                print(
                    f"Test: {step}/{len(loader)}"
                )

    average_loss = (
        total_loss / len(loader)
    )

    word_error_rate = wer(
        references,
        predictions,
    )

    character_error_rate = cer(
        references,
        predictions,
    )

    print()
    print("Test")
    print(
        f"Loss: {average_loss:.4f}"
    )

    print()
    print("Beam Search + KenLM")
    print(
        f"WER: {word_error_rate * 100:.2f}%"
    )
    print(
        f"CER: {character_error_rate * 100:.2f}%"
    )

    print()
    print(
        f"Beam width: {BEAM_WIDTH}"
    )
    print(
        f"LM alpha: {LM_ALPHA}"
    )
    print(
        f"LM beta: {LM_BETA}"
    )

    print()
    print("Examples:")

    for i in range(
        min(10, len(references))
    ):
        print()
        print(
            "Reference:",
            references[i]
        )

        print(
            "Prediction:",
            predictions[i]
        )


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    processor = load_processor()

    model = load_trained_model(
        processor
    ).to(device)

    decoder = create_decoder(
        processor
    )

    evaluate(
        model,
        processor,
        decoder,
        device,
    )


if __name__ == "__main__":
    main()
import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from peft import PeftModel
from jiwer import wer, cer

from dataset import SpeechDataset
from collator import DataCollatorCTC


MODEL_NAME = "facebook/wav2vec2-xls-r-300m"
MODEL_PATH = "outputs/lora"

VALID_CSV = "data/karakalpak-combined/valid.csv"
TEST_CSV = "data/karakalpak-combined/test.csv"


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)

base_model = Wav2Vec2ForCTC.from_pretrained(
    MODEL_NAME,
    vocab_size=len(processor.tokenizer),
    pad_token_id=processor.tokenizer.pad_token_id,
    ctc_loss_reduction="mean"
)

model = PeftModel.from_pretrained(
    base_model,
    MODEL_PATH
)

model = model.to(device)
model.eval()

collator = DataCollatorCTC(processor)


def evaluate(csv_path, name):
    dataset = SpeechDataset(
        csv_path=csv_path,
        processor=processor
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collator
    )

    predictions = []
    references = []
    total_loss = 0

    with torch.inference_mode():
        for step, batch in enumerate(loader):
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            attention_mask = batch.get("attention_mask")

            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()

            predicted_ids = torch.argmax(
                outputs.logits,
                dim=-1
            )

            prediction = processor.batch_decode(
                predicted_ids
            )

            labels = labels.clone()
            labels[labels == -100] = processor.tokenizer.pad_token_id

            reference = processor.batch_decode(
                labels,
                group_tokens=False
            )

            predictions.extend(prediction)
            references.extend(reference)

            if step % 100 == 0:
                print(f"{name}: {step}/{len(loader)}")

    average_loss = total_loss / len(loader)
    word_error_rate = wer(references, predictions)
    character_error_rate = cer(references, predictions)

    print()
    print(f"{name} Loss: {average_loss:.4f}")
    print(f"{name} WER: {word_error_rate * 100:.2f}%")
    print(f"{name} CER: {character_error_rate * 100:.2f}%")

    print()
    print("Examples:")

    for reference, prediction in list(
        zip(references, predictions)
    )[:5]:
        print("Reference:", reference)
        print("Prediction:", prediction)
        print()


# evaluate(VALID_CSV, "Validation")
evaluate(TEST_CSV, "Test")
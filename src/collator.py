import soundfile as sf


class DataCollatorCTC:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, rows):
        audio = [sf.read(row["audio_path"], dtype="float32")[0] for row in rows]
        batch = self.processor.feature_extractor(audio, sampling_rate=16000, padding=True, return_tensors="pt")

        labels = self.processor.tokenizer([row["text"] for row in rows], padding=True, return_tensors="pt")
        batch["labels"] = labels["input_ids"].masked_fill(labels["attention_mask"].ne(1), -100)

        return batch

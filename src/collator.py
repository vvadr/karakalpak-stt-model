import soundfile as sf

import config


class DataCollatorCTC:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, rows):
        batch = self.audio_batch([row["audio_path"] for row in rows])
        batch["labels"] = self.label_batch([row["text"] for row in rows])
        return batch

    def audio_batch(self, paths):
        audio = [sf.read(path, dtype="float32")[0] for path in paths]
        return self.processor.feature_extractor(audio, sampling_rate=config.SAMPLING_RATE, padding=True, return_tensors="pt")

    def label_batch(self, texts):
        labels = self.processor.tokenizer(texts, padding=True, return_tensors="pt")
        is_padding = labels["attention_mask"].ne(1)
        return labels["input_ids"].masked_fill(is_padding, config.IGNORED_LABEL_ID)

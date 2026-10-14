from processor import build_processor
from dataset import build_datasets


class DataCollatorCTC:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(input_features, padding=True, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, padding=True, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)
        batch["labels"] = labels

        return batch


def build_collator():
    processor = build_processor()
    return DataCollatorCTC(processor)


if __name__ == "__main__":
    processor = build_processor()
    train_dataset, _ = build_datasets()
    collator = DataCollatorCTC(processor)

    features = [train_dataset[0], train_dataset[1]]
    batch = collator(features)

    print("Input shape:", batch["input_values"].shape)
    print("Attention mask shape:", batch["attention_mask"].shape)
    print("Labels shape:", batch["labels"].shape)

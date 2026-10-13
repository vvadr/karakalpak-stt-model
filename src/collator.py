class DataCollatorCTC:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):

        input_features = [
            {
                "input_values": feature["input_values"]
            }
            for feature in features
        ]

        label_features = [
            {
                "input_ids": feature["labels"]
            }
            for feature in features
        ]

        batch = self.processor.pad(
            input_features,
            padding=True,
            return_tensors="pt"
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=True,
            return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1),
            -100
        )

        batch["labels"] = labels

        return batch
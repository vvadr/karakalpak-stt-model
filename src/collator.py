from dataclasses import dataclass
from typing import Dict, List, Union

import torch

from processor import processor


@dataclass
class DataCollatorCTC:
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]

        batch = processor.feature_extractor.pad(input_features, padding=self.padding, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]

        labels_batch = processor.tokenizer.pad(label_features, padding=self.padding, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)

        batch["labels"] = labels

        return batch
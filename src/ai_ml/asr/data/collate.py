import torch
from torch.nn.utils.rnn import pad_sequence

from ..features.log_mel import extract_log_mel

PAD_ID = 2

def collate_batch(samples,  tokenizer):
    features = []
    feature_lengths = []

    labels = []
    label_lengths = []

    for sample in samples:
        audio = sample["audio"]
        text = sample["text"]

        log_mel = extract_log_mel(audio)

        token_ids = tokenizer.encode(text).ids

        token_ids = torch.tensor(token_ids, dtype=torch.long)

        features.append(log_mel)
        feature_lengths.append(log_mel.shape[0])

        labels.append(token_ids)
        label_lengths.append(len(token_ids))

    features = pad_sequence(features, batch_first=True, padding_value=0.0)
    labels = pad_sequence(labels, batch_first=True, padding_value=PAD_ID)

    feature_lengths = torch.tensor(feature_lengths, dtype=torch.long)

    label_lengths = torch.tensor(label_lengths, dtype=torch.long)

    return {
        "features": features,
        "feature_lengths": feature_lengths,
        "labels": labels,
        "label_lengths": label_lengths
    }
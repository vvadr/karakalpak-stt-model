import numpy as np
from jiwer import wer, cer

from processor import build_processor


processor = build_processor()


def compute_metrics(pred):
    logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    pred_ids = np.argmax(logits, axis=-1)

    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_texts = processor.batch_decode(pred_ids)
    label_texts = processor.batch_decode(label_ids, group_tokens=False)

    return {"wer": wer(label_texts, pred_texts), "cer": cer(label_texts, pred_texts)}


if __name__ == "__main__":
    references = ["men kitap oqiyman", "bul jaqsi"]
    predictions = ["men kitap oqiman", "bul jaqsi"]

    print("WER:", wer(references, predictions))
    print("CER:", cer(references, predictions))

import jiwer
import numpy as np


def argmax_logits(logits, labels):
    return logits.argmax(dim=-1)


def decode(processor, pred_ids, label_ids):
    pad_id = processor.tokenizer.pad_token_id
    pred_ids = np.where(pred_ids == -100, pad_id, pred_ids)
    label_ids = np.where(label_ids == -100, pad_id, label_ids)

    predictions = processor.tokenizer.batch_decode(pred_ids)
    references = processor.tokenizer.batch_decode(label_ids, group_tokens=False)

    return predictions, references


def make_compute_metrics(processor):
    def compute_metrics(pred):
        predictions, references = decode(processor, pred.predictions, pred.label_ids)
        return {"wer": jiwer.wer(references, predictions), "cer": jiwer.cer(references, predictions)}

    return compute_metrics

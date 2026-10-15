import jiwer
import numpy as np

import config


def argmax_logits(logits, labels):
    return logits.argmax(dim=-1)


def replace_ignored_labels(ids, pad_id):
    return np.where(ids == config.IGNORED_LABEL_ID, pad_id, ids)


def decode(processor, pred_ids, label_ids):
    pad_id = processor.tokenizer.pad_token_id
    predictions = processor.tokenizer.batch_decode(replace_ignored_labels(pred_ids, pad_id))
    references = processor.tokenizer.batch_decode(replace_ignored_labels(label_ids, pad_id), group_tokens=False)
    return predictions, references


def wer_and_cer(references, predictions):
    return {"wer": jiwer.wer(references, predictions), "cer": jiwer.cer(references, predictions)}


def make_compute_metrics(processor):
    def compute_metrics(pred):
        predictions, references = decode(processor, pred.predictions, pred.label_ids)
        return wer_and_cer(references, predictions)

    return compute_metrics

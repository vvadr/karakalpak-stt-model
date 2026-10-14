import contextlib

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(device, precision):
    if precision == "bf16" and device.type in {"mps", "cuda"}:
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def create_model(config, processor, adapter=None):
    model = Wav2Vec2ForCTC.from_pretrained(
        config.base_model, revision=config.base_revision, vocab_size=len(processor.tokenizer),
        pad_token_id=processor.tokenizer.pad_token_id, bos_token_id=None, eos_token_id=None,
        ctc_loss_reduction="mean", ctc_zero_infinity=False,
        layerdrop=0.0, mask_time_prob=0.05, final_dropout=0.1,
        attn_implementation="sdpa",
    )
    model.freeze_feature_encoder()
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
    else:
        # ASR uses input_values; a text task_type wrapper would expect input_ids.
        model = get_peft_model(model, LoraConfig(
            r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"], bias="none", modules_to_save=["lm_head"],
        ))
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if not any("lm_head" in name and p.requires_grad for name, p in model.named_parameters()):
        raise ValueError("The new CTC head must be trainable and saved with LoRA")
    return model


def output_lengths(model, mask):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base._get_feat_extract_output_lengths(mask.sum(-1)).long()


def ctc_loss(model, logits, mask, labels):
    lengths = output_lengths(model, mask).cpu()
    labels = labels.cpu()
    targets = labels.ge(0)
    target_lengths = targets.sum(-1)
    for ids, length in zip(labels, lengths):
        ids = ids[ids.ge(0)]
        minimum = len(ids) + int((ids[1:] == ids[:-1]).sum())
        if minimum > int(length):
            raise ValueError("CTC target plus repeated letters exceeds audio output length")
        if (ids == model.config.pad_token_id).any():
            raise ValueError("CTC blank appears in the target")
    # MPS has no native CTC kernel. Moving logits to CPU preserves autograd to MPS.
    loss_device = torch.device("cpu") if logits.device.type == "mps" else logits.device
    log_probs = F.log_softmax(logits.float().to(loss_device), dim=-1).transpose(0, 1)
    return F.ctc_loss(
        log_probs, labels[targets].to(loss_device), lengths.to(loss_device),
        target_lengths.to(loss_device), blank=model.config.pad_token_id,
        reduction="mean", zero_infinity=False,
    )


def load_export(directory, device):
    processor = Wav2Vec2Processor.from_pretrained(directory)
    model = Wav2Vec2ForCTC.from_pretrained(directory, attn_implementation="sdpa").to(device)
    model.eval()
    return model, processor

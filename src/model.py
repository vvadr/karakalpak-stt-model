from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

import config

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj", "intermediate_dense", "output_dense"]
TRAINED_IN_FULL = ["lm_head"]


def gpu_supports_bf16():
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def load_base_model(processor):
    return Wav2Vec2ForCTC.from_pretrained(
        config.BASE_MODEL,
        vocab_size=len(processor.tokenizer),
        pad_token_id=processor.tokenizer.pad_token_id,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        layerdrop=0.0,
    )


def add_lora(model):
    lora_config = LoraConfig(
        r=config.LORA_RANK,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        modules_to_save=TRAINED_IN_FULL,
    )
    return get_peft_model(model, lora_config)


def build_model(processor):
    model = load_base_model(processor)
    model.freeze_feature_encoder()
    model = add_lora(model)
    model.print_trainable_parameters()
    return model


def load_trained_model(model_dir=None):
    model_dir = Path(model_dir or config.FINAL_MODEL_DIR)
    return Wav2Vec2ForCTC.from_pretrained(model_dir), Wav2Vec2Processor.from_pretrained(model_dir)

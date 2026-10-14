from peft import LoraConfig, get_peft_model
from transformers import Wav2Vec2ForCTC

import config


def build_model(processor):
    model = Wav2Vec2ForCTC.from_pretrained(
        config.BASE_MODEL,
        vocab_size=len(processor.tokenizer),
        pad_token_id=processor.tokenizer.pad_token_id,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        layerdrop=0.0,
    )
    model.freeze_feature_encoder()

    lora_config = LoraConfig(
        r=config.LORA_RANK,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "intermediate_dense", "output_dense"],
        modules_to_save=["lm_head"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model

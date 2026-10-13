from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
)

from transformers import Wav2Vec2ForCTC

from config import (
    MODEL_NAME,
    OUTPUT_DIR,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGETS,
)


def create_base_model(processor):
    return Wav2Vec2ForCTC.from_pretrained(
        MODEL_NAME,
        vocab_size=len(processor.tokenizer),
        pad_token_id=processor.tokenizer.pad_token_id,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
    )


def create_train_model(processor):
    model = create_base_model(processor)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        modules_to_save=["lm_head"],
        bias="none",
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.print_trainable_parameters()

    return model


def load_train_model(processor):
    base_model = create_base_model(processor)

    model = PeftModel.from_pretrained(
        base_model,
        str(OUTPUT_DIR),
        is_trainable=True,
    )

    model.print_trainable_parameters()

    return model


def load_trained_model(processor):
    base_model = create_base_model(processor)

    return PeftModel.from_pretrained(
        base_model,
        str(OUTPUT_DIR),
    )
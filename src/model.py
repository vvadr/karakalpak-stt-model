import os

from transformers import Wav2Vec2ForCTC
from peft import LoraConfig, get_peft_model, PeftModel

from processor import tokenizer


MODEL_NAME = "facebook/wav2vec2-xls-r-300m"
ADAPTER_PATH = "outputs/lora"

model = Wav2Vec2ForCTC.from_pretrained(
    MODEL_NAME,
    vocab_size=len(tokenizer),
    pad_token_id=tokenizer.pad_token_id,
    ctc_loss_reduction="mean",
)

if os.path.exists(ADAPTER_PATH):
    model = PeftModel.from_pretrained(
        model,
        ADAPTER_PATH,
        is_trainable=True,
    )

    print("Loaded previous LoRA model")

else:
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        modules_to_save=["lm_head"],
    )

    model = get_peft_model(model, lora_config)

    print("Created new LoRA model")
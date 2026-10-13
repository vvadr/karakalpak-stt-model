import os

from transformers import Wav2Vec2ForCTC
from peft import LoraConfig, get_peft_model, PeftModel

from processor import processor


MODEL_NAME = "facebook/wav2vec2-xls-r-300m"
ADAPTER_DIR = "outputs/lora"

tokenizer = processor.tokenizer


def create_base_model():
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME, vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id, ctc_loss_reduction="mean", ctc_zero_infinity=True, ignore_mismatched_sizes=True)
    model.freeze_feature_encoder()
    return model


base_model = create_base_model()

adapter_config_path = os.path.join(ADAPTER_DIR, "adapter_config.json")

if os.path.exists(adapter_config_path):
    print(f"Loading existing LoRA adapter from: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR, is_trainable=True)

else:
    print("Creating new LoRA adapter")

    lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "v_proj"], modules_to_save=["lm_head"], bias="none")
    model = get_peft_model(base_model, lora_config)


print("\nTrainable parameters:")
model.print_trainable_parameters()

print("\nlm_head status:")

for name, parameter in model.named_parameters():
    if "lm_head" in name:
        print(name, "trainable =", parameter.requires_grad)

print("\nCTC configuration:")
print("Tokenizer PAD ID:", tokenizer.pad_token_id)
print("Model PAD ID:", base_model.config.pad_token_id)
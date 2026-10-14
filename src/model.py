from transformers import Wav2Vec2ForCTC
from peft import LoraConfig, get_peft_model

from processor import build_processor


MODEL_NAME = "facebook/wav2vec2-xls-r-300m"


def build_model():
    processor = build_processor()

    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME, vocab_size=len(processor.tokenizer), pad_token_id=processor.tokenizer.pad_token_id, ctc_loss_reduction="mean", ctc_zero_infinity=True, ignore_mismatched_sizes=True)

    model.freeze_feature_encoder()

    lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, bias="none", target_modules=["q_proj", "v_proj"], modules_to_save=["lm_head"])

    model = get_peft_model(model, lora_config)

    return model


if __name__ == "__main__":
    model = build_model()

    model.print_trainable_parameters()

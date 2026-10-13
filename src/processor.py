from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

from config import (
    VOCAB_PATH,
    SAMPLE_RATE,
    OUTPUT_DIR,
)


def create_processor():
    tokenizer = Wav2Vec2CTCTokenizer(
        str(VOCAB_PATH),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
        bos_token=None,
        eos_token=None,
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=SAMPLE_RATE,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )

    return Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )


def load_processor():
    return Wav2Vec2Processor.from_pretrained(
        str(OUTPUT_DIR)
    )
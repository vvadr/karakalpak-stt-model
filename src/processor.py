from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

import config


def build_tokenizer():
    return Wav2Vec2CTCTokenizer(
        str(config.VOCAB_PATH),
        unk_token=config.UNKNOWN_TOKEN,
        pad_token=config.PAD_TOKEN,
        word_delimiter_token=config.WORD_SEPARATOR,
        bos_token=None,
        eos_token=None,
    )


def build_feature_extractor():
    return Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=config.SAMPLING_RATE,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )


def build_processor():
    return Wav2Vec2Processor(feature_extractor=build_feature_extractor(), tokenizer=build_tokenizer())


def show_example(text="qaraqalpaq tili"):
    processor = build_processor()
    ids = processor.tokenizer(text).input_ids
    print("Vocabulary size:", len(processor.tokenizer))
    print("PAD (CTC blank) id:", processor.tokenizer.pad_token_id)
    print("Text:", text)
    print("Token ids:", ids)
    print("Decoded:", processor.tokenizer.decode(ids, group_tokens=False))


if __name__ == "__main__":
    show_example()

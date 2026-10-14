from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

import config


def build_processor():
    tokenizer = Wav2Vec2CTCTokenizer(str(config.VOCAB_PATH), unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|", bos_token=None, eos_token=None)
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


if __name__ == "__main__":
    processor = build_processor()
    text = "qaraqalpaq tili"
    ids = processor.tokenizer(text).input_ids

    print("Vocabulary size:", len(processor.tokenizer))
    print("PAD (CTC blank) id:", processor.tokenizer.pad_token_id)
    print("Text:", text)
    print("Token ids:", ids)
    print("Decoded:", processor.tokenizer.decode(ids, group_tokens=False))

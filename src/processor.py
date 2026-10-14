from pathlib import Path

from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor


ROOT = Path(__file__).resolve().parent.parent
VOCAB_PATH = ROOT / "vocab.json"


def build_processor():
    tokenizer = Wav2Vec2CTCTokenizer(str(VOCAB_PATH), unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    return processor


if __name__ == "__main__":
    processor = build_processor()

    print("Vocabulary size:", len(processor.tokenizer))
    print("PAD token:", processor.tokenizer.pad_token)
    print("PAD token ID:", processor.tokenizer.pad_token_id)
    print("UNK token:", processor.tokenizer.unk_token)
    print("Word delimiter:", processor.tokenizer.word_delimiter_token)

    text = "salam"
    encoded = processor.tokenizer(text)

    print("Text:", text)
    print("Token IDs:", encoded.input_ids)
    print("Decoded:", processor.tokenizer.decode(encoded.input_ids))

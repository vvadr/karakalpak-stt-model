from dataclasses import dataclass


@dataclass
class Config:
    base_model: str = "facebook/wav2vec2-xls-r-300m"
    base_revision: str = "1a640f32ac3e39899438a2931f9924c02f080a54"
    dataset: str = "data/karakalpak-combined"
    epochs: int = 8
    batch_size: int = 4
    accumulation: int = 4
    learning_rate: float = 3e-4
    head_learning_rate: float = 1e-3
    warmup_fraction: float = 0.1
    validation_fraction: float = 0.1
    lora_rank: int = 16
    lora_alpha: int = 32
    seed: int = 42
    precision: str = "bf16"
    gradient_checkpointing: bool = True
    patience: int = 3
    minimum_epochs: int = 5
    workers: int = 0

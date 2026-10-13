import math
import random

import numpy as np
import torch

from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from config import (
    TRAIN_CSV,
    OUTPUT_DIR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    GRAD_ACCUMULATION,
    WARMUP_RATIO,
    SEED,
    LOG_EVERY,
)

from data import SpeechDataset, DataCollatorCTC
from model import (
    create_train_model,
    load_train_model,
)
from processor import create_processor


STATE_PATH = OUTPUT_DIR / "training_state.pt"
ADAPTER_PATH = OUTPUT_DIR / "adapter_config.json"


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def main():
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    processor = create_processor()

    train_dataset = SpeechDataset(TRAIN_CSV, processor,)

    collator = DataCollatorCTC(processor)

    shuffle_generator = torch.Generator()

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collator, generator=shuffle_generator,)

    resume = (STATE_PATH.exists() and ADAPTER_PATH.exists())

    if resume:
        print("Loading saved training checkpoint...")

        model = load_train_model(processor).to(device)
    else:
        print("Starting new training...")

        model = create_train_model(processor).to(device)

    optimizer = AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=LEARNING_RATE,)

    updates_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUMULATION)

    total_training_steps = (updates_per_epoch * EPOCHS)

    warmup_steps = int(total_training_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps,)

    start_epoch = 0

    if resume:
        checkpoint = torch.load(
            STATE_PATH,
            map_location=device,
            weights_only=False,
        )

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        scheduler.load_state_dict(
            checkpoint["scheduler"]
        )

        start_epoch = checkpoint["epoch"]

        print(
            f"Resuming from epoch "
            f"{start_epoch + 1}"
        )

    if start_epoch >= EPOCHS:
        print(
            f"Training already completed "
            f"{start_epoch}/{EPOCHS} epochs."
        )
        return

    for epoch in range(
        start_epoch,
        EPOCHS
    ):
        shuffle_generator.manual_seed(
            SEED + epoch
        )

        model.train()

        total_loss = 0

        optimizer.zero_grad(
            set_to_none=True
        )

        for step, batch in enumerate(train_loader):
            input_values = batch[
                "input_values"
            ].to(device)

            labels = batch[
                "labels"
            ].to(device)

            attention_mask = batch.get(
                "attention_mask"
            )

            if attention_mask is not None:
                attention_mask = (
                    attention_mask.to(device)
                )

            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss

            group_start = (
                step // GRAD_ACCUMULATION
            ) * GRAD_ACCUMULATION

            group_size = min(
                GRAD_ACCUMULATION,
                len(train_loader) - group_start,
            )

            scaled_loss = (
                loss / group_size
            )

            scaled_loss.backward()

            total_loss += loss.item()

            should_update = (
                (step + 1)
                % GRAD_ACCUMULATION
                == 0
                or
                step + 1
                == len(train_loader)
            )

            if should_update:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()
                scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

            if step % LOG_EVERY == 0:
                current_lr = (
                    scheduler.get_last_lr()[0]
                )

                print(
                    f"Epoch {epoch + 1}/{EPOCHS} "
                    f"| Step {step}/{len(train_loader)} "
                    f"| Loss {loss.item():.4f} "
                    f"| LR {current_lr:.8f}"
                )

        average_loss = (
            total_loss
            / len(train_loader)
        )

        print()
        print(
            f"Epoch {epoch + 1} finished "
            f"| Average loss: "
            f"{average_loss:.4f}"
        )

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        model.save_pretrained(
            str(OUTPUT_DIR),
            save_embedding_layers=False,
        )

        processor.save_pretrained(
            str(OUTPUT_DIR)
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            STATE_PATH,
        )

        print(
            f"Checkpoint saved: "
            f"epoch {epoch + 1}"
        )
        print()

    print("Training finished.")
    print(
        "Saved model:",
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
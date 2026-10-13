import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from processor import processor
from dataset import SpeechDataset
from collator import DataCollatorCTC
from model import model


TRAIN_CSV = "data/karakalpak-combined/train.csv"

train_dataset = SpeechDataset(
    csv_path=TRAIN_CSV,
    processor=processor
)

data_collator = DataCollatorCTC(
    processor=processor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=data_collator
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)
model.train()

optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

epochs = 5

for epoch in range(epochs):
    total_loss = 0

    for step, batch in enumerate(train_loader):
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].to(device)

        attention_mask = batch.get("attention_mask")

        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        optimizer.zero_grad()

        outputs = model(input_values=input_values, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if step % 10 == 0:
            print(f"Epoch {epoch + 1} | "f"Step {step} | "f"Loss {loss.item():.4f}")

    average_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch + 1} finished | "f"Average loss: {average_loss:.4f}")

model.save_pretrained("outputs/lora", save_embedding_layers=False)

processor.save_pretrained("outputs/lora")
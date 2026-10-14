import argparse
import gc
import importlib.metadata
import json
import math
import random
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor, get_cosine_schedule_with_warmup

from .config import Config
from .data import (Collator, DurationBatches, SpeechDataset, audit_dataset,
                   create_processor, read_manifest, split_training, write_json)
from .model import autocast, choose_device, create_model, ctc_loss, load_export, output_lengths


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader(root, frame, processor, config, shuffle=False):
    sampler = DurationBatches(frame, config.batch_size, config.seed, shuffle)
    return DataLoader(
        SpeechDataset(root, frame, processor), batch_sampler=sampler,
        collate_fn=Collator(processor), num_workers=config.workers,
    )


def forward(model, batch, device, precision):
    values = batch["input_values"].to(device)
    mask = batch["attention_mask"].to(device)
    with autocast(device, precision):
        logits = model(input_values=values, attention_mask=mask).logits
    loss = ctc_loss(model, logits, mask, batch["labels"])
    return logits, loss


def evaluate(model, batches, processor, device, precision, output_csv=None):
    model.eval()
    rows, total_loss, count, blanks, frames = [], 0.0, 0, 0, 0
    started = time.monotonic()
    with torch.inference_mode():
        for index, batch in enumerate(batches):
            logits, loss = forward(model, batch, device, precision)
            ids = logits.argmax(-1).cpu()
            lengths = output_lengths(model, batch["attention_mask"]).cpu().tolist()
            for path, reference, prediction, length in zip(batch["paths"], batch["texts"], ids, lengths):
                prediction = prediction[:length]
                hypothesis = processor.tokenizer.decode(prediction.tolist(), group_tokens=True,
                                                        skip_special_tokens=True)
                rows.append({"audio_path": path, "reference": reference, "prediction": hypothesis})
                blanks += int((prediction == processor.tokenizer.pad_token_id).sum())
                frames += length
            total_loss += float(loss) * len(batch["texts"])
            count += len(batch["texts"])
            if (index + 1) % 50 == 0:
                print(f"evaluation {count}/{len(batches.dataset)} recordings", flush=True)
    references, predictions = [r["reference"] for r in rows], [r["prediction"] for r in rows]
    metrics = {
        "recordings": count, "loss": total_loss / count,
        "wer": jiwer.wer(references, predictions), "cer": jiwer.cer(references, predictions),
        "empty_predictions": sum(not x for x in predictions),
        "blank_frame_fraction": blanks / frames, "seconds": time.monotonic() - started,
    }
    if output_csv:
        pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(json.dumps(metrics), flush=True)
    return metrics


def optimizer_for(model, config):
    head, adapters = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (head if "lm_head" in name else adapters).append(parameter)
    return torch.optim.AdamW([
        {"params": adapters, "lr": config.learning_rate},
        {"params": head, "lr": config.head_learning_rate},
    ], weight_decay=0.01)


def preflight(model, batch, device, config, directory):
    model.train()
    optimizer = optimizer_for(model, config)
    tracked = {name: p.detach().cpu().clone() for name, p in model.named_parameters()
               if p.requires_grad and ("lora_B" in name or "lm_head" in name)}
    started = time.monotonic()
    losses, gradients = [], {}
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        logits, loss = forward(model, batch, device, config.precision)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite CTC loss in preflight")
        loss.backward()
        for kind in ("lora_", "lm_head"):
            squared = sum(float(p.grad.detach().float().square().sum())
                          for name, p in model.named_parameters() if kind in name and p.grad is not None)
            gradients[kind] = math.sqrt(squared)
            if not math.isfinite(gradients[kind]) or gradients[kind] == 0:
                raise ValueError(f"Missing or invalid {kind} gradients")
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        print(f"preflight update {step + 1}: CTC loss={losses[-1]:.4f}", flush=True)
    changed = {kind: any(not torch.equal(tracked[name], p.detach().cpu())
                        for name, p in model.named_parameters() if name in tracked and kind in name)
               for kind in ("lora_B", "lm_head")}
    if not all(changed.values()):
        raise ValueError("LoRA or CTC head did not update")
    report = {"device": str(device), "precision": config.precision, "losses": losses,
              "gradient_norms": gradients, "weights_changed": changed,
              "seconds": time.monotonic() - started, "input_shape": list(batch["input_values"].shape),
              "label_shape": list(batch["labels"].shape), "logit_shape": list(logits.shape),
              "pad_and_blank_id": model.config.pad_token_id}
    write_json(directory / "preflight.json", report)
    print(json.dumps(report), flush=True)


def save_adapter(model, processor, directory):
    directory.mkdir(parents=True, exist_ok=True)
    # Wav2vec2 has no text embedding matrix. The CTC head is explicitly saved.
    model.save_pretrained(directory, save_embedding_layers=False)
    model.config.save_pretrained(directory)
    processor.save_pretrained(directory)


def main():
    defaults = Config()
    parser = argparse.ArgumentParser(description="Train wav2vec2 XLS-R 300M with character CTC and PEFT LoRA")
    parser.add_argument("--dataset", default=defaults.dataset)
    parser.add_argument("--output", default=None)
    parser.add_argument("--base-model", default=defaults.base_model)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--accumulation", type=int, default=defaults.accumulation)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--head-learning-rate", type=float, default=defaults.head_learning_rate)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default=defaults.precision)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", type=Path, help="An epoch checkpoint directory")
    args = parser.parse_args()
    config = Config(dataset=str(Path(args.dataset).resolve()), base_model=args.base_model,
                    epochs=args.epochs, batch_size=args.batch_size, accumulation=args.accumulation,
                    learning_rate=args.learning_rate, head_learning_rate=args.head_learning_rate,
                    precision=args.precision, gradient_checkpointing=not args.no_gradient_checkpointing)
    if min(config.epochs, config.batch_size, config.accumulation) < 1:
        parser.error("epochs, batch-size and accumulation must be positive")
    directory = Path(args.output or f"checkpoints/wav2vec2-xlsr300m-lora-kaa-{datetime.now():%Y%m%d-%H%M%S}").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if not args.resume and (directory / "settings.json").exists():
        parser.error("Output already contains a run; choose a new directory or --resume")
    if args.resume:
        saved_config = Config(**json.loads((directory / "settings.json").read_text()))
        if config != saved_config:
            parser.error("Resume settings must match the original run")
    seed_everything(config.seed)
    torch.set_num_threads(6)
    device = choose_device()
    print(f"device={device}; output={directory}", flush=True)
    train_pool, test = read_manifest(config.dataset, "train"), read_manifest(config.dataset, "test")
    audit = audit_dataset(config.dataset, train_pool, test)
    write_json(directory / "data_audit.json", audit)
    training, validation = split_training(train_pool, config.validation_fraction, config.seed)
    if args.resume:
        for name, frame in (("train", training), ("validation", validation), ("test", test)):
            saved = pd.read_csv(directory / f"{name}_split.csv", keep_default_na=False)
            if saved.audio_path.tolist() != frame.audio_path.tolist() or saved.text.tolist() != frame.text.tolist():
                raise ValueError("Dataset differs from the saved checkpoint split")
        processor = Wav2Vec2Processor.from_pretrained(directory / "processor")
    else:
        for name, frame in (("train", training), ("validation", validation), ("test", test)):
            frame.to_csv(directory / f"{name}_split.csv", index=False)
        processor = create_processor(train_pool.text, directory / "processor")
    write_json(directory / "settings.json", asdict(config))
    write_json(directory / "environment.json", {
        "device": str(device), "versions": {name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "peft", "soundfile", "jiwer")},
    })
    print(f"train={len(training)} validation={len(validation)} test={len(test)} vocab={len(processor.tokenizer)}", flush=True)
    model = create_model(config, processor, args.resume).to(device)
    counts = {"trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "total": sum(p.numel() for p in model.parameters())}
    write_json(directory / "parameters.json", counts)
    print(json.dumps(counts), flush=True)
    if args.preflight:
        small = training.sort_values("duration").head(min(2, config.batch_size))
        preflight(model, next(iter(loader(config.dataset, small, processor, config))), device, config, directory)
        return

    train_batches = loader(config.dataset, training, processor, config, shuffle=True)
    validation_batches = loader(config.dataset, validation, processor, config)
    optimizer = optimizer_for(model, config)
    steps_per_epoch = math.ceil(len(train_batches) / config.accumulation)
    total_steps = config.epochs * steps_per_epoch
    scheduler = get_cosine_schedule_with_warmup(optimizer, round(total_steps * config.warmup_fraction), total_steps)
    history, global_step, start_epoch, best_wer, best_cer, stale = [], 0, 0, math.inf, math.inf, 0
    if args.resume:
        state = torch.load(args.resume / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, global_step, start_epoch = state["history"], state["global_step"], state["epoch"]
        best_wer, best_cer, stale = state["best_wer"], state["best_cer"], state["stale"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "mps":
            torch.mps.set_rng_state(state["device_rng"])
        elif device.type == "cuda":
            torch.cuda.set_rng_state_all(state["device_rng"])
    else:
        baseline = evaluate(model, validation_batches, processor, device, config.precision,
                            directory / "baseline_validation_predictions.csv")
        write_json(directory / "baseline_validation_metrics.json", baseline)
    started = time.monotonic()
    trainable = [p for p in model.parameters() if p.requires_grad]
    best_directory = directory / "best_adapter"
    for epoch in range(start_epoch, config.epochs):
        model.train()
        train_batches.batch_sampler.epoch = epoch
        optimizer.zero_grad(set_to_none=True)
        running_loss, seen, window_loss, window_count = 0.0, 0, 0.0, 0
        epoch_started = time.monotonic()
        for index, batch in enumerate(train_batches):
            logits, loss = forward(model, batch, device, config.precision)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss in epoch {epoch + 1}, batch {index}")
            # Scale partial accumulation windows correctly at the end of an epoch.
            window_start = (index // config.accumulation) * config.accumulation
            window_size = min(config.accumulation, len(train_batches) - window_start)
            (loss / window_size).backward()
            value, batch_count = float(loss.detach()), len(batch["texts"])
            running_loss += value * batch_count
            seen += batch_count
            window_loss += value * batch_count
            window_count += batch_count
            del logits, loss
            if (index + 1) % config.accumulation == 0 or index + 1 == len(train_batches):
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                if not torch.isfinite(grad_norm):
                    raise ValueError("Nonfinite gradient norm")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress = {"status": "training", "epoch": epoch + 1, "epochs": config.epochs,
                            "step": global_step, "total_steps": total_steps, "recordings_seen_in_epoch": seen,
                            "training_loss": running_loss / seen, "window_loss": window_loss / window_count,
                            "learning_rates": scheduler.get_last_lr(), "elapsed_seconds": time.monotonic() - started}
                write_json(directory / "progress.json", progress)
                with (directory / "steps.jsonl").open("a") as handle:
                    handle.write(json.dumps(progress) + "\n")
                if global_step % 10 == 0 or index + 1 == len(train_batches):
                    print(f"epoch {epoch + 1}/{config.epochs} step {global_step}/{total_steps} "
                          f"loss {running_loss / seen:.4f} recent {window_loss / window_count:.4f} "
                          f"elapsed {time.monotonic() - epoch_started:.0f}s", flush=True)
                window_loss, window_count = 0.0, 0
        metrics = evaluate(model, validation_batches, processor, device, config.precision,
                           directory / f"validation_epoch_{epoch + 1:02d}.csv")
        record = {"epoch": epoch + 1, "step": global_step, "training_loss": running_loss / seen,
                  "validation": metrics, "epoch_seconds": time.monotonic() - epoch_started}
        history.append(record)
        write_json(directory / "history.json", history)
        if (metrics["wer"], metrics["cer"]) < (best_wer, best_cer):
            best_wer, best_cer, stale = metrics["wer"], metrics["cer"], 0
            save_adapter(model, processor, best_directory)
            write_json(best_directory / "validation_metrics.json", record)
        else:
            stale += 1
        checkpoint = directory / "training_checkpoints" / f"epoch-{epoch + 1:02d}"
        save_adapter(model, processor, checkpoint)
        torch.save({
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1, "global_step": global_step, "history": history,
            "best_wer": best_wer, "best_cer": best_cer, "stale": stale,
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
            "device_rng": torch.mps.get_rng_state() if device.type == "mps" else
                          (torch.cuda.get_rng_state_all() if device.type == "cuda" else None),
        }, checkpoint / "training_state.pt")
        if epoch + 1 >= config.minimum_epochs and stale >= config.patience:
            print("Validation early stopping", flush=True)
            break

    # Export the validation-selected adapter as a self-contained standard CTC model.
    model.cpu()
    del model, optimizer, scheduler
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    model = create_model(config, processor, best_directory)
    model.gradient_checkpointing_disable()
    merged = model.merge_and_unload()
    export_directory = directory / "model"
    merged.save_pretrained(export_directory)
    processor.save_pretrained(export_directory)
    del merged, model
    gc.collect()
    # Evaluate the model reloaded from disk, proving the delivered export works.
    model, processor = load_export(export_directory, device)
    test_metrics = evaluate(model, loader(config.dataset, test, processor, config), processor,
                            device, config.precision, directory / "test_predictions.csv")
    write_json(directory / "test_metrics.json", test_metrics)
    final = {"status": "complete", "epochs_completed": len(history), "steps": global_step,
             "best_validation_wer": best_wer, "best_validation_cer": best_cer,
             "test": test_metrics, "seconds": time.monotonic() - started,
             "model": str(export_directory), "best_adapter": str(best_directory),
             "training_checkpoints": str(directory / "training_checkpoints")}
    write_json(directory / "progress.json", final)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()

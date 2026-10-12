# Intern Learning Path — 3 Weeks (ASR)

2026-09-21 · @Someone

## How this works

You learn on your own; your mentor sets tasks, answers questions, and checks artifacts. Each day ends with a concrete artifact, not "I read about X."

- **Monday:** mentor hands out the week's tasks; this doc is the task sheet.
- **Daily async update (3 lines):** done / blocked / next. Post by end of day.
- **Check-ins:** Tuesday and Thursday, 15 min. Show your artifact and explain it.
- **Friday demo:** 10 min per intern, then feedback.
- **Question rule:** stuck 15 min → search or ask a peer. Stuck 1 hour → ask the mentor, with a hypothesis ("I think X fails because Y").
- **Timebox:** max 1 day per topic. Past that, raise it at the next check-in or ask.
- **Evidence rule:** every number in a writeup points to a log, config, or line of code.
- **Running thread:** the clips you annotate in week 1 become your eval set in week 3.

## Overview

|  | Week 1: Data | Week 2: Models | Week 3: Training & Eval |
| --- | --- | --- | --- |
| Focus | Audio + text features, annotation | Architecture, hands-on in code | Fine-tuning, eval, error analysis |
| Deliverables | Feature notebook, normalizer, 50+ annotated clips, baseline WER on own labels | Model walkthrough notebook, 1 paper summary, layer probe or freeze experiment | Fine-tuned checkpoint, eval report, error analysis tied to data |
| Done when | You can explain every step from raw audio to model input | You can trace one sample through the model and explain each tensor shape | WER compared vs. baseline; top error causes backed by examples |
| Annotation | 2–3 half-days | \~1 h/day | \~1 h/day (optional) |
| Check-ins | Tue, Thu 15 min + Fri demo | Tue, Thu 15 min + Fri demo | Tue, Thu 15 min + Fri demo |

## Week 1: Data

Goal: understand the data before touching models. By Friday you have labeled your own clips and measured a pretrained model on them.

| Day | Task | Artifact |
| --- | --- | --- |
| Mon | Setup: repo, cluster, data layout. Audio basics: sampling rate, resampling, formats | Script that loads, resamples, and prints dataset stats (durations, sample rates, total hours) |
| Tue | Audio features: STFT, mel, fbank, VAD, SNR | Notebook: spectrograms of a clean, a noisy, and a silent clip |
| Wed | Read the annotation guideline, then half-day annotation | 25 clips labeled; 20-clip overlap set shared with a peer |
| Thu | Text: normalization, Latin/Cyrillic, numbers, abbreviations, chars vs. BPE | Normalizer function + notes on where you and your peer disagreed |
| Fri AM | Half-day annotation | 50+ clips total |
| Fri PM | Run a pretrained model on your annotated clips | WER/CER on your own labels → demo |

**Resources**

- [Hugging Face Audio Course](https://huggingface.co/learn/audio-course), units 1–2
- [torchaudio feature extraction tutorial](https://pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html)
- [librosa docs](https://librosa.org/doc/latest/index.html)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [NeMo text processing](https://github.com/NVIDIA/NeMo-text-processing) (how normalization is structured)
- [jiwer](https://github.com/jitsi/jiwer) for WER/CER
- Annotation guideline: provided by mentor

## Week 2: Models

Goal: understand the architecture by inspecting a real checkpoint in code, not by reading alone.

| Day | Task | Artifact |
| --- | --- | --- |
| Mon | CTC: loss, alignment, blank token, greedy vs. beam decoding | Notebook: greedy and beam decode on real model logits |
| Tue | Feature encoder + conformer/transformer blocks | Printed module tree, parameter counts, tensor shapes per layer |
| Wed | Read the wav2vec 2.0 and w2v-BERT papers | 1-page summary: problem each solves, key idea, what changed between them |
| Thu | Forward hooks, layer outputs, freezing strategies | Small layer-wise probe or freeze ablation with results |
| Fri | Explain the architecture back to the mentor | Demo: trace one sample from waveform to text |

Annotation continues \~1 h/day.

**Resources**

- [Sequence Modeling with CTC](https://distill.pub/2017/ctc/) (Distill)
- [wav2vec 2.0 paper](https://arxiv.org/abs/2006.11477)
- [w2v-BERT paper](https://arxiv.org/abs/2108.06209)
- [Seamless paper](https://arxiv.org/abs/2312.05187) (w2v-BERT 2.0 encoder)
- [Conformer paper](https://arxiv.org/abs/2005.08100)
- [Hugging Face Audio Course](https://huggingface.co/learn/audio-course), ASR unit
- [PyTorch forward hooks](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook)

## Week 3: Training & Eval

Goal: fine-tune, evaluate on your own week-1 labels, and link every major error back to the data.

| Day | Task | Artifact |
| --- | --- | --- |
| Mon | Training config, data loading, experiment tracking | Launched run with logged loss and WER |
| Tue | LoRA vs. full fine-tuning on a data subset | 2 runs with configs and results compared |
| Wed | Eval on your week-1 annotated set, per-clip WER | Eval report vs. the week-1 baseline |
| Thu | Error analysis: substitutions, normalization, audio quality, dialect | Top 5 error causes, each with 3+ examples |
| Fri | Final demo: data → model → results → what you'd fix next | Demo + 1-page written summary |

**Resources**

- [Fine-tune w2v-BERT 2.0 for ASR](https://huggingface.co/blog/fine-tune-w2v2-bert) (HF blog)
- [Fine-tune wav2vec 2.0 with CTC](https://huggingface.co/blog/fine-tune-wav2vec2-english) (HF blog)
- [LoRA paper](https://arxiv.org/abs/2106.09685)
- [PEFT docs](https://huggingface.co/docs/peft)
- Lab training configs and eval scripts: provided by mentor

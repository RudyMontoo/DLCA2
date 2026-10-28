# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A from-scratch PyTorch implementation of DCGAN (Deep Convolutional GAN). This project (a CA-2 submission, see `NOTES.md`) trains it on **MNIST only** — the `mnist_colored` and `celeb` configs/dataset classes still exist in the repo but are out of scope here. No test suite, linter, or CI is configured.

For the full project write-up (5W, dataset source, line-by-line code explanation, and actual training results from this repo's run) see `NOTES.md`.

## Commands

Setup:
```
pip install -r requirements.txt
```

Train (also runs periodic inference to save sample grids; MNIST auto-downloads on first run, no manual data prep needed):
```
python -m tools.train --config config/mnist.yaml
```

Sanity-check model shapes directly:
```
python -m model.dcgan
```
(Note: the `__main__` block in `model/dcgan.py` calls `Generator()`/`Discriminator()` with no args — it will raise unless args are filled in; use it as a shape-check template, not as a runnable script as-is.)

## Data layout expectations

- **MNIST** (`dataset/mnist_dataset.py`): uses `torchvision.datasets.MNIST(root=im_path, download=True)` — auto-downloads the official dataset into `dataset_params.im_path` (`data/`) on first run. Labels are discarded (unconditional generation). This replaces the original repo's custom PNG-folder loader, which required a separate manual data-prep step despite some write-ups claiming otherwise.
- **CelebA** (`dataset/celeb_dataset.py`, not used in this project): expects `img_align_celeba/*.jpg` directly under the repo root (recursive glob), resized so the short side matches `im_size` then center-cropped square.
- All images are normalized to `[-1, 1]` to match the generator's `Tanh` output range.

## Architecture

**Config-driven, no hardcoded model shapes.** `model/dcgan.py` defines `Generator` and `Discriminator` as parallel `nn.ModuleList`s built entirely from per-layer lists passed in from the YAML config (`conv_channels`, `kernels`, `strides`, `paddings`, `output_paddings`). This means the number of conv layers, channel widths, and spatial transitions are all determined by config, not code — to change network depth/width, edit the config lists (keeping all per-layer lists the same length), not the model file.

- `Generator`: `ConvTranspose2d` stack, `latent_dim → im_channels`. Every layer has `BatchNorm2d` + `ReLU` except the last, which has no norm and `Tanh` instead (per DCGAN paper).
- `Discriminator`: mirrored `Conv2d` stack, `im_channels → 1`. First layer has bias and no BatchNorm (raw input), middle layers have BatchNorm + `LeakyReLU`, last layer has no activation (raw logit — paired with `BCEWithLogitsLoss`, so no `Sigmoid` in the model itself).
- The three configs (`config/mnist.yaml`, `config/mnist_colored.yaml`, `config/celeb.yaml`) each define a different image size / channel count / layer count by varying these lists in tandem with `dataset_params.im_size` — when adding a new dataset config, the generator/discriminator layer-list lengths and strides must be worked out to land exactly on `im_size` (e.g. celeb goes 1x1 → 64x64 over 5 layers; mnist goes 1x1 → 28x28 over 4 layers with mixed output_paddings to hit the non-power-of-2 size).

**Training loop** (`tools/train.py`): standard alternating GAN update per batch — discriminator step on real + detached-fake, then generator step through a fresh fake batch. Both optimizers are Adam with `betas=(0.5, 0.999)` (the DCGAN-paper default). Checkpoints (`generator`/`discriminator` state dicts) are saved every epoch under `train_params.task_name/`, and auto-loaded on startup if both files already exist — resuming training is implicit, just rerun the same command. Sample grids are written to `task_name/samples/<count>.png` every `save_sample_steps` steps via the module-level `infer()` function.

**Config keys** (see any file under `config/`) are consumed as three top-level blocks — `dataset_params`, `model_params`, `train_params` — unpacked by key name throughout `tools/train.py`; there's no schema/validation, so a missing or misnamed key fails with a raw `KeyError` at the point of use.

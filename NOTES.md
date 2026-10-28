# Project Notes — DCGAN for Handwritten Digit Synthesis (MNIST)
### CA-2, Deep Learning and GEN AI (CS312B), Track A — Generative Adversarial Networks

These notes exist so that anyone on the team (or the examiner during viva) can
explain **every line of code**, not just run it. Read top to bottom.

---

## 1. The 5 W's

### What
A **DCGAN (Deep Convolutional GAN)** — two competing neural networks, a
**Generator** and a **Discriminator** — trained together so the Generator
learns to turn random noise into images that look like handwritten digits,
while the Discriminator learns to tell real digits from generated ("fake")
ones. After training, the Generator alone can produce new, never-seen digit
images from random noise.

### Why
- Satisfies **Track A (Unit 3 — GANs)** of the CA-2 brief: "high-resolution
  synthesis (DCGAN, WGAN-GP)".
- MNIST is small (28×28, grayscale, 60,000 images) so a DCGAN trains in
  **minutes, not hours** — realistic for a CA-2 deadline.
- DCGAN is the foundational convolutional GAN architecture (Radford et al.,
  2015) — understanding it is a prerequisite for every more advanced GAN
  (Pix2Pix, CycleGAN, StyleGAN, etc.), so it demonstrates core concepts
  (adversarial loss, transposed convolutions, BatchNorm, mode collapse risk)
  clearly.

### Who
- Built for a CS312B CA-2 submission (max 3-member team / individual).
- Codebase originally from `explainingai-code/DCGAN-Pytorch`
  (https://github.com/explainingai-code/DCGAN-Pytorch), adapted here to:
  - use a **single dataset path (MNIST only)** — the CelebA and
    "mnist-colored" variants are left in the repo but are not part of this
    project's scope.
  - **auto-download MNIST** via `torchvision` instead of requiring a manual
    PNG-folder data-prep step (the original README's "it downloads
    automatically" claim was not actually true of the original code — this
    was fixed, see §3).

### When
Implemented and trained as part of the CA-2 cycle. The actual training run
referenced below (losses, sample grids) is the live run from this session —
see the "Results" table in §5, directly reproducible with the commands in §2.

### Where
- **Code**: this repository (`model/`, `dataset/`, `tools/`, `config/`).
- **Dataset**: downloaded automatically from the official MNIST source,
  mirrored by `torchvision` — see §3 for the exact URL/mechanism.
- **Outputs**: `task_mnist/` — contains the two trained model checkpoints
  (`generator_mnist_ckpt.pth`, `discriminator_mnist_ckpt.pth`) and
  `task_mnist/samples/<n>.png`, a 10×10 grid of generated digits saved every
  100 training steps (93 grids total across the 20-epoch run), showing the
  visual progression from pure noise to recognisable digits.

---

## 2. How to run it

```bash
pip install -r requirements.txt
python -m tools.train --config config/mnist.yaml
```

That's it — no manual dataset download or folder setup required (see §3).
Training prints per-epoch stats to stdout and writes checkpoints +
sample-image grids into `task_mnist/`. Re-running the same command resumes
from the saved checkpoints automatically (see §4.3, the checkpoint-loading
block).

---

## 3. Dataset: source and preprocessing

- **Source**: MNIST (Modified National Institute of Standards and
  Technology) handwritten digit database. Official source:
  http://yann.lecun.com/exdb/mnist/ — 60,000 training + 10,000 test grayscale
  images, 28×28 pixels, digits 0–9.
- **Access mechanism**: `torchvision.datasets.MNIST(root=im_path,
  train=True, download=True)` — this is a built-in torchvision dataset class
  that downloads and caches the official MNIST files the first time it's
  instantiated (into the `data/` folder here, per `config/mnist.yaml`'s
  `im_path`). No manual download, unzip, or PNG export needed.
- **Why this differs from the original repo**: the original
  `dataset/mnist_dataset.py` expected you to have *already* extracted MNIST
  into `data/train/images/<digit>/*.png` yourself (via a separate repo's
  instructions) — it did **not** auto-download anything despite some online
  write-ups claiming it did. This project replaces that loader with
  `torchvision`'s built-in MNIST dataset so "clone + pip install + run"
  genuinely works with zero manual data prep.
- **Preprocessing** (`__getitem__` in `dataset/mnist_dataset.py`):
  1. `torchvision.transforms.ToTensor()` — converts the PIL image to a
     `[0, 1]`-range float tensor of shape `(1, 28, 28)` (1 channel =
     grayscale).
  2. `(2 * im_tensor) - 1` — rescales from `[0, 1]` to `[-1, 1]`. This matters
     because the Generator's last layer uses `Tanh`, whose output range is
     `[-1, 1]` — real and fake images must be on the same numeric scale for
     the Discriminator to compare them meaningfully.

---

## 4. Line-by-line code walkthrough

### 4.1 `dataset/mnist_dataset.py` — loads and preprocesses MNIST

```python
import torchvision
from torch.utils.data.dataset import Dataset
```
Import torchvision (gives us the built-in `MNIST` dataset + image
transforms) and PyTorch's base `Dataset` class, which we subclass so this
object works with PyTorch's `DataLoader` (batching, shuffling, etc.).

```python
class MnistDataset(Dataset):
```
Defines our dataset wrapper. Subclassing `Dataset` requires implementing
`__len__` and `__getitem__` (below) — PyTorch calls these internally.

```python
    def __init__(self, split, im_path, im_size=28, im_channels=1, im_ext='png'):
        self.split = split
        self.im_size = im_size
        self.im_channels = im_channels
```
Constructor signature matches what `tools/train.py` calls it with. We store
`split` (`'train'`/`'test'`) and image shape info for reference
(`im_size`/`im_channels` aren't used for resizing here since MNIST images are
already a fixed 28×28×1 — they exist mainly so the config stays self
describing). `im_ext` is unused now (kept only for signature compatibility
with the original repo).

```python
        is_train = split == 'train'
        self.data = torchvision.datasets.MNIST(root=im_path, train=is_train, download=True)
        print('Found {} images for split {}'.format(len(self.data), self.split))
```
`is_train` converts our `'train'`/`'test'` string into the boolean
torchvision expects. `torchvision.datasets.MNIST(..., download=True)` is the
line doing the actual work: if the MNIST files aren't already cached under
`im_path`, it downloads them from the official source and unpacks them; if
they are cached, it loads them instantly from disk. `self.data` now behaves
like a list of `(PIL_image, label)` pairs. We print a count purely for
visibility while training starts.

```python
    def __len__(self):
        return len(self.data)
```
Tells `DataLoader` how many samples exist, so it knows how many
batches/iterations make up one epoch.

```python
    def __getitem__(self, index):
        im, label = self.data[index]
        im_tensor = torchvision.transforms.ToTensor()(im)
```
Fetches the `index`-th `(image, label)` pair. We discard `label` — this is
**unconditional** generation (plain DCGAN produces "a digit", not "specifically
a 7"), so labels aren't used anywhere in training. `ToTensor()` converts the
PIL image into a `(1, 28, 28)` float tensor scaled to `[0, 1]`.

```python
        im_tensor = (2 * im_tensor) - 1
        return im_tensor
```
Rescales to `[-1, 1]` (see §3) and returns just the image tensor — the
`DataLoader` later stacks many of these into a batch tensor of shape
`(batch_size, 1, 28, 28)`.

---

### 4.2 `model/dcgan.py` — Generator and Discriminator architectures

```python
import torch
import torch.nn as nn
```
Standard PyTorch imports — `nn` gives us layer building blocks
(`Conv2d`, `BatchNorm2d`, etc.).

```python
class Generator(nn.Module):
    def __init__(self, latent_dim, im_size, im_channels,
                 conv_channels, kernels, strides, paddings,
                 output_paddings):
        super().__init__()
        self.latent_dim = latent_dim
        self.im_size = im_size
        self.im_channels = im_channels
```
`latent_dim` is the size of the random noise vector fed in (100 here — a
point in a 100-dimensional "latent space"). `im_size`/`im_channels` describe
the target output image (28×28×1 for MNIST). The remaining args
(`conv_channels`, `kernels`, `strides`, `paddings`, `output_paddings`) are
**lists**, one entry per layer, pulled straight from the YAML config — this
is what makes the architecture config-driven instead of hardcoded.

```python
        activation = nn.ReLU()
        layers_dim = [self.latent_dim] + conv_channels + [self.im_channels]
```
Builds the full channel-width sequence: start at 100 (noise), pass through
each width in `conv_channels` (`[512, 256, 128]` for MNIST), end at 1
(grayscale output channel). So `layers_dim = [100, 512, 256, 128, 1]` —
4 transitions = 4 conv-transpose layers.

```python
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(layers_dim[i], layers_dim[i + 1],
                                   kernel_size=kernels[i],
                                   stride=strides[i],
                                   padding=paddings[i],
                                   output_padding=output_paddings[i],
                                   bias=False),
                nn.BatchNorm2d(layers_dim[i + 1]) if i != len(layers_dim) - 2 else nn.Identity(),
                activation if i != len(layers_dim) - 2 else nn.Tanh()
            )
            for i in range(len(layers_dim) - 1)
        ])
```
Builds one `nn.Sequential` block per layer transition and wraps them in an
`nn.ModuleList` (so PyTorch tracks all their parameters for training):
- `ConvTranspose2d` — a "deconvolution": it **upsamples** spatially (grows
  height/width) while changing the channel count from `layers_dim[i]` to
  `layers_dim[i+1]`. `bias=False` because the following `BatchNorm2d`
  already re-centers the output, making a bias term redundant.
- `BatchNorm2d(...) if i != len(layers_dim) - 2 else nn.Identity()` — every
  layer gets BatchNorm (stabilizes training — a core DCGAN-paper
  recommendation) **except the last layer** (`nn.Identity()` = pass-through,
  i.e. no-op), because we want the final output to be the raw image, not a
  normalized value.
- `activation if ... else nn.Tanh()` — every layer uses `ReLU` **except the
  last**, which uses `Tanh` to squash the final output into `[-1, 1]`
  (matching the dataset's normalization range from §3).

```python
    def forward(self, z):
        batch_size = z.shape[0]
        out = z.reshape(-1, self.latent_dim, 1, 1)
        for layer in self.layers:
            out = layer(out)
        out = out.reshape(batch_size, self.im_channels, self.im_size, self.im_size)
        return out
```
`forward` defines what happens when you call `generator(noise)`. The input
`z` is a flat `(batch_size, 100)` noise tensor; it's reshaped into
`(batch_size, 100, 1, 1)` — a "1×1 image with 100 channels" — because
`ConvTranspose2d` needs a 4D spatial input to start upsampling from. Each
layer is applied in sequence, progressively growing the spatial size
(1×1 → 4×4 → ... → 28×28) while shrinking the channel count (100 → 512 →
256 → 128 → 1). The final reshape is a safety/shape-enforcement step to
guarantee the exact `(batch, channels, H, W)` shape the rest of the code
expects.

```python
class Discriminator(nn.Module):
    def __init__(self, im_size, im_channels,
                 conv_channels, kernels, strides, paddings):
        super().__init__()
        self.img_size = im_size
        self.im_channels = im_channels
        activation = nn.LeakyReLU()
        layers_dim = [self.im_channels] + conv_channels + [1]
```
Mirror image of the Generator: starts at `im_channels` (1, the real/fake
image), passes through `conv_channels` (`[128, 256, 512]`), ends at `1` — a
single scalar "real vs fake" score per image. `LeakyReLU` (not plain `ReLU`)
is the DCGAN-paper recommendation for discriminators — it avoids "dead"
neurons that plain ReLU can cause when gradients are negative.

```python
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(layers_dim[i], layers_dim[i + 1],
                          kernel_size=kernels[i],
                          stride=strides[i],
                          padding=paddings[i],
                          bias=False if i != 0 else True),
                nn.BatchNorm2d(layers_dim[i + 1]) if i != len(layers_dim) - 2 and i != 0 else nn.Identity(),
                activation if i != len(layers_dim) - 2 else nn.Identity()
            )
            for i in range(len(layers_dim) - 1)
        ])
```
Regular (not transposed) `Conv2d` layers — these **downsample** spatially
(shrink H/W) while growing channel count, the exact reverse of the
Generator. Three special-cases worth knowing cold for the viva:
- `bias=False if i != 0 else True` — the **first** layer keeps its bias (it
  operates directly on raw, un-normalized pixel input, so a bias term is
  useful), later layers drop it (again, BatchNorm after them makes it
  redundant).
- BatchNorm is skipped on **both** the first layer (`i != 0`) and the last
  layer (`i != len(layers_dim) - 2`) — not applying BatchNorm to the raw
  input layer is a standard DCGAN-paper rule to avoid oscillation early in
  training.
- The last layer has `nn.Identity()` instead of an activation — it outputs
  a **raw logit**, not a probability. This is intentional: `nn.Identity()`
  is a no-op passthrough (keep output as-is), and the training loop pairs
  this raw score with `BCEWithLogitsLoss`, which applies `Sigmoid`
  internally in a more numerically-stable way than doing it manually.

```python
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out.reshape(x.size(0))
```
Runs the real/fake image `x` through every conv layer, then reshapes the
final `(batch, 1, 1, 1)` output down to a flat `(batch,)` vector of raw
scores — one score per image in the batch.

---

### 4.3 `tools/train.py` — the training loop

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
Uses the GPU if one is available (it was, in this run — an NVIDIA RTX 4050),
otherwise falls back to CPU.

```python
def train(args):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
```
Loads the YAML config file (`config/mnist.yaml`) passed via `--config` into
a nested Python dict.

```python
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']
```
Splits the config dict into its three logical blocks, used throughout the
rest of the function.

```python
    im_dataset_cls = {
        'mnist': MnistDataset,
        'celeb': CelebDataset
    }.get(dataset_config['name'])

    im_dataset = im_dataset_cls(split='train',
                                im_path=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'],
                                im_ext=dataset_config['im_ext'])
    mnist_loader = DataLoader(im_dataset, batch_size=train_config['batch_size'], shuffle=True)
```
Picks the dataset class by name from the config (`'mnist'` → our
`MnistDataset`), instantiates it (this is where MNIST gets downloaded —
see §3), then wraps it in a `DataLoader`, which handles batching
(`batch_size=128`) and shuffling (new random order every epoch, important so
the model doesn't memorize batch order).

```python
    generator = Generator(latent_dim=model_config['latent_dim'],
                          im_size=dataset_config['im_size'],
                          im_channels=dataset_config['im_channels'],
                          conv_channels=model_config['generator_channels'],
                          kernels=model_config['generator_kernels'],
                          strides=model_config['generator_strides'],
                          paddings=model_config['generator_paddings'],
                          output_paddings=model_config['generator_output_paddings']).to(device)

    discriminator = Discriminator(im_size=dataset_config['im_size'],
                                  im_channels=dataset_config['im_channels'],
                                  conv_channels=model_config['discriminator_channels'],
                                  kernels=model_config['discriminator_kernels'],
                                  strides=model_config['discriminator_strides'],
                                  paddings=model_config['discriminator_paddings']).to(device)
    generator.train()
    discriminator.train()
```
Builds both networks straight from config values (see §4.2) and moves them
onto the GPU/CPU (`.to(device)`). `.train()` puts both models in "training
mode" (affects layers like BatchNorm, which behave differently at train vs.
eval time).

```python
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
```
Creates the output folder (`task_mnist/`) if it doesn't already exist.

```python
    if os.path.exists(...'generator'...) and os.path.exists(...'discriminator'...):
        print('Loading checkpoints as found them')
        generator.load_state_dict(torch.load(...))
        discriminator.load_state_dict(torch.load(...))
```
If both checkpoint files already exist from a previous run, loads them
back in — this is what makes "just rerun the same command" resume training
instead of restarting from scratch.

```python
    optimizer_generator = Adam(generator.parameters(), lr=train_config['lr'], betas=(0.5, 0.999))
    optimizer_discriminator = Adam(discriminator.parameters(), lr=train_config['lr'], betas=(0.5, 0.999))
    criterion = torch.nn.BCEWithLogitsLoss()
```
Each network gets its own Adam optimizer (they're trained independently, in
alternating steps). `betas=(0.5, 0.999)` is the DCGAN-paper's recommended
Adam momentum setting (default PyTorch Adam uses `(0.9, 0.999)` — the lower
first beta reduces oscillation specific to adversarial training).
`BCEWithLogitsLoss` = binary cross-entropy loss that expects **raw logits**
as input (matches the Discriminator's final `Identity()` layer — see §4.2).

```python
    for epoch_idx in range(train_config['num_epochs']):
        ...
        for im in tqdm(mnist_loader):
            real_ims = im.float().to(device)
            batch_size = real_ims.shape[0]
```
Outer loop = epochs (20 here); inner loop = one batch of real images at a
time (`tqdm` just adds the progress bar you see in the terminal).

```python
            optimizer_discriminator.zero_grad()
            fake_im_noise = torch.randn((batch_size, model_config['latent_dim']), device=device)
            fake_ims = generator(fake_im_noise)
            real_label = torch.ones((batch_size, 1), device=device)
            fake_label = torch.zeros((batch_size, 1), device=device)
```
**Discriminator step, part 1**: zero out old gradients. Sample fresh random
noise and run it through the Generator to get a batch of fake images.
Define target labels: real images should be labeled `1`, fake images `0`.

```python
            disc_real_pred = discriminator(real_ims)
            disc_fake_pred = discriminator(fake_ims.detach())
            disc_real_loss = criterion(disc_real_pred.reshape(-1), real_label.reshape(-1))
            mean_real_dis_preds.append(torch.nn.Sigmoid()(disc_real_pred).mean().item())
            disc_fake_loss = criterion(disc_fake_pred.reshape(-1), fake_label.reshape(-1))
            mean_fake_dis_preds.append(torch.nn.Sigmoid()(disc_fake_pred).mean().item())
            disc_loss = (disc_real_loss + disc_fake_loss) / 2
            disc_loss.backward()
            optimizer_discriminator.step()
```
Runs the Discriminator on both real and fake images. `.detach()` on
`fake_ims` is critical: it stops gradients from flowing back into the
Generator during *this* step — we only want to update the Discriminator
here. Loss = average of (how wrong it was about real images) and (how
wrong it was about fake images). `Sigmoid()(...)` here is purely for
*logging* — converting the raw logit into a 0–1 "confidence" number to print
(`mean_real_dis_preds`/`mean_fake_dis_preds`), not used in the loss itself.
`.backward()` computes gradients, `.step()` applies the Adam update.

```python
            optimizer_generator.zero_grad()
            fake_im_noise = torch.randn((batch_size, model_config['latent_dim']), device=device)
            fake_ims = generator(fake_im_noise)
            disc_fake_pred = discriminator(fake_ims)
            gen_fake_loss = criterion(disc_fake_pred.reshape(-1), real_label.reshape(-1))
            gen_fake_loss.backward()
            optimizer_generator.step()
```
**Generator step**: sample a *new* batch of noise (fresh randomness, not
reused), generate fake images (this time **not** detached — gradients must
flow through the Generator). Run them through the Discriminator, but label
them as `real_label` (`1`) — the Generator's goal is to *fool* the
Discriminator into predicting "real" for its fakes. `.backward()` +
`.step()` update only the Generator's weights (the Discriminator isn't
touched this step, since we only call `optimizer_generator.step()`).

```python
            if steps % train_config['save_sample_steps'] == 0:
                generator.eval()
                infer(generated_sample_count, generator, train_config, model_config)
                generated_sample_count += 1
                generator.train()
```
Every `save_sample_steps` (100) training steps, temporarily switches the
Generator to `.eval()` mode, calls `infer()` (see below) to save a sample
image grid, then switches back to `.train()` to continue training.

```python
        torch.save(generator.state_dict(), ...)
        torch.save(discriminator.state_dict(), ...)
```
At the end of every epoch, saves both models' weights to disk (overwriting
the previous checkpoint) — this is what enables resuming later.

```python
def infer(generated_sample_count, generator, train_config, model_config):
    with torch.no_grad():
        fake_im_noise = torch.randn((train_config['num_samples'], model_config['latent_dim']), device=device)
        fake_ims = generator(fake_im_noise)
        fake_ims = (fake_ims + 1) / 2
        grid = make_grid(fake_ims, nrow=train_config['num_grid_rows'])
        img = torchvision.transforms.ToPILImage()(grid)
        ...
        img.save(...)
```
`torch.no_grad()` disables gradient tracking (we're not training here, just
generating — saves memory/compute). Generates `num_samples` (100) fresh
images from random noise, rescales from `[-1, 1]` back to `[0, 1]`
(inverse of the preprocessing in §3, needed to save as a viewable image),
arranges them into a `num_grid_rows`×`num_grid_rows` (10×10) grid, converts
to a PIL image, and saves as a PNG under `task_mnist/samples/`.

```python
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for dcgan training')
    parser.add_argument('--config', dest='config_path', default='config/mnist.yaml', type=str)
    args = parser.parse_args()
    train(args)
```
Script entry point: parses the `--config` command-line flag (defaults to
MNIST) and kicks off `train()`.

---

### 4.4 `config/mnist.yaml` — all hyperparameters in one place

```yaml
dataset_params:
  im_path: 'data'        # where torchvision downloads/caches MNIST
  im_ext: 'png'           # unused now, kept for interface compatibility
  im_channels: 1           # grayscale
  im_size: 28               # MNIST's native resolution
  name: 'mnist'             # selects MnistDataset in train.py's dict lookup

model_params:
  latent_dim: 100                              # noise vector size fed to Generator
  generator_channels: [512, 256, 128]          # hidden channel widths, noise -> image
  generator_kernels: [4,4,4,4]                 # conv kernel size per layer
  generator_strides: [2,2,2,2]                 # upsampling factor per layer (x2 each)
  generator_paddings: [1,1,1,1]
  generator_output_paddings: [1,1,0,0]         # fine-tunes exact output size (1x1->4->7->14->28)
  discriminator_channels: [128, 256, 512]      # hidden channel widths, image -> score (mirrors generator)
  discriminator_kernels: [4,4,4,4]
  discriminator_strides: [2,2,2,2]
  discriminator_paddings: [1,1,1,1]

train_params:
  task_name: 'task_mnist'    # output folder name
  batch_size: 128             # images per training step
  num_epochs: 20               # full passes over the 60,000-image dataset
  num_samples: 100              # images generated per saved sample grid
  num_grid_rows: 10              # grid is 10x10 = 100 images
  save_sample_steps: 100          # save a sample grid every 100 steps
  lr: 0.0005                       # Adam learning rate (same for both networks)
  ckpt_name: 'mnist_ckpt.pth'      # checkpoint filename suffix
```
These per-layer lists (`generator_channels`/`kernels`/`strides`/etc.) are
what `model/dcgan.py` consumes to build the network (§4.2) — changing
network depth/width means editing these lists (same length across all of
them), not the model code itself.

---

## 5. Results from this training run

Command used: `python -m tools.train --config config/mnist.yaml`
Hardware: NVIDIA RTX 4050 (GPU), ~17–20 seconds/epoch, 20 epochs total
(≈6 minutes end-to-end).

| Epoch | Generator Loss | Discriminator Loss | D(real) | D(fake) |
|------:|----------------:|--------------------:|--------:|--------:|
| 1  | 2.8301 | 0.3934 | 0.7387 | 0.2607 |
| 2  | 1.8959 | 0.3909 | 0.7250 | 0.2733 |
| 5  | 2.6919 | 0.2584 | 0.8184 | 0.1804 |
| 10 | 3.8486 | 0.1315 | 0.9059 | 0.0938 |
| 15 | 4.2258 | 0.1205 | 0.9190 | 0.0808 |
| 20 | 4.1816 | 0.1562 | 0.9039 | 0.0962 |

(Full 20-row table is in `train_mnist.log`.)

**Reading these numbers**: `D(real)`/`D(fake)` are the Discriminator's mean
predicted "real-ness" (via Sigmoid) for real vs. generated images. By epoch
20 it's ~90% confident on real images and only ~10% fooled by fakes — the
Discriminator is winning the adversarial game faster than the Generator is
catching up (Generator loss trending up, Discriminator loss trending down).
This is a textbook GAN training dynamic worth discussing on the "Results"
slide: it doesn't mean the model failed — the qualitative samples (below)
are still clearly recognisable digits — but it does mean a longer run, a
lower Discriminator learning rate, or label smoothing would likely improve
balance further.

**Qualitative outputs**: `task_mnist/samples/0.png` (first saved grid, pure
noise/random textures) vs. `task_mnist/samples/93.png` (last saved grid,
clear recognisable digits 0–9 in varied handwriting styles) — use these two
side by side on the "Results" slide as the before/after comparison.

**Artifacts for submission**:
- `task_mnist/generator_mnist_ckpt.pth`, `task_mnist/discriminator_mnist_ckpt.pth` — trained weights.
- `task_mnist/samples/*.png` — 93 sample grids showing training progression.
- `train_mnist.log` — full training console output (all 20 epochs' stats).

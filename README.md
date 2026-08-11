# Oxford Pets Transfer Learning

Fine-tuning a frozen ResNet-18 to tell apart 37 cat and dog breeds from the Oxford-IIIT Pet dataset. Only the final layer trains: 18,981 parameters out of 11,195,493, which is 0.17% of the model.

## What this is

ResNet-18 comes with weights already trained on ImageNet, which is about 1.2 million photos across 1000 categories. Its last layer, `net.fc`, is a `Linear(512, 1000)`: everything before that layer has boiled the image down to 512 numbers, and that layer turns those 512 numbers into one score per ImageNet class. Golden retriever, tabby cat, school bus, and 997 others.

Those 512 numbers are the interesting part. To produce them the network already had to learn edges, then texture, then fur patterning, ear shape, eye placement. None of that is specific to ImageNet's particular list of 1000 things. So I threw away the 1000-way classifier, bolted on a `Linear(512, 37)` for the 37 breeds, and froze everything else:

```python
for param in net.parameters():
    param.requires_grad = False
net.fc = nn.Linear(512, 37)
```

Assigning `net.fc` after the freeze loop is what makes this work. The new layer is a fresh module, so it never got `requires_grad = False`, and it ends up being the only thing the optimizer touches. So what carries over from ImageNet is the feature extractor underneath, while the part that actually assigns labels gets thrown away and rebuilt.

## Why freeze everything else

There are 3,680 training images across 37 classes, so roughly 99 photos per breed. ImageNet had 1.2 million. If I let gradients into all 11.2 million parameters I'd be fitting about 3,000 parameters per training image, which is nowhere near enough evidence to pin them down. The weights would drift toward accidents of these specific photos: the backgrounds they were shot against, the lighting, the cameras. They'd score better on this dataset while getting worse at anything else.

Freezing keeps the feature extractor exactly as ImageNet left it, so the only thing I'm training is the mapping from those 512 features onto 37 labels. That's a much smaller problem, and 3,300 images is enough to constrain it.

Two side benefits I didn't expect going in. The backward pass is much faster, and I had the reason wrong at first: I assumed gradients were travelling a shorter distance. That's not what happens. Since every backbone parameter has `requires_grad = False` and the input doesn't require grad either, autograd never records a graph for those layers during the forward pass, so there's nothing there for backward to traverse. It computes gradients for `fc`'s 18,981 parameters and stops. It saves memory too, because the intermediate activations that backward would otherwise have to keep around are never stored.

The second benefit is stability: a randomly-initialised `fc` produces large, meaningless gradients in the first few batches, and if the rest of the network were unfrozen, those would flow back and damage weights that took 1.2 million images to learn.

## Setup

Python 3.13, with torch 2.13.0 and torchvision 0.28.0.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision
python Ml.py
```

The dataset downloads itself. `datasets.OxfordIIITPet(root="data", download=True)` pulls it into `./data` on first run, which is gitignored. The download is about 790MB, and it ends up using around 1.6GB on disk because torchvision keeps the tarballs next to the extracted images.

The dataset ships with a fixed split: 3,680 images in `trainval` and 3,669 in `test`, decided by the dataset authors rather than by me. The reason it's pre-determined is comparability. If everyone split it differently, no two published accuracy numbers would mean the same thing. It's the same idea as fixing a random seed, just applied to the data.

I then split `trainval` again into 3,300 for training and 380 for validation, with a seed of 42 so that split at least is stable between runs.

## Details worth explaining

**The normalization numbers aren't mine.** The transform uses mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`. I didn't compute these from the pet photos, and computing them would be wrong. They describe *ImageNet's* pixel statistics. The pretrained weights were learned on inputs normalized this way, so feeding in anything else puts every layer outside the input range it was tuned for. Normally you'd calculate mean and std from your own dataset; with a pretrained model you inherit them.

A consequence that doubles as a sanity check: after `ToTensor()` pixels run 0 to 1, but after normalizing they run roughly -2.12 to 2.64, since `(0 - 0.485) / 0.229 = -2.12` and `(1 - 0.406) / 0.225 = 2.64`. If a tensor still tops out at 1.0, the normalize step didn't happen.

**`num_workers=4` and the `__main__` guard.** The guard is mandatory here, and I found that out by leaving it off. Workers are separate processes, and on macOS a new process re-imports the script from the top to get its definitions. Without the guard, each worker reaches the `DataLoader` line and tries to spawn four workers of its own. Python catches this and stops:

```
RuntimeError:
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module.
```

It took me a while to connect that message to `num_workers`, since nothing in it mentions the DataLoader. Everything that runs goes under `if __name__ == "__main__":` for that reason. Setting `num_workers=0` also makes it go away, which is a tempting fix and the wrong one.

What the workers actually do is prepare batches: open the JPEG, decode it, resize to 224x224, normalize, and stack 32 of them into one `[32, 3, 224, 224]` tensor. With `num_workers=0` that happens between training steps, so the GPU sits idle waiting for it. With 4, batches are being prepared while the model trains on the previous one. At 224x224 the decode-and-resize is genuinely expensive, so this matters here in a way it wouldn't on small images.

**Why `net.train()` sits inside the epoch loop.** It has to be there because validation runs at the end of every epoch and leaves the model in eval mode, so it has to be switched back at the top of the next one. What the modes change is BatchNorm. In train mode it normalizes using the current batch's statistics; in eval mode it uses a running average accumulated during training. At batch size 32 one batch's statistics are a rough estimate, which is fine while training but wrong at evaluation time, where an image shouldn't be scored differently depending on which other images happened to share its batch.

`torch.no_grad()` is a separate switch and it's easy to conflate the two. `eval()` changes what layers do. `no_grad()` turns off gradient bookkeeping. You want both during validation, for different reasons.

**Device is `mps` if available.** On my previous project (32x32 images, 61,326 parameters) moving to the GPU wasn't worth it, because the CPU spent all its time decoding files while the arithmetic itself was trivial. Here it's 49 times the pixels and about 180 times the parameters, so there's finally enough work to keep the GPU busy between transfers.

## Results

Validation accuracy, percent, over 5 epochs. Two runs of identical code with identical hyperparameters:

| run | epoch 0 | 1 | 2 | 3 | 4 |
|-----|------|------|------|------|------|
| 1 | 82.9 | 86.6 | 85.5 | 85.3 | 85.3 |
| 2 | 81.8 | 83.9 | 84.2 | 87.1 | 87.1 |

This is the part of the project I found most interesting, and it wasn't what I set out to look at.

Nothing differs between these two runs except the random initialisation of `fc` and the order the shuffled batches came in. The train/val split is seeded, so both runs saw exactly the same 380 validation images, but I never seeded anything else. Yet they end 1.8 points apart, and more importantly they have completely different *shapes*. Run 1 looks like a textbook overfitting curve: peaks at epoch 1, then declines and flattens. Run 2 looks like it's still improving at epoch 4 and would benefit from more training. Either run on its own would support a confident conclusion, and one of those conclusions has to be wrong.

The dip in run 1 is 1.3 points, which on a 380-image validation set works out to about five images changing their minds. Five images is a small enough difference that I doubt anything real is happening there, even though the curve looks like it is.

With 380 validation images the standard error on an accuracy near 85% is `sqrt(0.85 * 0.15 / 380)` = 1.8 percentage points. I want to be careful about that number, because one standard error is only about a 68% interval, and quoting it as though it were the threshold for a real result is exactly the mistake this section is warning about. The 95% interval is roughly twice as wide, ±3.6 points, and that's the figure to use when deciding whether a change did anything.

So the 1.8 point gap between my two runs is about one standard error, comfortably inside what reshuffling alone could produce. I have no grounds to call either run better.

One more caveat on that interval. ±3.6 is the uncertainty on a single accuracy against the true value. Comparing my two runs is a paired comparison, since both were scored on the same 380 images, and the correct test is McNemar's, which needs the count of images the two runs disagreed on. I didn't record that, so I can't run it.

That matters more than it first looks. The paired test would give a tighter bound than the unpaired one, and a tighter bound could in principle make 1.8 points significant. So ±3.6 doesn't show the two runs are equivalent. It shows I don't have the evidence to separate them, which is a weaker and less satisfying claim. My conclusion above should be read as "I can't tell" instead of "there's no difference", and the fix is recording which images each run got wrong.

For rough context, a 2026 paper (arXiv 2602.07534, Hera et al., "Fine-Grained Cat Breed Recognition with Global Context Vision Transformer", ICCIT 2025) reports 92.00% test accuracy with a GCViT-Tiny on this dataset. Their task is only the 12 cat breeds though, so it's a 12-way problem and my 37-way number doesn't line up against it. The same goes for their baseline table (VGG16 60.85%, ResNet50 71.39%, InceptionV3 84.94%, fine-tuned Xception 88.8%), which is all on that 12-class subset. Those baselines carry a further problem the paper is upfront about: they come from a re-split protocol where the test data was drawn from the training distribution, so they can't be compared to each other either, never mind to me.

I nearly wrote this paragraph as though my 85% sat respectably among those numbers. It doesn't, because they aren't measuring the same thing. Two details make that clearer. A 12-class problem is substantially easier than a 37-class one, so their VGG16 and ResNet50 baselines were solving something simpler and still landed well below my number, which should make me suspicious of those baselines before it makes me pleased with mine. And their model trains with rotation, horizontal flip, and brightness augmentation, which I use none of.

The honest position is that I have no comparable figure at all, because I haven't run my test set. That's the gap, and no amount of quoting other people's numbers fills it.

## Limitations

- No global seed. Only the train/val split is seeded, so weight initialisation and shuffle order still vary between runs, and two runs of the same code gave 85.3% and 87.1%.
- The validation set is 380 images across 37 classes, about 10 per class. At that size the 95% interval is ±3.6 points, so any improvement smaller than that is invisible here.
- The test set has never been run, so every number here is a validation number. Published results I found are on the 12-breed cat subset, so even after running the test set I'd need a 37-class source to compare against.
- No per-class accuracy, so I don't know which breeds fail. With 37 fine-grained classes, some are certainly far worse than the average, and the average hides that.
- No augmentation, and I don't log training accuracy, so the gap between train and validation is unmeasured. I can't actually confirm or rule out overfitting.

## Next steps

- Set a global seed, covering weight init and shuffle order, so two runs mean something when compared.
- Move to an 80/20 split, giving about 736 validation images. That takes the standard error from 1.8 down to 1.3 points, so the 95% interval narrows from ±3.6 to ±2.6. Better, though still not enough to see a one-point gain.
- Record per-image correctness on the validation set, so two runs can be compared with McNemar's test instead of by eyeballing two accuracy numbers.
- Unfreeze `layer4` at a low learning rate (1e-4) while `fc` keeps training at 0.01. The last block holds the most task-specific features, and this is usually the largest single gain available from a setup like this.
- Add `RandomHorizontalFlip` and `RandomResizedCrop`, to the training transform only. Augmenting validation would make the number noisier and mean something different each epoch.
- Try Adam at lr 0.001 instead of SGD.
- Then run the test set. Once, at the end.

## Attribution

Nothing here is mine except the 37-way layer on the end and the training loop. The two halves of this project come from entirely separate groups, who have nothing to do with each other:

**The model.** The ResNet architecture is from He et al., 2015, "Deep Residual Learning for Image Recognition", at Microsoft Research. The pretrained weights I actually load are distributed by torchvision (`ResNet18_Weights.DEFAULT`) under its BSD-3-Clause license. Those weights are only useful because they were trained on ImageNet (Deng et al., Stanford and Princeton), so the 512 features this project depends on are really ImageNet's contribution.

**The data.** The Oxford-IIIT Pet Dataset is from Parkhi et al., 2012, "Cats and Dogs", Visual Geometry Group, University of Oxford, released under CC BY-SA 4.0.

Worth being precise about, because it's the whole shape of the project: ResNet-18 was built to identify 1000 different subjects, and I need 37 of them, which aren't even a subset of the original 1000. That mismatch is the entire reason for the line `net.fc = nn.Linear(512, 37)`. Oxford supplied the 37 breeds and the photos, Microsoft supplied the architecture, ImageNet supplied the visual experience, and the one layer bridging them is what I trained.

No images from the dataset are redistributed here. `download=True` fetches them from Oxford on first run.

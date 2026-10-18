# Generalization to unseen generators, and dataset bias

## Why detectors trained on yesterday's generators struggle with today's

Ojha, Li, and Lee, in "Towards Universal Fake Image Detectors that
Generalize Across Generative Models" (CVPR 2023), studied why classifiers
trained to detect one family of generators (e.g. GANs) often fail badly
on a different family (e.g. diffusion models), even though both families
produce images that look similarly "fake" to a human. Their explanation
is that a classifier trained with a standard cross-entropy objective
tends to learn generator-specific artifacts rather than a general notion
of "generated-ness" — it effectively memorizes quirks of the generators
it saw during training. They proposed using large pretrained vision-
language features (e.g. from CLIP) with a simpler nearest-neighbor or
linear-probe decision rule on top, which generalized noticeably better to
generator families absent from training, compared to a classifier trained
end-to-end from scratch. The practical takeaway for any deployed detector
(including the one used in VerityAI) is that reported accuracy on a
held-out set from the *same* generators used in training should not be
read as the accuracy you would get on a brand-new generator released
after the detector was trained.

Source: Ojha, U., Li, Y., & Lee, Y. J. (2023). Towards Universal Fake
Image Detectors that Generalize Across Generative Models. *CVPR 2023*.
https://arxiv.org/abs/2302.10174

## Benchmark bias: CIFAKE as a cautionary example

Bird and Lotfi introduced CIFAKE (2023), a widely used benchmark that
pairs the real CIFAR-10 image set with a matching set of Stable-Diffusion-
generated images, and showed that a simple CNN classifier can reach very
high (>90%) accuracy distinguishing the two. CIFAKE is useful as a fast,
free, clearly-licensed benchmark, but it is also a good illustration of
dataset bias: CIFAR-10 images are only 32x32 pixels, so a classifier
trained on CIFAKE learns to separate "real CIFAR-10 photographs" from
"one specific generator's 32x32 outputs" — a much narrower and easier
problem than distinguishing a modern high-resolution AI-generated photo
from a modern high-resolution real one. High accuracy on CIFAKE (or any
single-generator, single-resolution benchmark) should not be read as
evidence that a detector will generalize to realistic, high-resolution,
multi-generator inputs "in the wild."

Source: Bird, J. J., & Lotfi, A. (2023). CIFAKE: Image Classification and
Explainable Identification of AI-Generated Synthetic Images. *arXiv:2303.14126*.
https://arxiv.org/abs/2303.14126

## What this means for VerityAI's own detector

VerityAI's default detector (see the About page for the exact model) was
trained on a specific, disclosed dataset and generator mix. Per the
findings above, that means:
- Its accuracy figures (when available from the model's own card) describe
  performance on *that* training distribution, not a universal guarantee.
- Images from generators released after the detector's training cut-off
  are effectively "unseen generators" to it, and the Ojha et al. findings
  suggest performance there is likely to be worse than on in-distribution
  images.
- A benchmark-style high accuracy number, if quoted anywhere, should be
  read with the CIFAKE lesson in mind: check what the benchmark actually
  contained before trusting the number as representative.

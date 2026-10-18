# Model cards: candidate detectors considered for VerityAI

This document summarizes the publicly available model cards for the
pretrained image-classification models considered as VerityAI's detector,
so the Research Assistant can answer questions like "why this model?" and
"what does the model actually claim?" with real, checkable information
rather than marketing claims.

## Nahrawy/AIorNot — selected as VerityAI's default detector

- **Base architecture:** `microsoft/swin-tiny-patch4-window7-224`
  (Swin Transformer, "tiny" size), fine-tuned for binary image
  classification.
- **Training data:** The "aiornot" image-classification competition
  dataset (~62,000 images labeled Real or AI).
- **License:** Apache-2.0 — fully permissive, including commercial use,
  modification, and redistribution. This was the deciding factor over
  several alternatives below that carry non-commercial or no-derivatives
  restrictions.
- **Size:** Swin-Tiny has roughly 28M parameters, the smallest of the
  candidates considered — important for an 8GB-RAM laptop and a
  roughly 1GB-RAM free hosting tier.
- **Known limitation (disclosed by VerityAI, not stated on the model
  card itself):** the aiornot competition predates the newest generation
  of image generators (e.g. SDXL-Turbo, Midjourney v6, DALL-E 3, Flux),
  so — per the generalization findings in document 02 — accuracy on
  those specific newer generators is unverified and likely lower than on
  the generators represented in its training data.

## Organika/sdxl-detector — considered, not selected

- **Base architecture:** Swin Transformer, fine-tuned from the older
  `umm-maybe/AI-image-detector`.
- **Training data:** Wikimedia images paired with SDXL-generated images
  produced from BLIP-captioned prompts of those same Wikimedia images —
  i.e. trained specifically to catch SDXL outputs.
- **License:** CC-BY-NC-3.0 — non-commercial use only. The model card
  itself notes the underlying detector's original training data may
  include images scraped from Reddit with uncertain copyright status,
  and recommends personal/educational use only.
- **Model size:** 86.8M parameters — roughly 3x larger than Swin-Tiny.
- **Why not selected:** narrower training scope (SDXL-specific) and a
  non-commercial license that would restrict portfolio/commercial reuse
  of this project's code if the model were bundled as the default.

## dima806/ai_vs_real_image_detection — considered, not selected

- **Base architecture:** `google/vit-base-patch16-224-in21k`
  (ViT-Base), fine-tuned for binary Real/Fake classification.
- **Training data:** Not explicitly documented on the model card, but
  the linked reference notebook strongly suggests the CIFAKE dataset
  (see document 02 for why CIFAKE's 32x32-upsampled images are not
  representative of realistic high-resolution inputs).
- **License:** Apache-2.0.
- **Reported metrics:** ~98% accuracy — but, per CIFAKE's known dataset
  bias, this number should not be read as representative of real-world,
  high-resolution performance.
- **Model size:** 85.8M parameters.
- **Why not selected:** likely CIFAKE-trained (per document 02's
  dataset-bias discussion), and larger than Swin-Tiny for no clear
  accuracy benefit on realistic inputs.

## umm-maybe/AI-image-detector — considered, not selected

- **Base architecture:** Vision Transformer (ViT), described by its own
  card as "a proof of concept."
- **Training data:** Created October 2022; predates Midjourney 5, SDXL,
  and DALL-E 3 entirely (per its own model card).
- **License:** Inconsistent on the model card itself — metadata says
  CC-BY-4.0, but the written license notice says CC-BY-NoDerivatives-4.0
  and explicitly prohibits using the model inside a text-to-image system
  to evade detection.
- **Scope:** The card itself states it targets *artistic* images and
  warns that ordinary photos, webcam captures, and screenshots "may
  throw it off."
- **Why not selected:** oldest of the candidates, license terms are
  internally inconsistent, and its own card recommends using the newer
  Organika/sdxl-detector instead for anything but very old generators
  like VQGAN+CLIP.

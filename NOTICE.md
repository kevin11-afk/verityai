# Third-party model notices

VerityAI's own source code is MIT-licensed (see LICENSE). The pretrained
models it downloads at runtime are distributed by their original authors
under their own licenses, summarized here for convenience — always check
the live model card for the current terms before any commercial reuse.

| Model | Hugging Face ID | License | Used for |
|---|---|---|---|
| Swin-Tiny fine-tune | `Nahrawy/AIorNot` | Apache-2.0 | Default image/video-frame detector |
| MiniLM | `sentence-transformers/all-MiniLM-L6-v2` | Apache-2.0 | RAG embeddings |

Alternative detectors discussed in `knowledge_base/04_detector_model_cards.md`
and considered but not bundled as the default (CC-BY-NC / CC-BY-ND
licensed) are **not** downloaded by this application unless you explicitly
change `VERITYAI_DETECTOR_MODEL` in `src/config.py` — doing so means you
accept that model's own license terms, not VerityAI's.

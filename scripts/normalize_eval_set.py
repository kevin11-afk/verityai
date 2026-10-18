#!/usr/bin/env python3
"""Control for format/size shortcuts: center-crop every image to a square,
resize to 512px and re-encode as JPEG q90, identically for both classes.

    python scripts/normalize_eval_set.py data/eval_sets/X data/eval_sets/X_normalized
"""
import sys
from pathlib import Path

from PIL import Image

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
for cls in ("real", "ai"):
    (dst / cls).mkdir(parents=True, exist_ok=True)
    for f in sorted((src / cls).glob("*")):
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            continue
        with Image.open(f) as im:
            im = im.convert("RGB")
            s = min(im.size)
            l, t = (im.width - s) // 2, (im.height - s) // 2
            im.crop((l, t, l + s, t + s)).resize((512, 512), Image.LANCZOS).save(dst / cls / (f.stem + ".jpg"), quality=90)
print("done", dst)

#!/usr/bin/env python3
"""Build lossless WebP alternatives without changing the publication artwork."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ('fig-product-map', 'fig-runtime-resources')


def verify_lossless(source, encoded):
    import io
    from PIL import Image

    with Image.open(source) as image:
        original = image.convert('RGBA')
        with Image.open(io.BytesIO(encoded)) as result:
            if result.size != original.size or result.convert('RGBA').tobytes() != original.tobytes():
                raise ValueError('Lossless image pixel verification failed: ' + str(source))


def encode_lossless(source):
    import io
    from PIL import Image

    with Image.open(source) as image:
        original = image.convert('RGBA')
        buffer = io.BytesIO()
        original.save(buffer, format='WEBP', lossless=True, method=6, exact=True)
        encoded = buffer.getvalue()
    verify_lossless(source, encoded)
    return encoded


def build(root=ROOT, check=False):
    root = Path(root)
    receipt_path = root / 'docs/website/published_assets.json'
    receipt = json.loads(receipt_path.read_text())
    records = []
    for name in IMAGES:
        source = root / 'docs/site/assets/img' / (name + '.png')
        target = source.with_suffix('.webp')
        if check:
            if not target.is_file():
                raise ValueError('Website WebP or receipt is stale: ' + name)
            data = target.read_bytes()
            verify_lossless(source, data)
        else:
            data = encode_lossless(source)
        if len(data) >= source.stat().st_size:
            raise ValueError('WebP is not smaller: ' + name)
        source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        output_sha = hashlib.sha256(data).hexdigest()
        entry = dict(sha256=output_sha, source_sha256=source_sha,
                     derivative='Lossless WebP; decoded RGBA pixels and dimensions identical to PNG')
        relative = target.relative_to(root).as_posix()
        if check:
            if receipt['files'].get(relative) != entry:
                raise ValueError('Website WebP or receipt is stale: ' + name)
        else:
            target.write_bytes(data)
            receipt['files'][relative] = entry
        records.append(dict(image=name, source_bytes=source.stat().st_size, webp_bytes=len(data),
                            decoded_pixels_equal=True, reduction_percent=round(100 * (1 - len(data) / source.stat().st_size), 2)))
    if not check:
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    print(json.dumps(dict(status='PASS', images=build(check=args.check)), indent=2))

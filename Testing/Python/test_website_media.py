"""Web delivery preserves source pixels and avoids unsolicited video downloads."""

import tempfile
import json
import hashlib
from pathlib import Path
import unittest

from scripts.optimize_website_images import encode_lossless, build
from scripts.optimize_website_video import faststart

ROOT = Path(__file__).resolve().parents[2]


class WebsiteMediaTests(unittest.TestCase):
    def test_lossless_webp_retains_rgb_and_transparency(self):
        import io
        from PIL import Image
        for mode, color in (('RGB', (36, 120, 214)), ('RGBA', (200, 30, 40, 120))):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'source.png'
                original = Image.new(mode, (23, 17), color)
                original.save(path)
                result = Image.open(io.BytesIO(encode_lossless(path))).convert('RGBA')
                self.assertEqual(result.size, original.size)
                self.assertEqual(result.tobytes(), original.convert('RGBA').tobytes())

    def test_published_webp_files_are_current_and_smaller(self):
        for record in build(check=True):
            self.assertTrue(record['decoded_pixels_equal'])
            self.assertLess(record['webp_bytes'], record['source_bytes'])

    def test_published_film_retains_full_frames_audio_and_verified_quality(self):
        entry = json.loads((ROOT / 'docs/website/published_assets.json').read_text())['files'][
            'docs/site/media/SlicerDSpace_promo_1080p.mp4']
        video = ROOT / 'docs/site/media/SlicerDSpace_promo_1080p.mp4'
        self.assertEqual(video.stat().st_size, entry['output_bytes'])
        self.assertEqual(hashlib.sha256(video.read_bytes()).hexdigest(), entry['sha256'])
        self.assertTrue(entry['audio_payload_identical'])
        self.assertEqual(entry['ssim_frames_checked'], 1944)
        self.assertGreaterEqual(entry['complete_frame_ssim'], .99)
        self.assertLess(entry['output_bytes'], entry['source_bytes'])
        self.assertTrue(faststart(video))
        before = next(row for row in entry['source_metadata']['streams'] if row['codec_type'] == 'video')
        after = next(row for row in entry['output_metadata']['streams'] if row['codec_type'] == 'video')
        for key in ('width', 'height', 'r_frame_rate', 'nb_read_frames'):
            self.assertEqual(before[key], after[key])


if __name__ == '__main__':
    unittest.main()

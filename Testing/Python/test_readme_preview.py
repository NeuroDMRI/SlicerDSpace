"""Keep the README animation moderate in size and preserve the complete timeline."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import make_readme_preview as preview


class ReadmePreviewTests(unittest.TestCase):
    def test_defaults_improve_resolution_and_cadence_without_excess(self):
        self.assertEqual(preview.DEFAULT_WIDTH, 800)
        self.assertEqual(preview.DEFAULT_FPS, 10)
        self.assertEqual(preview.frame_filters(960, 10, 1),
                         'setpts=(PTS-STARTPTS)/1,fps=10,scale=960:-2:flags=lanczos')

    def test_preview_preserves_all_transitions_at_moderate_speed(self):
        self.assertEqual(preview.DEFAULT_SPEED, 1.5)
        filters = preview.frame_filters(960, 10, preview.DEFAULT_SPEED)
        self.assertEqual(filters, 'setpts=(PTS-STARTPTS)/1.5,fps=10,scale=960:-2:flags=lanczos')
        for removed in ('select=', 'trim=', 'concat=', 'setpts=N/'):
            self.assertNotIn(removed, filters)

    def test_palette_and_animation_are_streamed_in_two_passes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'docs/site/media/SlicerDSpace_promo_1080p.mp4'
            source.parent.mkdir(parents=True)
            source.write_bytes(b'unchanged source video')
            target = root / 'preview.gif'
            commands = []

            def encode(command, **kwargs):
                self.assertTrue(kwargs['check'])
                commands.append(command)
                Path(command[-1]).write_bytes(b'GIF89a' if command[-1].endswith('.gif') else b'palette')

            with patch.object(preview, 'ROOT', root), patch.object(preview.subprocess, 'run', encode), \
                    patch('sys.argv', ['make_readme_preview', '--ffmpeg', 'ffmpeg', '--gifsicle', '', '--output', str(target)]), \
                    patch('builtins.print'):
                preview.main()
            self.assertEqual(len(commands), 2)
            self.assertIn('palettegen=max_colors=256:stats_mode=diff', commands[0][commands[0].index('-vf') + 1])
            self.assertNotIn('select=', commands[0][commands[0].index('-vf') + 1])
            self.assertIn('setpts=(PTS-STARTPTS)/1.5', commands[0][commands[0].index('-vf') + 1])
            self.assertIn('-loop', commands[1])
            self.assertEqual(commands[1][commands[1].index('-loop') + 1], '0')
            self.assertEqual(target.read_bytes(), b'GIF89a')
            self.assertEqual(source.read_bytes(), b'unchanged source video')
            self.assertFalse(list(root.glob('.readme-preview-*')))

    def test_checked_in_preview_is_smooth_moderate_and_decodable(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest('Pillow is needed to decode the preview')
        path = preview.ROOT / 'docs/assets/readme/SlicerDSpace_preview.gif'
        self.assertLess(path.stat().st_size, 20 * 1024 ** 2)
        with Image.open(path) as animation:
            self.assertEqual(animation.size, (560, 316))
            self.assertEqual(animation.info.get('loop'), 0)
            durations = []
            for index in range(animation.n_frames):
                animation.seek(index)
                animation.load()
                durations.append(animation.info['duration'])
            self.assertAlmostEqual(sum(durations) / 1000, 40.5, delta=.5)
            self.assertGreaterEqual(len(durations), 240)
            self.assertTrue(set(durations).issubset({160, 170}))

    def test_oversized_render_does_not_replace_previous_preview(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'docs/site/media/SlicerDSpace_promo_1080p.mp4'
            source.parent.mkdir(parents=True)
            source.write_bytes(b'video')
            target = root / 'preview.gif'
            target.write_bytes(b'previous preview')

            def encode(command, **kwargs):
                Path(command[-1]).write_bytes(b'x' * 2048)

            with patch.object(preview, 'ROOT', root), patch.object(preview.subprocess, 'run', encode), \
                    patch('sys.argv', ['make_readme_preview', '--ffmpeg', 'ffmpeg', '--gifsicle', '', '--output', str(target),
                                       '--max-size-mib', '.001']), patch('sys.stderr'), self.assertRaises(SystemExit):
                preview.main()
            self.assertEqual(target.read_bytes(), b'previous preview')

    def test_optional_compression_only_changes_encoding(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'docs/site/media/SlicerDSpace_promo_1080p.mp4'
            source.parent.mkdir(parents=True)
            source.write_bytes(b'video')
            target = root / 'preview.gif'
            commands = []

            def encode(command, **kwargs):
                commands.append(command)
                Path(command[-1]).write_bytes(b'GIF89a' if command[0] == 'gifsicle' else b'x' * 100)

            with patch.object(preview, 'ROOT', root), patch.object(preview.subprocess, 'run', encode), \
                    patch('sys.argv', ['make_readme_preview', '--ffmpeg', 'ffmpeg', '--gifsicle', 'gifsicle',
                                       '--output', str(target)]), patch('builtins.print'):
                preview.main()
            self.assertEqual(len(commands), 3)
            self.assertEqual(commands[-1][1:3], ['-O3', '--lossy=40'])
            for flag in ('--delay', '--resize', '--delete', '--speed'):
                self.assertNotIn(flag, commands[-1])
            self.assertEqual(target.read_bytes(), b'GIF89a')


if __name__ == '__main__':
    unittest.main()

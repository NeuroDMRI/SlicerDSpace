#!/usr/bin/env python3
"""Make a continuous, moderately accelerated README preview of the complete film."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WIDTH = 800
DEFAULT_FPS = 10
DEFAULT_SPEED = 1.5


def frame_filters(width, fps, speed):
    return f"setpts=(PTS-STARTPTS)/{speed:g},fps={fps},scale={width}:-2:flags=lanczos"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED, help="Playback multiplier; default: 1.5; 1 preserves normal speed")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--full", action="store_true", help=argparse.SUPPRESS)  # Compatibility: full timeline is now always used.
    parser.add_argument("--gifsicle", default=shutil.which("gifsicle"), help="Optional compressor for the README preview")
    parser.add_argument("--lossy", type=int, default=40, help="Gifsicle compression strength; 0 is lossless")
    parser.add_argument("--max-size-mib", type=float, default=20, help="Reject oversized previews without replacing the existing GIF")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/assets/readme/SlicerDSpace_preview.gif")
    args = parser.parse_args()
    if not args.ffmpeg:
        parser.error("Set --ffmpeg to an installed FFmpeg executable")
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be finite and positive")
    if not 320 <= args.width <= 1920 or args.width % 2:
        parser.error("--width must be even and between 320 and 1920")
    if not 1 <= args.fps <= 30:
        parser.error("--fps must be between 1 and 30")
    if not 0 <= args.lossy <= 100:
        parser.error("--lossy must be between 0 and 100")
    if not math.isfinite(args.max_size_mib) or args.max_size_mib <= 0:
        parser.error("--max-size-mib must be finite and positive")
    source = ROOT / "docs/site/media/SlicerDSpace_promo_1080p.mp4"
    target = args.output.resolve()
    if target == source.resolve():
        parser.error("The preview must not overwrite its source video")
    target.parent.mkdir(parents=True, exist_ok=True)
    frames = frame_filters(args.width, args.fps, args.speed)
    common = [args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-threads", "4"]
    compressed_output = False
    # A separate palette pass avoids buffering the whole animation in the split graph.
    with tempfile.TemporaryDirectory(prefix=".readme-preview-", dir=target.parent) as folder:
        palette = Path(folder) / "palette.png"
        preview = Path(folder) / "preview.gif"
        subprocess.run(common + ["-i", str(source), "-filter_threads", "2", "-vf",
                       frames + ",palettegen=max_colors=256:stats_mode=diff", "-frames:v", "1", str(palette)], check=True)
        filters = (frames + "[frames];[frames][1:v]"
                   "paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle")
        subprocess.run(common + ["-i", str(source), "-i", str(palette), "-filter_complex_threads", "2",
                       "-filter_complex", filters, "-an", "-loop", "0", str(preview)], check=True)
        if args.gifsicle:
            compressed = Path(folder) / "compressed.gif"
            subprocess.run([args.gifsicle, "-O3", f"--lossy={args.lossy}", str(preview), "-o", str(compressed)], check=True)
            if compressed.stat().st_size < preview.stat().st_size:
                preview = compressed
                compressed_output = True
        if preview.stat().st_size >= args.max_size_mib * 1024 ** 2:
            parser.error("Preview exceeds --max-size-mib; use --gifsicle or reduce --width/--fps. Existing GIF is unchanged.")
        preview.replace(target)
    print(json.dumps(dict(file=str(target), bytes=target.stat().st_size,
                          width=args.width, fps=args.fps, max_colors=256, playback_speed=args.speed,
                          timeline="complete film; no excerpts or removed transitions",
                          compression="gifsicle" if compressed_output else "ffmpeg", lossy=args.lossy if compressed_output else 0,
                          source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                          preview_sha256=hashlib.sha256(target.read_bytes()).hexdigest()), indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Encode and verify a smaller 1080p website film from the retained master."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'docs/site/media/SlicerDSpace_promo_1080p.mp4'


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def faststart(path):
    boxes = []
    with Path(path).open('rb') as stream:
        total = Path(path).stat().st_size
        position = 0
        while position < total:
            stream.seek(position)
            header = stream.read(16)
            if len(header) < 8:
                return False
            size, kind = struct.unpack('>I4s', header[:8])
            if size == 1:
                size = struct.unpack('>Q', header[8:16])[0]
            if size == 0:
                size = total - position
            if size < 8 or position + size > total:
                return False
            boxes.append(kind)
            position += size
    return b'moov' in boxes and b'mdat' in boxes and boxes.index(b'moov') < boxes.index(b'mdat')


def video_info(ffprobe, path):
    result = subprocess.run([ffprobe, '-v', 'error', '-count_frames', '-show_entries',
        'stream=codec_type,codec_name,width,height,r_frame_rate,nb_read_frames:format=duration,size',
        '-of', 'json', str(path)], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def audio_hash(ffmpeg, path):
    result = subprocess.run([ffmpeg, '-v', 'error', '-nostdin', '-i', str(path), '-map', '0:a:0',
        '-c', 'copy', '-f', 'hash', '-hash', 'sha256', '-'], check=True, capture_output=True, text=True)
    return result.stdout.strip().removeprefix('SHA256=')


def verify(ffmpeg, source, output, verification_dir):
    ffprobe = Path(ffmpeg).with_name('ffprobe')
    before, after = video_info(str(ffprobe), source), video_info(str(ffprobe), output)
    original_video = next(row for row in before['streams'] if row['codec_type'] == 'video')
    encoded_video = next(row for row in after['streams'] if row['codec_type'] == 'video')
    fields = ('width', 'height', 'r_frame_rate', 'nb_read_frames')
    if any(original_video[field] != encoded_video[field] for field in fields):
        raise ValueError('Video geometry, frame rate or frame count changed')
    if abs(float(before['format']['duration']) - float(after['format']['duration'])) > .01:
        raise ValueError('Video duration changed')
    source_audio, output_audio = audio_hash(ffmpeg, source), audio_hash(ffmpeg, output)
    if source_audio != output_audio:
        raise ValueError('Compressed audio payload changed')
    if not faststart(output) or output.stat().st_size >= source.stat().st_size:
        raise ValueError('Expected a smaller MP4 with fast-start metadata')
    verification_dir.mkdir(parents=True, exist_ok=True)
    stats = verification_dir / 'ssim_frames.txt'
    result = subprocess.run([ffmpeg, '-hide_banner', '-nostdin', '-nostats', '-threads', '2', '-i', str(source),
        '-threads', '2', '-i', str(output), '-filter_complex_threads', '2', '-lavfi',
        'ssim=stats_file=' + str(stats), '-an', '-f', 'null', '-'], check=True, capture_output=True, text=True)
    (verification_dir / 'ssim.log').write_text(result.stderr)
    match = re.search(r'\bAll:([0-9.]+)', result.stderr)
    if not match or float(match[1]) < .99:
        raise ValueError('The full-frame SSIM quality check is below 0.99')
    frame_stats = stats.read_text().splitlines()
    if len(frame_stats) != int(encoded_video['nb_read_frames']):
        raise ValueError('SSIM does not cover all video frames')
    return dict(source_metadata=before, output_metadata=after, audio_payload_sha256=output_audio,
                audio_payload_identical=True, complete_frame_ssim=float(match[1]),
                ssim_frames_checked=len(frame_stats), faststart=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Retained original master, never the compressed output')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--ffmpeg', default=shutil.which('ffmpeg'))
    parser.add_argument('--verify-existing', action='store_true', help='Audit an already encoded candidate without re-encoding')
    parser.add_argument('--verification-dir', type=Path, default=ROOT / 'ignore/validation/website_media_20261030')
    args = parser.parse_args()
    if not args.ffmpeg:
        parser.error('Pass --ffmpeg or install the maintainer-only FFmpeg tool')
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or not source.is_file():
        parser.error('Source must be a separate, existing master file')
    source_sha = digest(source)
    command = [args.ffmpeg, '-hide_banner', '-nostdin', '-nostats', '-y', '-i', str(source),
               '-map', '0:v:0', '-map', '0:a:0', '-c:v', 'libx264', '-preset', 'slow', '-crf', '24',
               '-profile:v', 'high', '-level:v', '4.1', '-threads', '4', '-pix_fmt', 'yuv420p',
               '-c:a', 'copy', '-movflags', '+faststart', str(output)]
    if not args.verify_existing:
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True)
    verified = verify(args.ffmpeg, source, output, args.verification_dir)
    if digest(source) != source_sha:
        raise ValueError('Master changed during encoding/verification')
    record = dict(sha256=digest(output), source_sha256=source_sha,
                  source_bytes=source.stat().st_size, output_bytes=output.stat().st_size,
                  reduction_percent=round(100 * (1 - output.stat().st_size / source.stat().st_size), 2),
                  derivative='H.264 High, CRF 24, 1080p/30fps retained, original AAC copied', **verified)
    (args.verification_dir / 'video_optimization.json').write_text(json.dumps(record, indent=2) + '\n')
    if output == OUTPUT:
        receipt_path = ROOT / 'docs/website/published_assets.json'
        receipt = json.loads(receipt_path.read_text())
        receipt['files'][output.relative_to(ROOT).as_posix()] = record
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', **record), indent=2))


if __name__ == '__main__':
    main()

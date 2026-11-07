"""Write DWI.nhdr for one subject with the plugin's own nhdr_write, as ensure_diffusion_headers does.

Usage: PythonSlicer make_dwi_nhdr.py <dwi.nii.gz> <bval> <bvec> <out_dir>
The detached header points to the absolute payload path, so the header can move.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "anatomy/parcellation/conversion"))
from nhdr_write import nhdr_write  # noqa: E402

dwi, bval, bvec, out_dir = map(Path, sys.argv[1:5])
out_dir.mkdir(parents=True, exist_ok=True)
target = out_dir / "DWI.nhdr"
nhdr_write(str(dwi), str(bval), str(bvec), str(target))
scaled = target.with_suffix(".raw.gz")
payload = scaled if scaled.is_file() else dwi
lines = [f"data file: {payload.resolve().as_posix()}" if line.startswith("data file:") else line
         for line in target.read_text().splitlines()]
target.write_text("\n".join(lines) + "\n")
print(target.read_text()[:1500])

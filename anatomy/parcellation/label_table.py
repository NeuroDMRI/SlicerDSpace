"""Names and colours of parcellation label ids, from the bundled FreeSurferColorLUT.txt.

Pure Python (numpy and nibabel only to read the label ids), so the pipeline writes the label tables outside
Slicer too. The label NIfTI is never changed; the tables are written beside it.
"""

from __future__ import annotations

import colorsys
import os
from pathlib import Path
from typing import Iterable, NamedTuple, Optional

LUT_PATH = Path(__file__).resolve().parent / "conversion" / "data" / "FreeSurferColorLUT.txt"
_GOLDEN = 0.618033988749895


class LabelEntry(NamedTuple):
    label: int
    name: str
    rgb: tuple[int, int, int]
    freesurfer_a: int  # FreeSurfer's fourth column: 0 means opaque
    generated: bool  # True when the id has no entry in the LUT


def read_freesurfer_lut(path: Path | str = LUT_PATH) -> dict[int, tuple[str, tuple[int, int, int, int]]]:
    """``{id: (name, (r, g, b, a))}`` from a FreeSurfer-format colour table; the first entry of an id wins."""

    entries: dict[int, tuple[str, tuple[int, int, int, int]]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split("#", 1)[0].split()
        if len(fields) < 6:
            continue
        try:
            label, rgba = int(fields[0]), tuple(int(value) for value in fields[2:6])
        except ValueError:
            continue
        entries.setdefault(label, (fields[1], rgba))  # type: ignore[arg-type]
    return entries


def generated_color(label: int) -> tuple[int, int, int]:
    """A fixed, well-separated colour for an id with no LUT entry (golden-ratio hue steps)."""

    hue = (int(label) * _GOLDEN) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.6, 0.9)
    return int(round(red * 255)), int(round(green * 255)), int(round(blue * 255))


def label_entries(labels: Iterable[int], lut: Optional[dict] = None) -> list[LabelEntry]:
    """One entry per id (and always 0), sorted. An id with no LUT entry gets ``Label_<id>`` and a generated colour."""

    table = read_freesurfer_lut() if lut is None else lut
    entries = []
    for label in sorted({int(value) for value in labels} | {0}):
        if label in table:
            name, (red, green, blue, alpha) = table[label]
            entries.append(LabelEntry(label, name, (red, green, blue), alpha, False))
        else:
            entries.append(LabelEntry(label, "Label_" + str(label), generated_color(label), 0, True))
    return entries


def present_labels(label_path: Path | str) -> list[int]:
    """The distinct ids stored in a label volume; non-integer values are an error, not rounded."""

    import nibabel as nib
    import numpy as np

    values = np.unique(np.asarray(nib.load(str(label_path)).dataobj))
    if values.size and not np.all(np.isfinite(values) & (values == np.round(values))):
        raise ValueError(f"Label volume holds non-integer values: {label_path}")
    return [int(value) for value in values]


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_label_tables(label_path: Path | str, ctbl_path: Path | str, freesurfer_path: Path | str) -> list[int]:
    """Write a Slicer ``.ctbl`` and a FreeSurfer-format ``.txt`` with exactly the ids present (plus 0).

    The ``.ctbl`` alpha is 0 for id 0 and 255 for every label; the FreeSurfer table keeps FreeSurfer's own fourth
    column. Returns the ids that had no LUT entry (named ``Label_<id>``).
    """

    entries = label_entries(present_labels(label_path))
    source = Path(label_path).name
    count = max(entry.label for entry in entries) + 1
    ctbl = [f"# Color table file {Path(ctbl_path).name}", f"# {count} values",
            f"# Labels present in {source}; names and colours from FreeSurferColorLUT.txt"]
    freesurfer = [f"# Labels present in {source}; names and colours from FreeSurferColorLUT.txt",
                  "#No. Label Name:                            R   G   B   A"]
    for entry in entries:
        red, green, blue = entry.rgb
        alpha = 0 if entry.label == 0 else 255
        ctbl.append(f"{entry.label} {entry.name} {red} {green} {blue} {alpha}")
        freesurfer.append(f"{entry.label:<5d} {entry.name:<40s} {red:3d} {green:3d} {blue:3d} {entry.freesurfer_a:3d}")
    _write_text(Path(ctbl_path), "\n".join(ctbl) + "\n")
    _write_text(Path(freesurfer_path), "\n".join(freesurfer) + "\n")
    return [entry.label for entry in entries if entry.generated]

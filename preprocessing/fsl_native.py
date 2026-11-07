"""Run the MRtrix 3.0.8 correction stages used by DSPace without a shell."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import math
from pathlib import Path

from shared.pe_resolution import DWIFSLPREPROC_DEFAULT_READOUT_S


_FSL_STRIDES_3D = "-1,+2,+3"
_FSL_STRIDES_4D = "-1,+2,+3,+4"
_PE_DIRECTIONS = {
    "LR": (1, 0, 0), "RL": (-1, 0, 0),
    "PA": (0, 1, 0), "AP": (0, -1, 0),
    "IS": (0, 0, 1), "SI": (0, 0, -1),
}
# MRtrix 3.0.8's default BZeroThreshold, in s/mm².
_BZERO_THRESHOLD = 10.0


def _require_file(path: Path) -> None:
    if not path.is_file() or not path.stat().st_size:
        raise RuntimeError(f"FSL correction did not produce a nonempty file: {path}")


def _fsl_image(prefix: Path) -> Path:
    for extension in (".nii.gz", ".nii"):
        path = prefix.with_name(prefix.name + extension)
        if path.is_file() and path.stat().st_size:
            return path
    raise RuntimeError(f"FSL correction did not produce a NIfTI image: {prefix}.nii[.gz]")


def _write_pe_table(path: Path, direction: tuple[int, ...], readout: float, volumes: int, *, paired=False) -> None:
    def row(sign):
        return " ".join([*(str(sign * axis) for axis in direction), str(readout)]) + "\n"

    if paired:
        half = volumes // 2
        text = row(1) * half + row(-1) * half
    else:
        text = row(1) * volumes
    path.write_text(text, encoding="ascii")


def _same_grid(first: dict, second: dict) -> bool:
    if first["size"][:3] != second["size"][:3]:
        return False
    for one, two in zip(first["spacing"][:3], second["spacing"][:3]):
        if one and two and not math.isnan(one) and not math.isnan(two):
            if abs(two - one) / (0.5 * (one + two)) > 1e-4:
                return False
    for one, two in zip(first["transform"][:3], second["transform"][:3]):
        if any(abs(a - b) > 1e-4 for a, b in zip(one[:3], two[:3])) or abs(one[3] - two[3]) > 1e-2:
            return False
    return True


def _validate_gradients(bvec: Path, bval: Path, volumes: int) -> None:
    _require_file(bvec)
    _require_file(bval)
    try:
        vectors = [[float(value) for value in line.split()]
                   for line in bvec.read_text(encoding="ascii").splitlines() if line.strip()]
        values = [float(value) for value in bval.read_text(encoding="ascii").split()]
        if (len(vectors) != 3 or any(len(row) != volumes for row in vectors) or len(values) != volumes
                or not all(math.isfinite(value) for row in vectors for value in row)
                or not all(math.isfinite(value) for value in values)):
            raise ValueError("shape or numeric values do not match DWI volumes")
    except ValueError as exc:
        raise RuntimeError(f"Invalid FSL gradient tables: {bvec}, {bval}") from exc


def run_fsl_correction(
    *,
    input_dwi: Path,
    input_bvec: Path,
    input_bval: Path,
    output_dwi: Path,
    output_bvec: Path,
    output_bval: Path,
    scratch_dir: Path,
    phase_encoding: str,
    readout_time: float | None,
    se_epi_pair: Path | None,
    commands: Mapping[str, str],
    eddy_options: Sequence[str],
    run_stage: Callable[[list[str]], str],
    topup_config: Path | None = None,
    force: bool = False,
) -> None:
    """Correct a single-PE DWI with an optional balanced forward/reverse pair.

    The caller resolves the absolute binaries and configures the FSL runtime.
    The callback runs one argv and raises on nonzero exit. MRtrix performs the
    FSL phase and gradient conversions in the image's actual on-disk strides.
    """
    direction = _PE_DIRECTIONS.get(phase_encoding.upper())
    if direction is None:
        raise ValueError(f"Unsupported anatomical phase encoding direction: {phase_encoding}")
    readout = DWIFSLPREPROC_DEFAULT_READOUT_S if readout_time is None else float(readout_time)
    if not math.isfinite(readout) or readout <= 0:
        raise ValueError("FSL readout time must be positive and finite.")
    use_topup = se_epi_pair is not None
    required = ["mrconvert", "mrinfo", "dwi2mask", "maskfilter", "eddy"]
    if use_topup:
        required.extend(("mrcat", "mrtransform", "mrcalc", "topup", "applytopup"))
        if topup_config is None or not topup_config.is_file():
            raise FileNotFoundError(f"FSL Topup configuration not found: {topup_config}")
    for name in required:
        if not commands.get(name) or not Path(commands[name]).is_absolute():
            raise ValueError(f"FSL correction requires an absolute executable path for {name}.")
    outputs = (output_dwi, output_bvec, output_bval)
    if not force:
        for output in outputs:
            if output.exists():
                raise FileExistsError(output)

    scratch = scratch_dir.resolve()
    scratch.mkdir(parents=True)

    def call(tool: str, *args: str | Path) -> str:
        return run_stage([str(commands[tool]), *(str(arg) for arg in args)])

    def header(image: Path, name: str) -> dict:
        info = scratch / name
        call("mrinfo", image, "-json_all", info)
        _require_file(info)
        return json.loads(info.read_text(encoding="utf-8"))

    def pad_last_slice(image: Path, stem: str, axis: int, length: int, *, clear_gradients=False) -> Path:
        last_slice = scratch / f"{stem}_slice{axis}.mif"
        padded = scratch / f"{stem}_pad{axis}.mif"
        clear = ("-clear", "dw_scheme") if clear_gradients else ()
        call("mrconvert", image, last_slice, "-coord", str(axis), str(length - 1), *clear)
        call("mrcat", image, last_slice, padded, "-axis", str(axis))
        _require_file(padded)
        return padded

    dwi = scratch / "dwi.mif"
    call("mrconvert", input_dwi, dwi, "-fslgrad", input_bvec, input_bval)
    _require_file(dwi)
    dwi_header = header(dwi, "dwi_header.json")
    shape = dwi_header["size"]
    if len(shape) != 4 or any(size <= 1 for size in shape[:3]) or shape[3] < 1:
        raise ValueError(f"FSL correction requires a 4D DWI on a nondegenerate 3D grid: {shape}")
    volume_count = shape[3]
    gradients = dwi_header.get("keyval", {}).get("dw_scheme", [])
    if len(gradients) != volume_count or any(len(gradient) != 4 for gradient in gradients):
        raise ValueError("DWI gradient table does not match input volumes.")
    first_bzero = next((index for index, grad in enumerate(gradients) if grad[3] <= _BZERO_THRESHOLD), 0)
    before_eddy: list[str] = []
    after_eddy: list[str] = []
    if first_bzero:
        permutation = [first_bzero, *range(first_bzero), *range(first_bzero + 1, volume_count)]
        inverse_permutation = [0] * volume_count
        for index, original in enumerate(permutation):
            inverse_permutation[original] = index
        before_eddy.extend(("-coord", "3", ",".join(map(str, permutation))))
        after_eddy.extend(("-coord", "3", ",".join(map(str, inverse_permutation))))

    dwi_pe = scratch / "dwi_manual_pe_scheme.txt"
    _write_pe_table(dwi_pe, direction, readout, volume_count)
    mask_source = dwi
    eddy_topup: list[str] = []
    if use_topup:
        pair_header = header(se_epi_pair, "se_epi_header.json")
        pair_shape = pair_header["size"]
        if len(pair_shape) != 4 or pair_shape[3] < 2 or pair_shape[3] % 2:
            raise ValueError("Topup requires a 4D SE-EPI pair with equally many forward and reverse volumes.")
        pair = se_epi_pair
        if not _same_grid(dwi_header, pair_header):
            regridded = scratch / "se_epi_regrid_sinc.mif"
            call("mrtransform", pair, regridded, "-reorient_fod", "no", "-interp", "sinc", "-template", dwi)
            pair = scratch / "se_epi_regrid.mif"
            call("mrcalc", regridded, "0.0", "-max", pair)
            _require_file(pair)

        for axis, length in enumerate(shape[:3]):
            if length % 2:
                pair = pad_last_slice(pair, "se_epi", axis, length)
                dwi = pad_last_slice(dwi, "dwi", axis, length, clear_gradients=True)
                after_eddy.extend(("-coord", str(axis), f"0:{length - 1}"))

        pair_pe = scratch / "se_epi_manual_pe_scheme.txt"
        _write_pe_table(pair_pe, direction, readout, pair_shape[3], paired=True)
        topup_image = scratch / "topup_in.nii"
        topup_datain = scratch / "topup_datain.txt"
        call("mrconvert", pair, topup_image, "-import_pe_table", pair_pe,
             "-strides", _FSL_STRIDES_4D, "-export_pe_topup", topup_datain)
        _require_file(topup_image)
        _require_file(topup_datain)
        field = scratch / "field"
        field_map = scratch / "field_map.nii.gz"
        topup_stdout = call(
            "topup", f"--imain={topup_image}", f"--datain={topup_datain}", f"--out={field}",
            f"--fout={field_map}", f"--config={topup_config}", f"--logout={scratch / 'topup_in.topup_log'}",
            "--verbose",
        )
        (scratch / "topup_output.txt").write_text(topup_stdout or "", encoding="utf-8")
        _fsl_image(scratch / "field_fieldcoef")
        _require_file(scratch / "field_movpar.txt")
        _fsl_image(scratch / "field_map")

        # These two acquisition modes have only one PE setting for the DWIs.
        # Applytopup is used only to prepare the mask. Eddy corrects the
        # unwarped input once using the Topup coefficients.
        apply_input = scratch / "applytopup_in.nii"
        apply_bvec, apply_bval = scratch / "applytopup_in.bvec", scratch / "applytopup_in.bval"
        apply_config, apply_index = scratch / "applytopup_config.txt", scratch / "applytopup_indices.txt"
        call("mrconvert", dwi, apply_input, "-import_pe_table", dwi_pe,
             "-strides", _FSL_STRIDES_4D, "-export_pe_eddy", apply_config, apply_index,
             "-export_grad_fsl", apply_bvec, apply_bval)
        _require_file(apply_input)
        _require_file(apply_config)
        _validate_gradients(apply_bvec, apply_bval, volume_count)
        apply_prefix = scratch / "applytopup_corrected"
        # FSL's TopupFileReader only recognises '/' as a directory separator.
        topup_option = f"--topup={field.as_posix()}"
        call("applytopup", f"--imain={apply_input}", f"--datain={apply_config}", "--inindex=1",
             topup_option, f"--out={apply_prefix}", "--method=jac")
        mask_source = scratch / "applytopup_corrected.mif"
        call("mrconvert", _fsl_image(apply_prefix), mask_source, "-fslgrad", apply_bvec, apply_bval)
        _require_file(mask_source)
        eddy_topup.append(topup_option)

    mask_raw = scratch / "eddy_mask_raw.mif"
    mask_dilated = scratch / "eddy_mask_dilated.mif"
    mask = scratch / "eddy_mask.nii"
    call("dwi2mask", mask_source, mask_raw)
    call("maskfilter", mask_raw, "dilate", mask_dilated)
    call("mrconvert", mask_dilated, mask, "-datatype", "float32", "-strides", _FSL_STRIDES_3D)
    _require_file(mask)

    eddy_input = scratch / "eddy_in.nii"
    eddy_bvec, eddy_bval = scratch / "bvecs", scratch / "bvals"
    eddy_config, eddy_index = scratch / "eddy_config.txt", scratch / "eddy_indices.txt"
    call("mrconvert", dwi, eddy_input, "-import_pe_table", dwi_pe, *before_eddy,
         "-strides", _FSL_STRIDES_4D, "-export_grad_fsl", eddy_bvec, eddy_bval,
         "-export_pe_eddy", eddy_config, eddy_index)
    for path in (eddy_input, eddy_config, eddy_index):
        _require_file(path)
    _validate_gradients(eddy_bvec, eddy_bval, volume_count)
    if eddy_index.read_text(encoding="ascii").split() != ["1"] * volume_count:
        raise RuntimeError(f"Eddy index table does not match the single-PE DWI: {eddy_index}")

    eddy_prefix = scratch / "dwi_post_eddy"
    eddy_stdout = call(
        "eddy", f"--imain={eddy_input}", f"--mask={mask}", f"--acqp={eddy_config}", f"--index={eddy_index}",
        f"--bvecs={eddy_bvec}", f"--bvals={eddy_bval}", *eddy_topup, *eddy_options,
        f"--out={eddy_prefix}", "--verbose",
    )
    (scratch / "eddy_output.txt").write_text(eddy_stdout or "", encoding="utf-8")
    eddy_output = _fsl_image(eddy_prefix)
    rotated_bvec = scratch / "dwi_post_eddy.eddy_rotated_bvecs"
    _validate_gradients(rotated_bvec, eddy_bval, volume_count)

    result = scratch / "result.mif"
    restored_strides = ",".join(map(str, dwi_header["strides"]))
    call("mrconvert", eddy_output, result, *after_eddy, "-strides", restored_strides,
         "-fslgrad", rotated_bvec, eddy_bval)
    _require_file(result)
    result_header = header(result, "result_header.json")
    if result_header["size"] != shape or not _same_grid(dwi_header, result_header):
        raise RuntimeError(f"Corrected DWI does not match the input grid and volume count: {result}")

    overwrite = ("-force",) if force else ()
    call("mrconvert", result, output_dwi, "-export_grad_fsl", output_bvec, output_bval, *overwrite)
    _require_file(output_dwi)
    _validate_gradients(output_bvec, output_bval, volume_count)

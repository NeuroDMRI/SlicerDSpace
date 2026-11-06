"""Native PNL CNN brain mask in Slicer's Python: SimpleITK rigid registration and the vendored CNN.

T2 design A1-A6. Run as one subprocess ``[PythonSlicer, shared/pnl_mask_native.py, ...]``; torch and
SimpleITK are imported only inside functions, so importing this module stays cheap. ANTs is not used:
A3 replaces ``antsRegistrationSyNQuick.sh -t r`` and A4 replaces ``antsApplyTransforms`` (inverse
rigid, linear) and ``ImageMath FillHoles``. Normalisation, three-view inference, fusion weights and the
0.45 threshold are the vendored package code (``shared/pnl_cnn``), unchanged.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

ROUTE = "native-sitk"
TEMPLATE_NAME = "IITmean_b0_256.nii.gz"
VIEWS = ("sagittal", "coronal", "axial")


@dataclass(frozen=True)
class RegistrationParameters:
    """A3: one ANTs ``Rigid[0.1]`` stage with ``MI[f,m,1,32,Regular,0.25]``, run as three single-level registrations."""

    histogram_bins: int = 32
    sampling_percentage: float = 0.25
    sampling_seed: int = 1
    learning_rate: float = 0.1
    maximum_step_physical: float = 0.1
    line_search_lower_limit: float = 0.0
    line_search_upper_limit: float = 2.0
    line_search_epsilon: float = 0.2
    line_search_iterations: int = 20
    convergence_minimum: float = 1e-6
    convergence_window: int = 10
    shrink_factors: tuple = (8, 4, 2)
    smoothing_sigmas_voxels: tuple = (3.0, 2.0, 1.0)
    iterations: tuple = (1000, 500, 250)
    # ImageRegistrationMethod constant name; "Once" is the second M-A T3 fallback change (design §3 M-A).
    learning_rate_estimation: str = "EachIteration"
    # ANTs passes the winsorise quantiles as C++ float (helper.hxx:80); Quantile takes them as double.
    winsorize_lower: float = float(np.float32(0.005))
    winsorize_upper: float = float(np.float32(0.995))
    winsorize_bins: int = 256
    winsorize_marginal_scale: float = 10.0


PARAMETERS = RegistrationParameters()


def _vendor(name):
    """Vendored ``pnl_cnn`` module, without importing the ``shared`` package when run as a script."""
    if __package__:
        return importlib.import_module(f"{__package__}.pnl_cnn.{name}")
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    return importlib.import_module(f"pnl_cnn.{name}")


def default_threads() -> int:
    """A6: ``SLICERDSPACE_STAGE_CPUS``, else the process CPU affinity, else ``os.cpu_count()`` (as teem_dti)."""
    requested = os.environ.get("SLICERDSPACE_STAGE_CPUS", "").strip()
    if requested.isdigit() and int(requested) > 0:
        return int(requested)
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def apply_threads(threads: int) -> int:
    import SimpleITK as sitk
    import torch

    torch.set_num_threads(threads)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(threads)
    return threads


# --------------------------------------------------------------------------- arrays (A3 image input)
def to_numpy(image) -> np.ndarray:
    """SimpleITK image -> array indexed [i, j, k] like nibabel ``get_fdata()``."""
    import SimpleITK as sitk

    return sitk.GetArrayFromImage(image).transpose(2, 1, 0)


def to_sitk(array: np.ndarray, reference):
    """Array indexed [i, j, k] -> SimpleITK image with the geometry of ``reference``."""
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(np.ascontiguousarray(np.asarray(array).transpose(2, 1, 0)))
    image.CopyInformation(reference)
    return image


# --------------------------------------------------------------------------- winsorise (A3, ITK histogram rule)
def histogram_bins(lower: float, upper: float, size: int = 256):
    """Bin limits as ``itk::Statistics::Histogram::Initialize``: float32 interval and products, double sums."""
    interval = (np.float32(upper) - np.float32(lower)) / np.float32(size)
    index = np.arange(size, dtype=np.float32)
    mins = np.float64(lower) + (index * interval).astype(np.float64)
    maxs = np.float64(lower) + ((index + np.float32(1)) * interval).astype(np.float64)
    maxs[size - 1] = np.float64(upper)
    return mins, maxs


def histogram_index(values: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    """Bin index as ``Histogram::GetIndex`` (ClipBinsAtEnds on); -1 marks a value outside the histogram."""
    values = np.asarray(values, dtype=np.float64)
    size = len(mins)
    index = np.full(values.shape, -1, dtype=np.int64)
    # x >= the last maximum can only be x == maximum here (the maximum is the image maximum plus a
    # non-negative margin), and AlmostEquals then puts it in the last bin.
    index[values >= maxs[-1]] = size - 1
    active = (values >= mins[0]) & (values < maxs[-1])
    begin = np.zeros(values.shape, dtype=np.int64)
    end = np.full(values.shape, size - 1, dtype=np.int64)
    mid = (end + 1) // 2
    for _ in range(4 * size.bit_length() + 8):
        if not active.any():
            break
        median = mins[np.clip(mid, 0, size - 1)]
        lower = active & (values < median)
        greater = active & (values > median)
        equal = active & (values == median)
        inside = greater & (values < maxs[np.clip(mid, 0, size - 1)]) & (values >= median)
        found = equal | inside
        index[found] = mid[found]
        active &= ~found
        end = np.where(lower, mid - 1, end)
        begin = np.where(greater & ~inside, mid + 1, begin)
        # C++ integer division truncates toward zero.
        mid = np.where(active, begin + np.trunc((end - begin) / 2).astype(np.int64), mid)
    if active.any():
        raise RuntimeError("Histogram bin search did not converge")
    return index


def histogram_quantile(counts: np.ndarray, mins: np.ndarray, maxs: np.ndarray, p: float) -> float:
    """``Histogram::Quantile`` (itkHistogram.hxx), both branches, in double precision."""
    size = len(counts)
    total = float(np.sum(counts, dtype=np.float64))
    cumulated = 0.0
    if p < 0.5:
        n, p_n = 0, 0.0
        while True:
            f_n = float(counts[n])
            cumulated += f_n
            p_n_prev, p_n = p_n, cumulated / total
            n += 1
            if not (n < size and p_n < p):
                break
        proportion = f_n / total
        low, high = float(mins[n - 1]), float(maxs[n - 1])
        return low + ((p - p_n_prev) / proportion) * (high - low)
    n, m, p_n = size - 1, 0, 1.0
    while True:
        f_n = float(counts[n])
        cumulated += f_n
        p_n_prev, p_n = p_n, 1.0 - cumulated / total
        n -= 1
        m += 1
        if not (m < size and p_n > p):
            break
    proportion = f_n / total
    low, high = float(mins[n + 1]), float(maxs[n + 1])
    return high - ((p_n_prev - p) / proportion) * (high - low)


def winsorize_window(array: np.ndarray, parameters: RegistrationParameters = PARAMETERS) -> tuple[float, float]:
    """Window bounds of ANTs ``PreprocessImage``: 256-bin histogram over all voxels, float32 quantiles."""
    values, counts = np.unique(np.asarray(array, dtype=np.float64), return_counts=True)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Winsorise input must be finite and non-empty")
    low, high = float(values[0]), float(values[-1])
    if high == low:
        # SPEC-GAP: a constant image gives an empty intensity window (ANTs would divide by zero); refuse it.
        raise ValueError("Registration input is constant; no intensity window exists")
    size = parameters.winsorize_bins
    margin = ((high - low) / float(size)) / float(parameters.winsorize_marginal_scale)
    if np.finfo(np.float64).max - high > margin:
        high = high + margin
    mins, maxs = histogram_bins(low, high, size)
    index = histogram_index(values, mins, maxs)
    kept = index >= 0
    histogram = np.bincount(index[kept], weights=counts[kept], minlength=size)
    lower = histogram_quantile(histogram, mins, maxs, parameters.winsorize_lower)
    upper = histogram_quantile(histogram, mins, maxs, parameters.winsorize_upper)
    return float(np.float32(lower)), float(np.float32(upper))


def winsorize(image, parameters: RegistrationParameters = PARAMETERS):
    import SimpleITK as sitk

    lower, upper = winsorize_window(to_numpy(image), parameters)
    return sitk.IntensityWindowing(image, lower, upper, 0.0, 1.0), (lower, upper)


# --------------------------------------------------------------------------- registration (A3)
def configure_level(method, parameters: RegistrationParameters, level: int) -> None:
    """Set metric, sampling, optimizer and pyramid for one single-level registration."""
    import SimpleITK as sitk

    method.SetMetricAsMattesMutualInformation(parameters.histogram_bins)
    method.SetMetricSamplingStrategy(method.REGULAR)
    # ANTs runs one multi-level method whose sampling seed advances one step per level (ITK
    # itkImageRegistrationMethodv4.hxx), so level k uses seed + k (finding A-O1).
    method.SetMetricSamplingPercentage(parameters.sampling_percentage, parameters.sampling_seed + level)
    method.MetricUseFixedImageGradientFilterOff()
    method.MetricUseMovingImageGradientFilterOff()
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsConjugateGradientLineSearch(
        learningRate=parameters.learning_rate,
        numberOfIterations=parameters.iterations[level],
        convergenceMinimumValue=parameters.convergence_minimum,
        convergenceWindowSize=parameters.convergence_window,
        lineSearchLowerLimit=parameters.line_search_lower_limit,
        lineSearchUpperLimit=parameters.line_search_upper_limit,
        lineSearchEpsilon=parameters.line_search_epsilon,
        lineSearchMaximumIterations=parameters.line_search_iterations,
        estimateLearningRate=getattr(method, parameters.learning_rate_estimation),
        maximumStepSizeInPhysicalUnits=parameters.maximum_step_physical)
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel([parameters.shrink_factors[level]])
    method.SetSmoothingSigmasPerLevel([parameters.smoothing_sigmas_voxels[level]])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()


def register(fixed, moving, parameters: RegistrationParameters = PARAMETERS, log=None):
    """Rigid fixed(template)->moving(b0) transform; returns (final composite, initial, optimised Euler, record)."""
    import SimpleITK as sitk

    fixed_metric, fixed_window = winsorize(fixed, parameters)
    moving_metric, moving_window = winsorize(moving, parameters)
    initial = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
                                                sitk.CenteredTransformInitializerFilter.MOMENTS)
    current = sitk.Euler3DTransform()
    levels = []
    for level in range(len(parameters.shrink_factors)):
        started = time.monotonic()
        method = sitk.ImageRegistrationMethod()
        configure_level(method, parameters, level)
        method.SetMovingInitialTransform(initial)
        method.SetInitialTransform(current, inPlace=False)
        result = method.Execute(fixed_metric, moving_metric)
        composite = result if isinstance(result, sitk.CompositeTransform) else sitk.CompositeTransform(result)
        current = sitk.Euler3DTransform(composite.GetNthTransform(composite.GetNumberOfTransforms() - 1))
        levels.append({"seed": parameters.sampling_seed + level,  # A-O1: ANTs' seed advances one step per level
                       "shrink": parameters.shrink_factors[level],
                       "sigma_voxels": parameters.smoothing_sigmas_voxels[level],
                       "iterations": method.GetOptimizerIteration(),
                       "metric": method.GetMetricValue(),
                       "stop": method.GetOptimizerStopConditionDescription(),
                       "seconds": time.monotonic() - started})
        if log:
            log(f"[PNL native] level {level + 1}: {levels[-1]['iterations']} iterations, metric {levels[-1]['metric']:.6f}")
    # ITK applies a moving initial transform after the optimised one: x -> initial(current(x)).
    final = sitk.CompositeTransform([initial, current])
    record = {"winsorize_window": {"fixed": fixed_window, "moving": moving_window},
              "initial": {"parameters": list(initial.GetParameters()), "fixed": list(initial.GetFixedParameters())},
              "optimised": {"parameters": list(current.GetParameters()), "fixed": list(current.GetFixedParameters())},
              "levels": levels}
    return final, initial, current, record


def warp(moving, fixed, transform):
    """A3 Warped image: raw float64 b0 on the template grid, linear, default 0."""
    import SimpleITK as sitk

    return sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat64)


# --------------------------------------------------------------------------- inverse and hole filling (A4)
def inverse_mask(template_mask: np.ndarray, fixed, moving, transform) -> np.ndarray:
    """Template-grid binary mask -> float32 on the b0 grid by the inverse transform, linear, default 0."""
    import SimpleITK as sitk

    image = to_sitk(np.asarray(template_mask, dtype=np.float32), fixed)
    resampled = sitk.Resample(image, moving, transform.GetInverse(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    return to_numpy(resampled)


def fill_holes(values: np.ndarray) -> np.ndarray:
    """ANTs ``ImageMath FillHoles 2`` rule: fill every background component except the largest.

    Object = value >= 0.5; background components use face connectivity. A tie for the largest keeps
    the component that comes first in ITK raster order (x fastest), as ITK's relabel order does.
    """
    from scipy import ndimage

    values = np.asarray(values, dtype=np.float32)
    labels, count = ndimage.label(~(values >= 0.5))
    filled = values.copy()
    if count > 1:
        raster = labels.transpose(2, 1, 0).ravel()
        found, first = np.unique(raster, return_index=True)
        first_by_label = dict(zip(found.tolist(), first.tolist()))
        sizes = np.bincount(raster, minlength=count + 1)
        keep = min(range(1, count + 1), key=lambda label: (-int(sizes[label]), first_by_label[label]))
        filled[(labels > 0) & (labels != keep)] = 1
    return filled


def finish_mask(filled: np.ndarray) -> np.ndarray:
    """``astype(uint8)`` truncation and the package's binary/non-empty check."""
    values = np.asarray(filled, dtype=np.float32).astype(np.uint8)
    if not np.isin(values, [0, 1]).all() or not values.any():
        raise RuntimeError("Native output must be binary and nonempty")
    return values


def write_mask(values: np.ndarray, baseline_image, output: Path) -> None:
    """int16 NIfTI with the b0 header (package ``finish_native``), staged then renamed."""
    import nibabel as nib

    header = baseline_image.header.copy()
    header.set_data_dtype(np.int16)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".pnl-native-", suffix=".nii.gz", delete=False) as stream:
        staged = Path(stream.name)
    try:
        nib.save(nib.Nifti1Image(values, baseline_image.affine, header), str(staged))
        result = nib.load(str(staged))
        if result.shape != baseline_image.shape[:3] or not np.allclose(result.affine, baseline_image.affine,
                                                                       rtol=0, atol=1e-4):
            raise RuntimeError("Native geometry changed")
        staged.replace(output)
    finally:
        staged.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- pipeline
def run(baseline: Path, output: Path, model_dir: Path, template: Path, *, device: str = "auto",
        threads: int | None = None, keep_dir: Path | None = None,
        parameters: RegistrationParameters = PARAMETERS, percentile: float = 99, log=None) -> dict:
    """Baseline mean-b0 file (A5) -> native-grid int16 mask; returns the receipt."""
    import nibabel as nib
    import SimpleITK as sitk

    log = log or (lambda message: print(message, flush=True))

    baseline, output, model_dir, template = map(lambda p: Path(p).expanduser().absolute(),
                                                (baseline, output, model_dir, template))
    protected = [baseline, template, model_dir / "weights.json", *(model_dir / f"{v}.pt" for v in VIEWS)]
    if output.is_symlink() or any(output.resolve() == path.resolve() for path in protected):
        raise ValueError("Output must not overwrite an input or model resource or follow a symlink")
    if not output.name.endswith(".nii.gz"):
        raise ValueError("Output must have the .nii.gz suffix")
    weights = _vendor("weights")
    weights.check(template, weights.TEMPLATE_SHA256)
    inference = _vendor("inference")
    threads = threads if threads is not None else default_threads()
    if threads < 1:
        raise ValueError("Threads must be positive")
    apply_threads(threads)
    selection = inference.select_device(device)
    log(f"[PNL native] device {selection['resolved']}, {threads} threads")
    phases = {}

    started = time.monotonic()
    image = nib.load(str(baseline))
    moving_values = _vendor("pipeline").mean_b0(image)
    raw = sitk.ReadImage(str(baseline), sitk.sitkFloat64)
    # O4 contract: ITK and nibabel read the same voxels in the same [i, j, k] order.
    if raw.GetDimension() != 3 or not np.array_equal(to_numpy(raw), image.get_fdata()):
        raise RuntimeError("SimpleITK and nibabel disagree on the baseline voxels")
    moving = to_sitk(moving_values, raw)
    fixed = sitk.ReadImage(str(template), sitk.sitkFloat64)
    phases["read_inputs"] = time.monotonic() - started

    started = time.monotonic()
    log("[PNL native] rigid registration to the IIT template")
    transform, _, _, registration = register(fixed, moving, parameters, log=log)
    warped = warp(moving, fixed, transform)
    phases["registration"] = time.monotonic() - started

    started = time.monotonic()
    data = inference.normalize(to_numpy(warped), percentile)
    _, _, template_mask = inference.predict(data, model_dir, device=selection["resolved"], batch_size=1)
    phases["inference"] = time.monotonic() - started

    started = time.monotonic()
    log("[PNL native] inverse resample and hole filling")
    inverse = inverse_mask(template_mask, fixed, moving, transform)
    filled = fill_holes(inverse)
    values = finish_mask(filled)
    phases["inverse_and_fill"] = time.monotonic() - started

    started = time.monotonic()
    write_mask(values, image, output)
    if keep_dir is not None:
        keep_dir = Path(keep_dir)
        keep_dir.mkdir(parents=True, exist_ok=True)
        sitk.WriteTransform(transform, str(keep_dir / "transform.tfm"))
        sitk.WriteImage(warped, str(keep_dir / "Warped.nii.gz"))
        sitk.WriteImage(to_sitk(np.asarray(template_mask, dtype=np.float32), fixed), str(keep_dir / "template_mask.nii.gz"))
        sitk.WriteImage(to_sitk(inverse, moving), str(keep_dir / "inverse.nii.gz"))
        sitk.WriteImage(to_sitk(filled, moving), str(keep_dir / "filled.nii.gz"))
    phases["write"] = time.monotonic() - started

    import torch

    return {**selection, "route": ROUTE, "status": "completed", "device": selection["resolved"],
            "threads": threads, "simpleitk_version": sitk.Version_VersionString(), "torch_version": torch.__version__,
            "numpy_version": np.__version__, "python": sys.executable,
            "vendored_package_version": importlib.import_module(weights.__package__).__version__,
            "source_commit": weights.SOURCE_COMMIT, "input_sha256": _sha256(baseline),
            "output_sha256": _sha256(output), "template_sha256": weights.TEMPLATE_SHA256,
            "weights": json.loads((model_dir / "weights.json").read_text(encoding="utf-8")),
            "percentile": percentile, "view_weights": [0.1, 0.4, 0.5], "threshold": 0.45,
            "registration_parameters": asdict(parameters), "registration": registration,
            "inverse_interpolation": "SimpleITK linear (float32) then FillHoles rule and uint8 cast",
            "phase_seconds": phases, "human_anatomical_review": "pending"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, required=True, help="float32 mean-b0 NIfTI (brain_mask._mean_b0)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=None,
                        help="override the A6 default (SLICERDSPACE_STAGE_CPUS, affinity, cpu_count)")
    parser.add_argument("--keep-dir", type=Path, default=None, help="also write transform and intermediates here")
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be positive")
    record = {"route": ROUTE, "requested": args.device, "framework": "pytorch"}
    try:
        record = run(args.input, args.output, args.model_dir, args.template or args.model_dir / TEMPLATE_NAME,
                     device=args.device, threads=args.threads, keep_dir=args.keep_dir)
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        args.record.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        raise
    args.record.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

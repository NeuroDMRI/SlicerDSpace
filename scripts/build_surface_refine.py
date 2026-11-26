#!/usr/bin/env python3
"""Build and smoke-test the standalone precompiled DDSurfer surface_refine shared library."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CPP_SOURCE = REPO_ROOT / "anatomy" / "surface" / "ddsurfer_backend" / "inference" / "surface_refine.cpp"
DEFAULT_OUT_DIR = REPO_ROOT / "anatomy" / "surface" / "ddsurfer_backend" / "inference" / "bin"

EXPORTED_SYMBOLS = (
    "refine_dl_surface_physics_flow",
    "refine_dl_pial_physics_flow",
    "count_self_intersections_tdf",
    "fast_affine_resample_tdf",
    "fast_gaussian_3d_tdf",
    "compute_3d_sheetness_tdf",
    "fast_edt_3d_tdf",
    "compute_vertex_normals_tdf",
    "fast_gmm_2c_tdf",
)


def normalized_platform() -> tuple[str, str, str]:
    sys_name = platform.system().lower()
    mach = platform.machine().lower()
    arch = {"amd64": "x86_64", "x64": "x86_64", "aarch64": "arm64"}.get(mach, mach)
    ext = ".dll" if sys_name == "windows" else (".dylib" if sys_name == "darwin" else ".so")
    return sys_name, arch, ext


def find_compiler(sys_name: str) -> tuple[str, str]:
    """Return (compiler_path, toolchain_kind) where toolchain_kind is 'gcc', 'clang', or 'msvc'."""
    env_cxx = os.environ.get("CXX", "").strip()
    if env_cxx and shutil.which(env_cxx):
        resolved = shutil.which(env_cxx)
        assert resolved is not None
        low = Path(resolved).name.lower()
        return resolved, ("clang" if "clang" in low else ("msvc" if low in ("cl", "cl.exe") else "gcc"))

    if sys_name == "windows":
        # Prefer MinGW-w64 g++ for full OpenMP 4.5 + static standalone DLL with zero external runtime deps.
        win_candidates = [
            shutil.which("g++"),
            r"C:\msys64\mingw64\bin\g++.exe",
            r"C:\msys64\ucrt64\bin\g++.exe",
            r"C:\ProgramData\chocolatey\bin\g++.exe",
        ]
        for cand in win_candidates:
            if cand and Path(cand).is_file():
                return str(cand), "gcc"
        cl = shutil.which("cl")
        if cl:
            return cl, "msvc"
        raise RuntimeError("No C++17 compiler (MinGW g++ or MSVC cl) found on Windows.")

    if sys_name == "darwin":
        for name in ("clang++", "c++", "g++"):
            found = shutil.which(name)
            if found:
                return found, "clang" if "clang" in name or name == "c++" else "gcc"
        raise RuntimeError("No C++17 compiler found on macOS.")

    for name in ("g++", "clang++", "c++"):
        found = shutil.which(name)
        if found:
            return found, "clang" if "clang" in name else "gcc"
    raise RuntimeError("No C++17 compiler found on Linux.")


def build_library(out_dir: Path, *, source: Path = CPP_SOURCE) -> Path:
    sys_name, arch, ext = normalized_platform()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    lib_tag = "darwin_universal2" if sys_name == "darwin" else f"{sys_name}_{arch}"
    out_path = out_dir / f"libsurface_refine_{lib_tag}{ext}"
    tmp_path = out_dir / f".tmp_{out_path.name}"

    cxx, kind = find_compiler(sys_name)
    commands: list[list[str]] = []
    if sys_name == "windows":
        if kind == "gcc":
            # Static libgcc/libstdc++/libgomp/winpthread -> zero external DLL dependencies.
            commands.append([
                cxx, "-O3", "-fno-math-errno", "-funroll-loops", "-std=c++17", "-fopenmp",
                "-shared", "-static", "-static-libgcc", "-static-libstdc++",
                str(source), "-o", str(tmp_path),
                "-Wl,-Bstatic", "-lgomp", "-lpthread", "-Wl,-Bdynamic",
            ])
            commands.append([
                cxx, "-O3", "-fno-math-errno", "-funroll-loops", "-std=c++17",
                "-shared", "-static", "-static-libgcc", "-static-libstdc++",
                str(source), "-o", str(tmp_path),
            ])
        else:
            commands.append([
                cxx, "/nologo", "/O2", "/std:c++17", "/EHsc", "/MT", "/LD",
                str(source), f"/Fe:{tmp_path}",
            ])
    elif sys_name == "darwin":
        # Universal2 fat dylib supports both 3D Slicer 5.10 (x86_64 Rosetta) and native arm64 Python/Slicer.
        commands.append([
            cxx, "-O3", "-fno-math-errno", "-funroll-loops", "-std=c++17",
            "-mmacosx-version-min=13.0", "-arch", "arm64", "-arch", "x86_64",
            "-shared", "-fPIC",
            "-install_name", f"@loader_path/{out_path.name}",
            str(source), "-o", str(tmp_path),
        ])
    else:
        commands.append([
            cxx, "-O3", "-fno-math-errno", "-funroll-loops", "-std=c++17", "-fopenmp",
            "-shared", "-fPIC", "-static-libstdc++", "-static-libgcc",
            str(source), "-o", str(tmp_path),
        ])
        commands.append([
            cxx, "-O3", "-fno-math-errno", "-funroll-loops", "-std=c++17", "-fopenmp",
            "-shared", "-fPIC",
            str(source), "-o", str(tmp_path),
        ])

    last_err = ""
    built = False
    try:
        for cmd in commands:
            print("[build_surface_refine] Running:", " ".join(cmd), flush=True)
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if proc.returncode == 0 and tmp_path.is_file() and tmp_path.stat().st_size > 0:
                os.replace(tmp_path, out_path)
                built = True
                break
            last_err = proc.stdout.strip()
            print(f"[build_surface_refine] Attempt failed (rc={proc.returncode}): {last_err[-400:]}", flush=True)
    finally:
        tmp_path.unlink(missing_ok=True)
        for leftover in out_dir.glob(".tmp_*.exp"):
            leftover.unlink(missing_ok=True)
        for leftover in out_dir.glob(".tmp_*.lib"):
            leftover.unlink(missing_ok=True)
        for leftover in out_dir.glob("*.obj"):
            leftover.unlink(missing_ok=True)

    if not built:
        raise RuntimeError(f"Failed to build {out_path.name}:\n{last_err}")

    if sys_name == "darwin" and shutil.which("strip"):
        subprocess.run(["strip", "-x", str(out_path)], check=False)
    elif sys_name == "linux" and shutil.which("strip"):
        subprocess.run(["strip", "--strip-unneeded", str(out_path)], check=False)

    return out_path


def verify_library(lib_path: Path) -> None:
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(lib_path.parent))
    lib = ctypes.CDLL(str(lib_path))
    for sym in EXPORTED_SYMBOLS:
        fn = getattr(lib, sym, None)
        if fn is None:
            raise RuntimeError(f"Missing exported symbol {sym!r} in {lib_path}")

    # Numerical smoke check on fast_gaussian_3d_tdf + count_self_intersections_tdf
    c_fp = ctypes.POINTER(ctypes.c_float)
    c_dp = ctypes.POINTER(ctypes.c_double)
    c_ip = ctypes.POINTER(ctypes.c_int)
    lib.fast_gaussian_3d_tdf.argtypes = [c_fp, c_fp, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int]
    lib.fast_gaussian_3d_tdf.restype = None
    src = (ctypes.c_float * 27)(*([1.0] * 27))
    dst = (ctypes.c_float * 27)()
    lib.fast_gaussian_3d_tdf(src, dst, 3, 3, 3, 1.0, 2)
    if abs(dst[13] - 1.0) > 1e-4:
        raise RuntimeError(f"Gaussian smoke check failed: center={dst[13]}")

    lib.count_self_intersections_tdf.argtypes = [c_dp, ctypes.c_int, c_ip, ctypes.c_int, ctypes.c_int]
    lib.count_self_intersections_tdf.restype = ctypes.c_int
    verts = (ctypes.c_double * 12)(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    faces = (ctypes.c_int * 12)(0, 1, 2, 0, 1, 3, 1, 2, 3, 2, 0, 3)
    si = lib.count_self_intersections_tdf(verts, 4, faces, 4, 2)
    if si != 0:
        raise RuntimeError(f"Expected 0 self-intersections on regular tetrahedron, got {si}")

    digest = hashlib.sha256(lib_path.read_bytes()).hexdigest()
    print(f"[build_surface_refine] Verified {lib_path.name} ({lib_path.stat().st_size:,} bytes, sha256={digest})", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--source", type=Path, default=CPP_SOURCE)
    args = parser.parse_args(argv)
    lib_path = build_library(args.out_dir, source=args.source)
    verify_library(lib_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

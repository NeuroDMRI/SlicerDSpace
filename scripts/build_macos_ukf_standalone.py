#!/usr/bin/env python3
"""Build a standalone, Slicer-independent Apple Silicon (arm64) UKFTractography binary for macOS."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile

DEFAULT_REPO_URL = "https://github.com/NeuroDMRI/ukftractography_gpu.git"
DEFAULT_REPO_REF = "main"
DETERMINISTIC_MTIME = 1700000000
ARCHIVE_NAME = "ukf-standalone-darwin-arm64.tar.gz"
RUNTIME_DIR_NAME = "ukf-darwin-arm64"
ALLOWED_SYSTEM_PREFIXES = ("/usr/lib/", "/System/Library/Frameworks/")

_SUPERBUILD_CMAKELISTS = """cmake_minimum_required(VERSION 3.22)
project(UKFStandaloneCPUDependencies LANGUAGES C CXX)
include(ExternalProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(UKF_DEPENDENCIES "${CMAKE_BINARY_DIR}/dependencies")
set(UKF_SOURCE "@UKF_SOURCE_DIR@" CACHE PATH "Path to ukftractography source")
set(UKF_STANDALONE_SOURCE "@UKF_STANDALONE_SOURCE_DIR@" CACHE PATH "Path to standalone CPU CMake dir")

set(common_args
  -DCMAKE_BUILD_TYPE:STRING=Release
  -DBUILD_TESTING:BOOL=OFF
  -DBUILD_SHARED_LIBS:BOOL=OFF
  -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
  -DCMAKE_CXX_STANDARD:STRING=17
  -DCMAKE_CXX_STANDARD_REQUIRED:BOOL=ON
  -DCMAKE_OSX_ARCHITECTURES:STRING=arm64
  -DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=13.0
  -DCMAKE_INSTALL_PREFIX:PATH=${UKF_DEPENDENCIES}
  -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
  -DCMAKE_POLICY_DEFAULT_CMP0024:STRING=OLD
  -DCMAKE_DISABLE_FIND_PACKAGE_X11:BOOL=ON
  -DCMAKE_IGNORE_PREFIX_PATH:PATH=/opt/homebrew
)

set(zlib_static "${UKF_DEPENDENCIES}/lib/libz.a")

ExternalProject_Add(zlib
  GIT_REPOSITORY https://github.com/madler/zlib.git
  GIT_TAG v1.3.1
  GIT_SHALLOW TRUE
  CMAKE_CACHE_ARGS
    ${common_args}
    -DZLIB_BUILD_EXAMPLES:BOOL=OFF
)

ExternalProject_Add(Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  GIT_SHALLOW TRUE
  CMAKE_CACHE_ARGS
    ${common_args}
    -DEIGEN_BUILD_DOC:BOOL=OFF
    -DEIGEN_BUILD_TESTING:BOOL=OFF
    -DEIGEN_BUILD_BLAS:BOOL=OFF
    -DEIGEN_BUILD_LAPACK:BOOL=OFF
    -DCMAKE_Fortran_COMPILER:FILEPATH=NOTFOUND
)

ExternalProject_Add(ITK
  GIT_REPOSITORY https://github.com/InsightSoftwareConsortium/ITK.git
  GIT_TAG v5.4.2
  GIT_SHALLOW TRUE
  CMAKE_CACHE_ARGS
    ${common_args}
    -DBUILD_EXAMPLES:BOOL=OFF
    -DITK_BUILD_DEFAULT_MODULES:BOOL=OFF
    -DModule_ITKCommon:BOOL=ON
    -DModule_ITKIOXML:BOOL=ON
    -DKWSYS_USE_MD5:BOOL=ON
    -DITK_USE_FFTWD:BOOL=OFF
    -DITK_USE_FFTWF:BOOL=OFF
  INSTALL_COMMAND ""
)

ExternalProject_Add(VTK
  GIT_REPOSITORY https://github.com/Kitware/VTK.git
  GIT_TAG v9.4.1
  GIT_SHALLOW TRUE
  GIT_SUBMODULES ""
  GIT_CONFIG core.longpaths=true
  CMAKE_CACHE_ARGS
    ${common_args}
    -DVTK_WRAP_PYTHON:BOOL=OFF
    -DVTK_GROUP_ENABLE_StandAlone:STRING=DONT_WANT
    -DVTK_GROUP_ENABLE_Rendering:STRING=NO
    -DVTK_MODULE_ENABLE_VTK_CommonSystem:STRING=YES
    -DVTK_MODULE_ENABLE_VTK_CommonCore:STRING=YES
    -DVTK_MODULE_ENABLE_VTK_CommonMath:STRING=YES
    -DVTK_MODULE_ENABLE_VTK_CommonMisc:STRING=YES
    -DVTK_MODULE_ENABLE_VTK_CommonTransforms:STRING=YES
    -DVTK_MODULE_ENABLE_VTK_IOLegacy:STRING=YES
    -DVTK_MODULE_ENABLE_VTK_IOXML:STRING=YES
  INSTALL_COMMAND ""
)

ExternalProject_Add(teem
  GIT_REPOSITORY https://github.com/Slicer/teem.git
  GIT_TAG e4746083c0e1dc0c137124c41eca5d23adf73bfa
  DEPENDS zlib
  CMAKE_CACHE_ARGS
    ${common_args}
    -DTeem_PTHREAD:BOOL=OFF
    -DTeem_BZIP2:BOOL=OFF
    -DTeem_ZLIB:BOOL=ON
    -DTeem_PNG:BOOL=OFF
    -DZLIB_INCLUDE_DIR:PATH=${UKF_DEPENDENCIES}/include
    -DZLIB_LIBRARY:FILEPATH=${zlib_static}
  INSTALL_COMMAND ""
)

ExternalProject_Add(SlicerExecutionModel
  GIT_REPOSITORY https://github.com/Slicer/SlicerExecutionModel.git
  GIT_TAG 0e28e41aa167b477bc4889725ca08157c152e584
  DEPENDS ITK zlib
  CMAKE_CACHE_ARGS
    ${common_args}
    -DITK_DIR:PATH=${CMAKE_BINARY_DIR}/ITK-prefix/src/ITK-build
    -DZLIB_INCLUDE_DIR:PATH=${UKF_DEPENDENCIES}/include
    -DZLIB_LIBRARY:FILEPATH=${zlib_static}
  INSTALL_COMMAND ""
)

ExternalProject_Add(backend
  SOURCE_DIR "${UKF_STANDALONE_SOURCE}"
  BINARY_DIR "${CMAKE_BINARY_DIR}/backend"
  DEPENDS zlib Eigen ITK VTK teem SlicerExecutionModel
  CMAKE_CACHE_ARGS
    ${common_args}
    -DUKF_SOURCE:PATH=${UKF_SOURCE}
    -DITK_DIR:PATH=${CMAKE_BINARY_DIR}/ITK-prefix/src/ITK-build
    -DVTK_DIR:PATH=${CMAKE_BINARY_DIR}/VTK-prefix/src/VTK-build
    -DTeem_DIR:PATH=${CMAKE_BINARY_DIR}/teem-prefix/src/teem-build
    -DSlicerExecutionModel_DIR:PATH=${CMAKE_BINARY_DIR}/SlicerExecutionModel-prefix/src/SlicerExecutionModel-build
    -DEigen3_DIR:PATH=${UKF_DEPENDENCIES}/share/eigen3/cmake
    -DZLIB_INCLUDE_DIR:PATH=${UKF_DEPENDENCIES}/include
    -DZLIB_LIBRARY:FILEPATH=${zlib_static}
  INSTALL_COMMAND ""
)
"""

_STANDALONE_CMAKELISTS = """cmake_minimum_required(VERSION 3.22)
project(UKFStandaloneCPU LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Target macOS architecture" FORCE)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(UKF_SOURCE "" CACHE PATH "Root of ukftractography_gpu checkout")
if(NOT EXISTS "${UKF_SOURCE}/ukf/cli.cc")
  message(FATAL_ERROR "UKF_SOURCE does not point to ukftractography_gpu: ${UKF_SOURCE}")
endif()

find_package(Threads REQUIRED)
find_package(ITK 5 REQUIRED)
include(${ITK_USE_FILE})
find_package(VTK 9 REQUIRED COMPONENTS CommonSystem CommonCore CommonMath CommonMisc
  CommonTransforms IOLegacy IOXML)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(ZLIB REQUIRED)
find_package(Teem REQUIRED)
find_package(SlicerExecutionModel REQUIRED)
include(${GenerateCLP_USE_FILE})

configure_file("${UKF_SOURCE}/UKFTractography/UKFTractography.xml"
  "${CMAKE_CURRENT_BINARY_DIR}/UKFTractography.xml" COPYONLY)
set(cli_sources "${UKF_SOURCE}/ukf/cli.cc")
generateclp(cli_sources "${CMAKE_CURRENT_BINARY_DIR}/UKFTractography.xml")

set(UKF_REVISION "source-archive")
if(EXISTS "${UKF_SOURCE}/.git")
  find_package(Git REQUIRED)
  execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${UKF_SOURCE}" rev-parse HEAD
    OUTPUT_VARIABLE UKF_REVISION OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/git_version.cc"
  "#include \\"git_version.h\\"\\nconst char UKF_GIT_HASH[] = \\"${UKF_REVISION}\\";\\n")

set(base_sources utilities.cc tractography.cc unscented_kalman_filter.cc seed.cc
  ukffiber.cc NrrdData.cc vtk_writer.cc dwi_normalize.cc thread.cc
  QuadProg++_Eigen.cc filter_model.cc filter_Full1T.cc filter_Full1T_FW.cc
  filter_Full2T.cc filter_Full2T_FW.cc filter_Full3T.cc filter_NODDI1F.cc
  filter_NODDI2F.cc filter_Simple1T.cc filter_Simple1T_FW.cc filter_Simple2T.cc
  filter_Simple2T_FW.cc filter_Simple3T.cc)
list(TRANSFORM base_sources PREPEND "${UKF_SOURCE}/ukf/")

add_library(UKFBase STATIC ${base_sources} ${cli_sources}
  "${CMAKE_CURRENT_BINARY_DIR}/git_version.cc" "${UKF_SOURCE}/common/BRAINSThreadControl.cxx")
target_compile_definitions(UKFBase PUBLIC UKF_STATIC TEEM_STATIC)
target_include_directories(UKFBase PUBLIC "${UKF_SOURCE}/ukf" "${UKF_SOURCE}/common"
  "${CMAKE_CURRENT_BINARY_DIR}" ${Teem_INCLUDE_DIRS})
target_link_libraries(UKFBase PUBLIC Eigen3::Eigen Threads::Threads
  ${ITK_LIBRARIES} ${VTK_LIBRARIES} ${SlicerExecutionModel_LIBRARIES} teem ZLIB::ZLIB)

add_executable(UKFTractography
  "${UKF_SOURCE}/platforms/standalone/main.cc"
  "${UKF_SOURCE}/UKFTractography/UKFTractography.cxx")
target_link_libraries(UKFTractography PRIVATE UKFBase)
set_target_properties(UKFTractography PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
"""


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    print("[cmd]", " ".join(str(x) for x in cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, env=env, check=True, text=True)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_binary_dependencies(binary: Path) -> dict[str, object]:
    """Verify native arm64 architecture, system-only dynamic libraries, and CLI XML output."""

    file_out = subprocess.check_output(["file", str(binary)], text=True).strip()
    if "arm64" not in file_out or "Mach-O" not in file_out:
        raise RuntimeError(f"Expected native Mach-O arm64 binary for {binary}, got: {file_out}")

    otool_out = subprocess.check_output(["otool", "-L", str(binary)], text=True)
    deps: list[str] = []
    disallowed: list[str] = []
    for raw_line in otool_out.splitlines()[1:]:
        line = raw_line.strip()
        if not line:
            continue
        dep_path = re.split(r"\s+\(compatibility version", line, maxsplit=1)[0].strip()
        deps.append(dep_path)
        if not dep_path.startswith(ALLOWED_SYSTEM_PREFIXES):
            disallowed.append(dep_path)
    if disallowed:
        raise RuntimeError(
            f"Standalone binary {binary} links non-system libraries: {disallowed}"
        )

    xml_proc = subprocess.run([str(binary), "--xml"], capture_output=True, text=True, timeout=30)
    if xml_proc.returncode != 0:
        raise RuntimeError(f"{binary} --xml failed ({xml_proc.returncode}): {xml_proc.stderr}")
    if "<executable>" not in xml_proc.stdout or "<name>dwiFile</name>" not in xml_proc.stdout:
        raise RuntimeError(f"Unexpected --xml output from {binary}: {xml_proc.stdout[:400]}")

    return {"file": file_out, "deps": deps}


def package_standalone_tarball(
    *,
    binary: Path,
    xml_path: Path,
    source_repo: str,
    source_commit: str,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Package deterministic ukf-standalone-darwin-arm64.tar.gz and its .sha256 sidecar."""

    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / ARCHIVE_NAME
    tmp_path = archive_path.with_name(archive_path.name + ".tmp")

    runtime_meta = {
        "kind": "ukf-standalone",
        "platform": "darwin-arm64",
        "arch": "arm64",
        "slicer_independent": True,
        "activation_key_included": False,
        "executable": "bin/UKFTractography",
        "source_repo": source_repo,
        "source_commit": source_commit,
    }
    meta_bytes = (json.dumps(runtime_meta, indent=2) + "\n").encode("utf-8")

    entries: list[tuple[str, bytes, int]] = [
        (f"{RUNTIME_DIR_NAME}/bin/UKFTractography", binary.read_bytes(), 0o755),
        (f"{RUNTIME_DIR_NAME}/bin/UKFTractography.xml", xml_path.read_bytes(), 0o644),
        (f"{RUNTIME_DIR_NAME}/runtime.json", meta_bytes, 0o644),
    ]
    dirs = (
        f"{RUNTIME_DIR_NAME}",
        f"{RUNTIME_DIR_NAME}/bin",
    )

    with tarfile.open(tmp_path, "w:gz", format=tarfile.PAX_FORMAT, compresslevel=6) as tf:
        for d in dirs:
            d_info = tarfile.TarInfo(name=d)
            d_info.type = tarfile.DIRTYPE
            d_info.mode = 0o755
            d_info.mtime = DETERMINISTIC_MTIME
            d_info.uid = 0
            d_info.gid = 0
            d_info.uname = "root"
            d_info.gname = "root"
            tf.addfile(d_info)
        for arcname, payload, mode in entries:
            f_info = tarfile.TarInfo(name=arcname)
            f_info.type = tarfile.REGTYPE
            f_info.size = len(payload)
            f_info.mode = mode
            f_info.mtime = DETERMINISTIC_MTIME
            f_info.uid = 0
            f_info.gid = 0
            f_info.uname = "root"
            f_info.gname = "root"
            tf.addfile(f_info, io.BytesIO(payload))

    os.replace(tmp_path, archive_path)
    digest = _sha256_file(archive_path)
    sha_path = out_dir / f"{ARCHIVE_NAME}.sha256"
    sha_path.write_text(f"{digest}  {ARCHIVE_NAME}\n", encoding="utf-8")
    return archive_path, sha_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--ref", default=DEFAULT_REPO_REF)
    parser.add_argument("--jobs", type=int, default=max(2, os.cpu_count() or 4))
    args = parser.parse_args(argv)

    if sys.platform != "darwin":
        raise RuntimeError("build_macos_ukf_standalone.py must run on macOS.")

    work_dir = args.work_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    src_dir = work_dir / "ukf_src"
    sb_src = work_dir / "superbuild_cpu"
    standalone_src = work_dir / "standalone_cpu"
    sb_build = work_dir / "build"

    work_dir.mkdir(parents=True, exist_ok=True)
    if (src_dir / ".git").is_dir():
        _run(["git", "fetch", "--depth=1", "origin", args.ref], cwd=src_dir)
        _run(["git", "checkout", "--force", "FETCH_HEAD"], cwd=src_dir)
    else:
        shutil.rmtree(src_dir, ignore_errors=True)
        _run(["git", "clone", "--depth=1", "--branch", args.ref, args.repo_url, str(src_dir)])

    source_commit = subprocess.check_output(
        ["git", "-C", str(src_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    print(f"[info] Building standalone CPU arm64 UKF from {args.repo_url} @ {source_commit}", flush=True)

    sb_src.mkdir(parents=True, exist_ok=True)
    standalone_src.mkdir(parents=True, exist_ok=True)
    (sb_src / "CMakeLists.txt").write_text(
        _SUPERBUILD_CMAKELISTS
        .replace("@UKF_SOURCE_DIR@", src_dir.as_posix())
        .replace("@UKF_STANDALONE_SOURCE_DIR@", standalone_src.as_posix()),
        encoding="utf-8",
    )
    (standalone_src / "CMakeLists.txt").write_text(_STANDALONE_CMAKELISTS, encoding="utf-8")

    py_bin_dir = Path(sys.executable).resolve().parent
    cmake_candidate = py_bin_dir / "cmake"
    cmake_bin = str(cmake_candidate) if cmake_candidate.is_file() else (shutil.which("cmake") or "cmake")
    ninja_candidate = py_bin_dir / "ninja"
    ninja_bin = str(ninja_candidate) if ninja_candidate.is_file() else shutil.which("ninja")

    build_env = os.environ.copy()
    path_parts = [str(py_bin_dir)] + [p for p in build_env.get("PATH", "").split(os.pathsep) if p and p != str(py_bin_dir)]
    build_env["PATH"] = os.pathsep.join(path_parts)

    gen_args: list[str] = ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja_bin}"] if ninja_bin else []
    c_comp = shutil.which("clang") or "/usr/bin/clang"
    cxx_comp = shutil.which("clang++") or "/usr/bin/clang++"
    _run([cmake_bin, "--version"], env=build_env)
    _run([
        cmake_bin,
        "-S", str(sb_src),
        "-B", str(sb_build),
        *gen_args,
        f"-DCMAKE_C_COMPILER={c_comp}",
        f"-DCMAKE_CXX_COMPILER={cxx_comp}",
        "-DCMAKE_OSX_ARCHITECTURES=arm64",
        "-DCMAKE_OSX_DEPLOYMENT_TARGET=13.0",
    ], env=build_env)
    _run([cmake_bin, "--build", str(sb_build), "--parallel", str(max(1, int(args.jobs)))], env=build_env)

    binary = sb_build / "backend" / "bin" / "UKFTractography"
    xml_path = src_dir / "UKFTractography" / "UKFTractography.xml"
    if not binary.is_file():
        raise FileNotFoundError(f"Built UKFTractography binary missing: {binary}")

    info = verify_binary_dependencies(binary)
    print(f"[verify] file: {info['file']}", flush=True)
    print(f"[verify] otool deps: {info['deps']}", flush=True)

    archive_path, sha_path = package_standalone_tarball(
        binary=binary,
        xml_path=xml_path,
        source_repo=args.repo_url,
        source_commit=source_commit,
        out_dir=out_dir,
    )
    print(f"[done] Packaged {archive_path} and {sha_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

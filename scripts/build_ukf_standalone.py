#!/usr/bin/env python3
"""Build a standalone, Slicer-independent CPU UKFTractography binary for Windows x64, Linux x64, or macOS arm64."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET
import zipfile

DEFAULT_REPO_URL = "https://github.com/NeuroDMRI/ukftractography_gpu.git"
DEFAULT_REPO_REF = "main"
DETERMINISTIC_MTIME = 1700000000
DETERMINISTIC_ZIP_DT = (2023, 11, 14, 22, 13, 20)

PLATFORM_SPECS: dict[str, dict[str, str]] = {
    "windows-x64": {
        "os": "windows",
        "arch": "x86_64",
        "archive_name": "ukf-standalone-windows-x64.zip",
        "runtime_dir": "ukf-windows-x64",
        "exe_name": "UKFTractography.exe",
    },
    "linux-x64": {
        "os": "linux",
        "arch": "x86_64",
        "archive_name": "ukf-standalone-linux-x64.tar.gz",
        "runtime_dir": "ukf-linux-x64",
        "exe_name": "UKFTractography",
    },
    "darwin-arm64": {
        "os": "darwin",
        "arch": "arm64",
        "archive_name": "ukf-standalone-darwin-arm64.tar.gz",
        "runtime_dir": "ukf-darwin-arm64",
        "exe_name": "UKFTractography",
    },
}

ALLOWED_MACOS_PREFIXES = ("/usr/lib/", "/System/Library/Frameworks/")
ALLOWED_LINUX_BASENAMES = frozenset({
    "linux-vdso.so.1",
    "libc.so.6",
    "libm.so.6",
    "libpthread.so.0",
    "libdl.so.2",
    "librt.so.1",
    "ld-linux-x86-64.so.2",
})
ALLOWED_WINDOWS_DLL_PREFIXES = (
    "api-ms-win-",
    "ext-ms-win-",
    "vcruntime140",
    "msvcp140",
    "concrt140",
)
ALLOWED_WINDOWS_DLLS = frozenset({
    "kernel32.dll",
    "advapi32.dll",
    "user32.dll",
    "ws2_32.dll",
    "rpcrt4.dll",
    "shell32.dll",
    "ole32.dll",
    "oleaut32.dll",
    "bcrypt.dll",
    "crypt32.dll",
    "shlwapi.dll",
    "psapi.dll",
    "dbghelp.dll",
    "ntdll.dll",
    "ucrtbase.dll",
    "ucrtbased.dll",
    "iphlpapi.dll",
    "userenv.dll",
    "netapi32.dll",
    "secur32.dll",
    "version.dll",
})
MSVC_REDIST_DLLS = ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll")

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
  -DCMAKE_INSTALL_PREFIX:PATH=${UKF_DEPENDENCIES}
  -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
  -DCMAKE_POLICY_DEFAULT_CMP0024:STRING=OLD
  -DCMAKE_POLICY_VERSION_MINIMUM:STRING=3.5
  -DCMAKE_DISABLE_FIND_PACKAGE_X11:BOOL=ON
)

if(APPLE)
  list(APPEND common_args
    -DCMAKE_OSX_ARCHITECTURES:STRING=arm64
    -DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=13.0
    -DCMAKE_IGNORE_PREFIX_PATH:PATH=/opt/homebrew
  )
endif()

if(WIN32)
  set(zlib_static "${UKF_DEPENDENCIES}/lib/zlibstatic.lib")
else()
  set(zlib_static "${UKF_DEPENDENCIES}/lib/libz.a")
endif()

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
if(APPLE)
  set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Target macOS architecture" FORCE)
endif()
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

if(MSVC)
  target_compile_definitions(UKFBase PUBLIC NOMINMAX _USE_MATH_DEFINES)
  target_compile_options(UKFBase PRIVATE /O2 /bigobj /utf-8 /arch:AVX2)
  target_compile_options(UKFTractography PRIVATE /O2 /utf-8 /arch:AVX2)
  if(EXISTS "${UKF_SOURCE}/platforms/standalone/UKFGPU.manifest")
    target_sources(UKFTractography PRIVATE "${UKF_SOURCE}/platforms/standalone/UKFGPU.manifest")
  endif()
elseif(UNIX AND NOT APPLE)
  target_compile_options(UKFBase PRIVATE -O3 -DNDEBUG -mavx2 -mfma)
  target_compile_options(UKFTractography PRIVATE -O3 -DNDEBUG -mavx2 -mfma)
  target_link_options(UKFTractography PRIVATE -static-libstdc++ -static-libgcc)
elseif(APPLE)
  target_compile_options(UKFBase PRIVATE -O3 -DNDEBUG)
  target_compile_options(UKFTractography PRIVATE -O3 -DNDEBUG)
endif()

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


def _parse_pe_imports(binary: Path) -> list[str]:
    """Read imported DLL names from a Windows PE32+ executable without external tools."""
    data = binary.read_bytes()
    if len(data) < 0x40 or data[:2] != b"MZ":
        raise RuntimeError(f"Not a valid PE binary: {binary}")
    pe_off = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe_off : pe_off + 4] != b"PE\x00\x00":
        raise RuntimeError(f"Invalid PE header in {binary}")
    machine, num_sections = struct.unpack_from("<HH", data, pe_off + 4)
    if machine != 0x8664:
        raise RuntimeError(f"Expected x86_64 PE (0x8664), got 0x{machine:04x}")
    opt_hdr_size = struct.unpack_from("<H", data, pe_off + 20)[0]
    opt_off = pe_off + 24
    magic = struct.unpack_from("<H", data, opt_off)[0]
    if magic != 0x20B:
        raise RuntimeError(f"Expected PE32+ (0x20B), got 0x{magic:04x}")
    import_rva, import_size = struct.unpack_from("<II", data, opt_off + 120)
    sec_off = opt_off + opt_hdr_size
    sections = []
    for i in range(num_sections):
        off = sec_off + i * 40
        vsize, vaddr, rsize, rptr = struct.unpack_from("<IIII", data, off + 8)
        sections.append((vaddr, max(vsize, rsize), rptr))

    def rva_to_off(rva: int) -> int | None:
        for vaddr, span, rptr in sections:
            if vaddr <= rva < vaddr + span:
                return rptr + (rva - vaddr)
        return None

    if import_rva == 0 or import_size == 0:
        return []
    desc_off = rva_to_off(import_rva)
    if desc_off is None:
        return []
    imports: list[str] = []
    while desc_off + 20 <= len(data):
        _, _, _, name_rva, first_thunk = struct.unpack_from("<IIIII", data, desc_off)
        if name_rva == 0 and first_thunk == 0:
            break
        name_off = rva_to_off(name_rva)
        if name_off is not None:
            end = data.find(b"\x00", name_off)
            if end > name_off:
                imports.append(data[name_off:end].decode("ascii", errors="replace"))
        desc_off += 20
    return imports


def _verify_xml(binary: Path) -> dict[str, int]:
    xml_proc = subprocess.run([str(binary), "--xml"], capture_output=True, text=True, timeout=30)
    if xml_proc.returncode not in (0, 1):
        raise RuntimeError(f"{binary} --xml failed ({xml_proc.returncode}): {xml_proc.stderr}")
    stdout = xml_proc.stdout or ""
    start = stdout.find("<?xml")
    if start < 0:
        start = stdout.find("<executable")
    end = stdout.find("</executable>", max(0, start))
    if start < 0 or end < 0:
        raise RuntimeError(f"Incomplete --xml output from {binary} (rc={xml_proc.returncode}): {stdout[:400]} | {xml_proc.stderr[:400]}")
    xml_text = stdout[start : end + len("</executable>")]
    root = ET.fromstring(xml_text)
    if root.tag != "executable" or "dwiFile" not in {n.text for n in root.findall("./parameters/*/name") if n.text}:
        raise RuntimeError(f"Unexpected --xml schema from {binary}: {xml_text[:400]}")
    return {"xml_returncode": xml_proc.returncode, "xml_bytes": len(xml_text)}


def verify_binary_dependencies(binary: Path, target: str) -> dict[str, object]:
    """Verify architecture, system-only dynamic libraries, and CLI XML output."""
    deps: list[str] = []
    disallowed: list[str] = []
    file_desc = ""

    if target == "darwin-arm64":
        file_desc = subprocess.check_output(["file", str(binary)], text=True).strip()
        if "arm64" not in file_desc or "Mach-O" not in file_desc:
            raise RuntimeError(f"Expected native Mach-O arm64 binary for {binary}, got: {file_desc}")
        otool_out = subprocess.check_output(["otool", "-L", str(binary)], text=True)
        for raw_line in otool_out.splitlines()[1:]:
            line = raw_line.strip()
            if not line:
                continue
            dep_path = re.split(r"\s+\(compatibility version", line, maxsplit=1)[0].strip()
            deps.append(dep_path)
            if not dep_path.startswith(ALLOWED_MACOS_PREFIXES):
                disallowed.append(dep_path)
    elif target == "linux-x64":
        file_desc = subprocess.check_output(["file", str(binary)], text=True).strip()
        if "ELF 64-bit" not in file_desc or "x86-64" not in file_desc:
            raise RuntimeError(f"Expected ELF 64-bit x86-64 binary for {binary}, got: {file_desc}")
        ldd_out = subprocess.check_output(["ldd", str(binary)], text=True)
        for raw_line in ldd_out.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            token = line.split()[0]
            base = Path(token).name
            deps.append(token)
            if base not in ALLOWED_LINUX_BASENAMES:
                disallowed.append(line)
    elif target == "windows-x64":
        deps = _parse_pe_imports(binary)
        file_desc = f"PE32+ x86_64 ({binary.stat().st_size:,} bytes)"
        for dll in deps:
            low = dll.lower()
            if low in ALLOWED_WINDOWS_DLLS or low.startswith(ALLOWED_WINDOWS_DLL_PREFIXES):
                continue
            disallowed.append(dll)
    else:
        raise ValueError(f"Unsupported target: {target}")

    if disallowed:
        raise RuntimeError(f"Standalone binary {binary} links non-system libraries: {disallowed}")

    xml_info = _verify_xml(binary)
    return {"file": file_desc, "deps": deps, **xml_info}


def _find_msvc_redist_dlls() -> dict[str, Path]:
    """Locate MSVC C++ runtime DLLs so standalone UKFTractography.exe works even without VC++ Redist."""
    found: dict[str, Path] = {}
    search_dirs: list[Path] = []
    redist_env = os.environ.get("VCToolsRedistDir", "").strip()
    if redist_env:
        redist_root = Path(redist_env)
        for sub in ("x64/Microsoft.VC143.CRT", "x64/Microsoft.VC142.CRT", "x64/Microsoft.VC141.CRT"):
            cand = redist_root / sub
            if cand.is_dir():
                search_dirs.append(cand)
        if redist_root.is_dir():
            for cand in sorted(redist_root.glob("x64/Microsoft.VC*.CRT")):
                if cand.is_dir() and cand not in search_dirs:
                    search_dirs.append(cand)
    sys32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    if sys32.is_dir():
        search_dirs.append(sys32)
    for name in MSVC_REDIST_DLLS:
        for d in search_dirs:
            cand = d / name
            if cand.is_file():
                found[name] = cand
                break
    return found


def package_standalone_archive(
    *,
    target: str,
    binary: Path,
    xml_path: Path,
    source_repo: str,
    source_commit: str,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Package deterministic standalone UKF archive and its .sha256 sidecar."""
    spec = PLATFORM_SPECS[target]
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_name = spec["archive_name"]
    runtime_dir = spec["runtime_dir"]
    exe_name = spec["exe_name"]
    archive_path = out_dir / archive_name
    tmp_path = archive_path.with_name(archive_path.name + ".tmp")

    runtime_meta = {
        "kind": "ukf-standalone",
        "platform": target,
        "arch": spec["arch"],
        "slicer_independent": True,
        "activation_key_included": False,
        "executable": f"bin/{exe_name}",
        "source_repo": source_repo,
        "source_commit": source_commit,
    }
    meta_bytes = (json.dumps(runtime_meta, indent=2) + "\n").encode("utf-8")

    entries: list[tuple[str, bytes, int]] = [
        (f"{runtime_dir}/bin/{exe_name}", binary.read_bytes(), 0o755),
        (f"{runtime_dir}/bin/UKFTractography.xml", xml_path.read_bytes(), 0o644),
        (f"{runtime_dir}/runtime.json", meta_bytes, 0o644),
    ]
    if target == "windows-x64":
        for dll_name, dll_path in sorted(_find_msvc_redist_dlls().items()):
            entries.append((f"{runtime_dir}/bin/{dll_name}", dll_path.read_bytes(), 0o755))

    if archive_name.endswith(".zip"):
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for arcname, payload, mode in entries:
                zi = zipfile.ZipInfo(filename=arcname, date_time=DETERMINISTIC_ZIP_DT)
                zi.compress_type = zipfile.ZIP_DEFLATED
                zi.external_attr = (stat.S_IFREG | mode) << 16
                zf.writestr(zi, payload)
    else:
        dirs = (f"{runtime_dir}", f"{runtime_dir}/bin")
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
    sha_path = out_dir / f"{archive_name}.sha256"
    sha_path.write_text(f"{digest}  {archive_name}\n", encoding="utf-8")
    return archive_path, sha_path


def _detect_target() -> str:
    if sys.platform == "win32":
        return "windows-x64"
    if sys.platform == "darwin":
        return "darwin-arm64"
    if sys.platform.startswith("linux"):
        return "linux-x64"
    raise RuntimeError(f"Unsupported host platform: {sys.platform}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=sorted(PLATFORM_SPECS), default=None)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--ref", default=DEFAULT_REPO_REF)
    parser.add_argument("--jobs", type=int, default=max(2, os.cpu_count() or 4))
    args = parser.parse_args(argv)

    target = args.target or _detect_target()
    spec = PLATFORM_SPECS[target]
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
    print(f"[info] Building standalone CPU {target} UKF from {args.repo_url} @ {source_commit}", flush=True)

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
    scripts_dir = py_bin_dir / "Scripts"
    build_env = os.environ.copy()
    extra_paths = [str(py_bin_dir)]
    if scripts_dir.is_dir():
        extra_paths.append(str(scripts_dir))
    build_env["PATH"] = os.pathsep.join(
        extra_paths + [p for p in build_env.get("PATH", "").split(os.pathsep) if p and p not in extra_paths]
    )

    cmake_bin = shutil.which("cmake", path=build_env["PATH"]) or "cmake"
    ninja_bin = shutil.which("ninja", path=build_env["PATH"])
    gen_args: list[str] = ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja_bin}"] if ninja_bin else []

    cmake_args = [cmake_bin, "-S", str(sb_src), "-B", str(sb_build), *gen_args, "-DCMAKE_BUILD_TYPE=Release"]
    if target == "darwin-arm64":
        c_comp = shutil.which("clang") or "/usr/bin/clang"
        cxx_comp = shutil.which("clang++") or "/usr/bin/clang++"
        cmake_args.extend([
            f"-DCMAKE_C_COMPILER={c_comp}",
            f"-DCMAKE_CXX_COMPILER={cxx_comp}",
            "-DCMAKE_OSX_ARCHITECTURES=arm64",
            "-DCMAKE_OSX_DEPLOYMENT_TARGET=13.0",
        ])
    elif target == "windows-x64":
        cl_exe = shutil.which("cl.exe", path=build_env["PATH"]) or shutil.which("cl", path=build_env["PATH"])
        if cl_exe:
            cmake_args.extend([f"-DCMAKE_C_COMPILER={cl_exe}", f"-DCMAKE_CXX_COMPILER={cl_exe}"])
    elif target == "linux-x64":
        c_comp = shutil.which("gcc") or shutil.which("cc")
        cxx_comp = shutil.which("g++") or shutil.which("c++")
        if c_comp and cxx_comp:
            cmake_args.extend([f"-DCMAKE_C_COMPILER={c_comp}", f"-DCMAKE_CXX_COMPILER={cxx_comp}"])

    _run([cmake_bin, "--version"], env=build_env)
    _run(cmake_args, env=build_env)
    _run([cmake_bin, "--build", str(sb_build), "--parallel", str(max(1, int(args.jobs)))], env=build_env)

    binary = sb_build / "backend" / "bin" / spec["exe_name"]
    if not binary.is_file():
        cand = sb_build / "backend" / "bin" / "Release" / spec["exe_name"]
        if cand.is_file():
            binary = cand
    xml_path = src_dir / "UKFTractography" / "UKFTractography.xml"
    if not binary.is_file():
        raise FileNotFoundError(f"Built UKFTractography binary missing: {binary}")

    info = verify_binary_dependencies(binary, target)
    print(f"[verify] file: {info['file']}", flush=True)
    print(f"[verify] deps: {info['deps']}", flush=True)

    archive_path, sha_path = package_standalone_archive(
        target=target,
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

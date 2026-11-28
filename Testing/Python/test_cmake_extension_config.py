from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _cmake_executable() -> str | None:
    configured = os.environ.get("CMAKE_EXECUTABLE", "").strip()
    if configured and Path(configured).is_file():
        return configured
    return shutil.which("cmake")


def _captured_text(value) -> str:
    """TimeoutExpired keeps the output read before the kill; on POSIX it is bytes even with text=True."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _system_state() -> str:
    """Diagnostic for the CI-only configure timeout: system load and every process with its CPU and memory."""
    lines = []
    try:
        lines.append("load average (1, 5, 15 min): %.2f %.2f %.2f" % os.getloadavg())
    except (AttributeError, OSError) as exc:
        lines.append(f"load average: unavailable ({exc})")
    command = ["ps", "-axo", "pid,ppid,pcpu,pmem,rss,etime,state,command"]
    if os.name == "nt":
        command = ["tasklist", "/v"]
    try:
        listed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
        rows = listed.stdout.splitlines()
        if os.name != "nt" and len(rows) > 1:
            def cpu(row: str) -> float:
                try:
                    return float(row.split(None, 3)[2])
                except (IndexError, ValueError):
                    return 0.0

            rows = rows[:1] + sorted(rows[1:], key=cpu, reverse=True)
        lines.append(f"$ {' '.join(command)} (sorted by %CPU)")
        lines.extend(row[:300] for row in rows)
        if listed.stderr:
            lines.append(listed.stderr[-2000:])
    except (OSError, subprocess.SubprocessError) as exc:
        lines.append(f"process list: unavailable ({exc})")
    return "\n".join(lines)


def _profile_report(path: Path, limit: int = 15) -> str:
    """Slowest steps of a cmake --profiling-output file (google-trace). A killed cmake leaves the file cut off, so the
    events are read one by one until the first incomplete one; steps that began and never ended are listed first."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"profile: unavailable ({exc})"
    decoder, events, index = json.JSONDecoder(), [], 0
    while True:
        while index < len(text) and text[index] in "[,] \t\r\n":
            index += 1
        if index >= len(text):
            break
        try:
            event, index = decoder.raw_decode(text, index)
        except ValueError:
            break
        events.append(event)
    if not events:
        return f"profile: no complete event in {path.name} ({len(text)} bytes)"
    first = events[0].get("ts", 0)
    last = max(event.get("ts", 0) for event in events)
    stack, finished = [], []
    for event in events:
        if event.get("ph") == "B":
            stack.append(event)
        elif event.get("ph") == "E" and stack:
            begin = stack.pop()
            finished.append((event.get("ts", 0) - begin.get("ts", 0), len(stack), begin))

    def describe(begin: dict) -> str:
        args = begin.get("args", {})
        return (f"{begin.get('name', '?')} {str(args.get('functionArgs', ''))[:120]} "
                f"@ {str(args.get('location', ''))[-90:]}")

    lines = [f"profile: {len(events)} events, last event at +{(last - first) / 1e6:.2f} s after the first"]
    lines.append("steps that began and did not end (outermost first):")
    for depth, begin in enumerate(stack):
        lines.append(f"  started +{(begin.get('ts', 0) - first) / 1e6:8.2f} s, open for at least "
                     f"{(last - begin.get('ts', 0)) / 1e6:8.2f} s  depth {depth}  {describe(begin)}")
    lines.append(f"slowest {limit} finished steps:")
    for duration, depth, begin in sorted(finished, key=lambda item: item[0], reverse=True)[:limit]:
        lines.append(f"  {duration / 1e6:8.2f} s  depth {depth}  {describe(begin)}")
    return "\n".join(lines)


def _run_configure(command: list, timeout: float, profile: Path):
    """Run the configure step. Returns (CompletedProcess, None), or (None, diagnostic text) on a timeout. The process
    list is taken before the kill, so it shows what cmake and its children were doing at that moment."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        state = _system_state()
        process.kill()
        stdout, stderr = process.communicate()
        return None, (f"cmake configure timed out after {timeout} s\n--- stdout so far ---\n{_captured_text(stdout)}\n"
                      f"--- stderr so far ---\n{_captured_text(stderr)}\n--- system state before the kill ---\n"
                      f"{state}\n--- cmake profile ---\n{_profile_report(profile)}")
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr), None


class CMakeExtensionConfigTest(unittest.TestCase):
    def test_sdk_prerequisites_are_requested_before_project(self):
        content = (REPO_ROOT / 'CMakeLists.txt').read_text(encoding='utf-8')
        self.assertLess(content.index('find_package(Slicer COMPONENTS ConfigurePrerequisites REQUIRED)'),
                        content.index('project(SlicerDSpace)'))

    def test_extension_configures_with_slicer_cmake_contract(self) -> None:
        cmake = _cmake_executable()
        if not cmake:
            self.skipTest("CMake is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sdk = root / "mock-slicer-sdk"
            sdk.mkdir()
            (sdk / "SlicerConfig.cmake").write_text(
                "\n".join(
                    (
                        'set(Slicer_USE_FILE "${CMAKE_CURRENT_LIST_DIR}/SlicerUse.cmake")',
                        'set(Slicer_EXTENSION_CPACK "${CMAKE_CURRENT_LIST_DIR}/SlicerExtensionCPack.cmake")',
                        'set(Slicer_INSTALL_QTSCRIPTEDMODULES_LIB_DIR "lib/Slicer-5.10/qt-scripted-modules")',
                        "",
                    )
                ),
                encoding="utf-8",
            )
            (sdk / "SlicerUse.cmake").write_text(
                "\n".join(
                    (
                        "include(CMakeParseArguments)",
                        "include(CTest)",
                        "macro(slicerMacroBuildScriptedModule)",
                        '  cmake_parse_arguments(SM "WITH_GENERIC_TESTS" "NAME" "SCRIPTS;RESOURCES" ${ARGN})',
                        '  if(NOT SM_NAME OR NOT SM_SCRIPTS)',
                        '    message(FATAL_ERROR "Invalid scripted module declaration")',
                        "  endif()",
                        '  file(WRITE "${CMAKE_BINARY_DIR}/scripted-module.txt" "${SM_NAME}\\n${SM_SCRIPTS}\\n")',
                        '  foreach(resource IN LISTS SM_SCRIPTS SM_RESOURCES)',
                        '    get_filename_component(resource_dir "${resource}" DIRECTORY)',
                        '    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/${resource}" DESTINATION "${Slicer_INSTALL_QTSCRIPTEDMODULES_LIB_DIR}/${resource_dir}")',
                        '  endforeach()',
                        "endmacro()",
                        "function(slicer_add_python_unittest)",
                        '  file(APPEND "${CMAKE_BINARY_DIR}/registered-tests.txt" "${ARGN}\\n")',
                        "endfunction()",
                        "",
                    )
                ),
                encoding="utf-8",
            )
            (sdk / "SlicerExtensionCPack.cmake").write_text(
                'file(WRITE "${CMAKE_BINARY_DIR}/extension-cpack.txt" "${PROJECT_NAME}\\n${EXTENSION_DEPENDS}\\n")\n',
                encoding="utf-8",
            )
            build = root / "build"
            profile = root / "cmake-profile.json"
            completed, timed_out = _run_configure(
                [
                    cmake,
                    "-S",
                    REPO_ROOT.as_posix(),
                    "-B",
                    build.as_posix(),
                    f"-DSlicer_DIR={sdk.as_posix()}",
                    "-DBUILD_TESTING=ON",
                    f"--profiling-output={profile.as_posix()}",
                    "--profiling-format=google-trace",
                ],
                120,
                profile,
            )
            if timed_out:
                # The last "-- " line, the open profile steps and the process list name what did not finish.
                self.fail(timed_out)

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            module_contract = (build / "scripted-module.txt").read_text(encoding="utf-8")
            package_contract = (build / "extension-cpack.txt").read_text(encoding="utf-8")
            self.assertIn("SlicerDSpace", module_contract)
            self.assertIn("SlicerDSpace.py", module_contract)
            self.assertIn("SlicerDSpace", package_contract)
            self.assertNotIn("SlicerDMRI", package_contract)  # D9: PyTorch is the only extension dependency
            self.assertIn("PyTorch", package_contract)
            registered = (build / "Testing/Python/CTestTestfile.cmake").read_text(encoding="utf-8")
            for script in (REPO_ROOT / "Testing/Python").glob("test_*.py"):
                self.assertIn(script.name, registered)
            self.assertIn('backend-unit', registered)
            self.assertNotIn('slicer.testing.runUnitTest', registered)
            prefix = root / "installed"
            installed = subprocess.run([cmake, "--install", str(build), "--prefix", str(prefix)],
                                       check=False, capture_output=True, text=True, timeout=120)
            self.assertEqual(installed.returncode, 0, installed.stdout + installed.stderr)
            from scripts.extension_runtime.runtime_resources import runtime_files
            from scripts.stage_slicer_extension import stage_extension
            module = prefix / "lib/Slicer-5.10/qt-scripted-modules"
            actual = sorted(path.relative_to(module).as_posix() for path in module.rglob("*") if path.is_file())
            self.assertEqual(actual, runtime_files(REPO_ROOT))
            staged = stage_extension(root / "staged", slicer_version="Slicer-5.10", clean=True, link_mode="copy")
            staged_module = Path(staged["module_dir"])
            for relative in actual:
                self.assertEqual((module / relative).read_bytes(), (staged_module / relative).read_bytes(), relative)


if __name__ == "__main__":
    unittest.main()

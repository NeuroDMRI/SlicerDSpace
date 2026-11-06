from __future__ import annotations

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
            completed = subprocess.run(
                [
                    cmake,
                    "-S",
                    REPO_ROOT.as_posix(),
                    "-B",
                    build.as_posix(),
                    f"-DSlicer_DIR={sdk.as_posix()}",
                    "-DBUILD_TESTING=ON",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )

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

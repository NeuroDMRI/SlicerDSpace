"""B2: keep the test process out of the real per-user data folders.

``install()`` points the home folder and the per-user data variables of every OS at one temporary folder for the
whole process, and it makes ``os.environ.clear()`` (so ``patch.dict(os.environ, ..., clear=True)``) put those
variables back. The code then resolves, under the temporary folder: the SHA-256 cache and the user config
(``default_user_config_path``), the model assets, the managed runtimes and shims (``default_user_model_assets_root``),
and every ``Path.home()`` / ``~`` path (for example ``~/.cache``).

Who imports this module: ``conftest.py`` (pytest), ``test_00_user_data_isolation.py`` (``unittest discover`` imports
it first), and the test modules that need it when they run alone. CMake sets the same variables for each test.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

# Path variables of the SlicerDSpace code. A value left over in the developer's shell would point at real data.
_OVERRIDES = ("USER_CONFIG", "MODEL_ASSETS_DIR", "RUNTIME_ROOT")
_PREFIXES = ("SLICERDSPACE_", "SLICERDDSURFER_")
_installed: Path | None = None


class _ProtectedEnviron(os._Environ):
    """``os.environ`` whose ``clear()`` keeps the isolation variables (``patch.dict(..., clear=True)`` calls it)."""

    protected: dict[str, str] = {}

    def clear(self) -> None:
        super().clear()
        for key, value in self.protected.items():
            self[key] = value


def isolated_variables(home: Path) -> dict[str, str]:
    return {
        "HOME": str(home),
        "USERPROFILE": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_DATA_HOME": str(home / ".local" / "share"),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "APPDATA": str(home / "AppData" / "Roaming"),
        "LOCALAPPDATA": str(home / "AppData" / "Local"),
        "SLICERDSPACE_TEST_HOME": str(home),
    }


def install() -> Path:
    """Install once per process; return the temporary home folder."""
    global _installed
    if _installed is not None:
        return _installed
    home = Path(tempfile.mkdtemp(prefix="slicerdspace-test-home-")).resolve()
    atexit.register(shutil.rmtree, home, ignore_errors=True)
    for key in list(os.environ):
        if key.startswith(_PREFIXES) and key.split("_", 1)[1] in _OVERRIDES:
            del os.environ[key]
    variables = isolated_variables(home)
    os.environ.update(variables)
    _ProtectedEnviron.protected = variables
    os.environ.__class__ = _ProtectedEnviron  # same object, so `from os import environ` holders stay valid
    if os.sys.platform == "darwin":
        try:
            import torch as _darwin_preload_torch  # noqa: F401
        except Exception:
            pass
    _installed = home
    return home


def is_inside(path: Path | str, folder: Path) -> bool:
    try:
        Path(path).resolve().relative_to(folder.resolve())
    except ValueError:
        return False
    return True

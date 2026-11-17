"""Read-only discovery and user-private import of a user's FreeSurfer activation file."""

import os
from pathlib import Path

REGISTRATION_URL = 'https://surfer.nmr.mgh.harvard.edu/registration.html'


def usable_license(path):
    """Check readable, nonempty text; FreeSurfer performs activation validation."""
    if path is None:
        return False
    try:
        path = Path(path).expanduser()
        if not path.is_file() or not 0 < path.stat().st_size <= 65536:
            return False
        data = path.read_bytes()
        return b'\0' not in data and bool(data.decode('utf-8-sig').strip())
    except (OSError, ValueError, TypeError, UnicodeError):
        return False


def managed_surface_license_path(*, home=None, environ=None):
    """Return SlicerDSpace's user-private license storage path outside the extension install tree."""
    from shared.user_config import default_user_config_path

    return default_user_config_path(home=home, environ=environ).parent / 'freesurfer' / 'license.txt'


def import_surface_license(source_path, *, home=None, environ=None):
    """Copy a user-supplied license.txt into SlicerDSpace's user-private configuration directory."""
    if not usable_license(source_path):
        raise ValueError('Invalid FreeSurfer license file.')
    src = Path(source_path).expanduser().resolve()
    dest = managed_surface_license_path(home=home, environ=environ).expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(dest.parent, 0o700)
    except OSError:
        pass
    if not (dest.exists() and dest.resolve() == src):
        tmp = dest.with_name(dest.name + '.tmp')
        tmp.write_bytes(src.read_bytes())
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, dest)
    else:
        try:
            os.chmod(dest, 0o600)
        except OSError:
            pass
    return dest.resolve()


def resolve_surface_license(explicit=None, *, freesurfer_home=None, home=None, environ=None):
    """Explicit choices are authoritative; never silently replace a broken choice."""
    env = os.environ if environ is None else environ
    if explicit:
        return Path(explicit).expanduser().resolve()
    selected = env.get('FS_LICENSE') or env.get('FREESURFER_LICENSE')
    if selected:
        return Path(selected).expanduser().resolve()
    user = Path(home).expanduser() if home is not None else Path.home()
    candidates = [
        managed_surface_license_path(home=home, environ=env),
        user / '.licenses/freesurfer/license.txt',
        user / '.freesurfer/license.txt',
        user / 'license.txt',
    ]
    for root in (freesurfer_home, env.get('FREESURFER_HOME')):
        if root:
            directory = Path(root).expanduser()
            try:
                names = {path.name for path in directory.iterdir()}
            except OSError:
                continue
            # Look at stored names: case-insensitive filesystems alias the
            # public LICENSE.txt notice to a lookup for private license.txt.
            candidates.extend(directory / name for name in ('license.txt', '.license') if name in names)
    return next((path.resolve() for path in candidates if usable_license(path)), None)

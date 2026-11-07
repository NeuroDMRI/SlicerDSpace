"""Read-only discovery of a user's FreeSurfer activation file."""

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


def resolve_surface_license(explicit=None, *, freesurfer_home=None, home=None, environ=None):
    """Explicit choices are authoritative; never silently replace a broken choice."""
    env = os.environ if environ is None else environ
    if explicit:
        return Path(explicit).expanduser().resolve()
    selected = env.get('FS_LICENSE') or env.get('FREESURFER_LICENSE')
    if selected:
        return Path(selected).expanduser().resolve()
    user = Path(home).expanduser() if home is not None else Path.home()
    candidates = [user / '.licenses/freesurfer/license.txt', user / '.freesurfer/license.txt',
                  user / 'license.txt']
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

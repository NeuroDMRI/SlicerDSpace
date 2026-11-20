"""Read-only discovery and user-private import of a user's FreeSurfer activation file."""

import os
import re
from pathlib import Path

REGISTRATION_URL = 'https://surfer.nmr.mgh.harvard.edu/registration.html'


_EMAIL_TOKEN = re.compile(r'[^\s@]+@[^\s@]+')


def license_problem(path):
    """LE-8: why ``path`` is not a plausible FreeSurfer license file, or an empty string when it looks usable.

    A real license.txt has the registration e-mail on line 1, a number on line 2 and one or two key lines
    (three or four lines in all). Only that shape is checked: FreeSurfer itself validates the activation.
    """
    if path is None:
        return 'No FreeSurfer license file is set.'
    try:
        path = Path(path).expanduser()
        if not path.is_file():
            return f'FreeSurfer license file not found: {path}. Choose the license.txt that FreeSurfer e-mailed to you.'
        if not 0 < path.stat().st_size <= 65536:
            return f'FreeSurfer license file {path} is empty or too large. Choose the license.txt that FreeSurfer e-mailed to you.'
        data = path.read_bytes()
        if b'\0' in data:
            return f'FreeSurfer license file {path} is not a text file. Choose the license.txt that FreeSurfer e-mailed to you.'
        lines = [line.strip() for line in data.decode('utf-8-sig').splitlines() if line.strip()]
    except (OSError, ValueError, TypeError, UnicodeError):
        return f'FreeSurfer license file {path} cannot be read as text. Choose the license.txt that FreeSurfer e-mailed to you.'
    if len(lines) < 3 or not _EMAIL_TOKEN.search(lines[0]):
        return (
            f'{path} does not look like a FreeSurfer license: line 1 must hold your registration e-mail '
            f'and the file needs at least 3 non-empty lines (found {len(lines)}). '
            f'Download license.txt again from {REGISTRATION_URL}.'
        )
    return ''


def usable_license(path):
    """Check readable text in the FreeSurfer license layout; FreeSurfer performs activation validation."""
    return not license_problem(path)


def managed_surface_license_path(*, home=None, environ=None):
    """Return SlicerDSpace's user-private license storage path outside the extension install tree."""
    from shared.user_config import default_user_config_path

    return default_user_config_path(home=home, environ=environ).parent / 'freesurfer' / 'license.txt'


def import_surface_license(source_path, *, home=None, environ=None):
    """Copy a user-supplied license.txt into SlicerDSpace's user-private configuration directory."""
    if not usable_license(source_path):
        raise ValueError('Invalid FreeSurfer license file. ' + license_problem(source_path))
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

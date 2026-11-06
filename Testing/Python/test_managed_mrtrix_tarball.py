"""B2: macOS MRtrix3 tarball profile. A synthetic tarball, a mocked download and mocked subprocess; no network."""

from contextlib import nullcontext
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from shared import managed_runtimes as runtime

COMMANDS = runtime.PROFILES['mrtrix']['commands']
PINNED = dict(runtime.MRTRIX_TARBALL)


def build_tarball(path, extra=(), skip_commands=False, top='mrtrix3'):
    """A tiny archive with the real layout: everything under mrtrix3/."""
    with tarfile.open(path, 'w:gz') as tar:
        folder = tarfile.TarInfo(top)
        folder.type = tarfile.DIRTYPE
        folder.mode = 0o755
        tar.addfile(folder)
        for name in ([] if skip_commands else COMMANDS):
            body = b'#!/bin/sh\necho "== ' + name.encode() + b' 3.0.8 =="\n'
            member = tarfile.TarInfo(f'{top}/bin/{name}')
            member.size, member.mode = len(body), 0o755
            tar.addfile(member, io.BytesIO(body))
        for member, body in extra:
            tar.addfile(member, io.BytesIO(body) if body is not None else None)


def link(name, target, hard=False):
    member = tarfile.TarInfo(name)
    member.type = tarfile.LNKTYPE if hard else tarfile.SYMTYPE
    member.linkname = target
    return member, None


def regular(name, body=b'x'):
    member = tarfile.TarInfo(name)
    member.size = len(body)
    return member, body


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class MrtrixTarballBase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / 'runtimes'
        self.root.mkdir()
        self.source = Path(self.temp.name) / 'source.tar.gz'
        build_tarball(self.source)
        for patcher in (patch.object(runtime.platform, 'system', return_value='Darwin'),
                        patch.object(runtime, 'host_machine', return_value='arm64'),
                        patch.object(runtime.platform, 'mac_ver', return_value=('15.1', ('', '', ''), '')),
                        patch.object(runtime, 'setup_lock', return_value=nullcontext())):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.pin(self.source)

    def pin(self, archive):
        """Make the synthetic archive the pinned download (size and SHA-256 of its bytes)."""
        patcher = patch.dict(runtime.MRTRIX_TARBALL, size=Path(archive).stat().st_size, sha256=sha256(archive))
        patcher.start()
        self.addCleanup(patcher.stop)

    def download(self, archive=None):
        data = Path(archive or self.source).read_bytes()
        return patch.object(runtime.urllib.request, 'urlopen', side_effect=lambda *a, **k: FakeResponse(data))

    def fake_tools(self, version='== mrconvert 3.0.8 =='):
        """mrconvert --version prints a version line; a conversion copies its input (the round-trip check)."""
        def run(argv, **kwargs):
            if argv[-1] == '--version':
                return SimpleNamespace(returncode=0, stdout=version + '\n', stderr='')
            shutil.copy(argv[1], argv[2])
            return SimpleNamespace(returncode=0, stdout='', stderr='')
        return patch.object(runtime.subprocess, 'run', side_effect=run)

    @property
    def prefix(self):
        return self.root / 'mrtrix-3.0.8-macos'


class ProfileTests(MrtrixTarballBase):
    def test_darwin_mrtrix_profile_is_a_tarball_with_the_pinned_artifact(self):
        spec = runtime.profile_spec('mrtrix')
        self.assertEqual((spec['kind'], spec['directory'], spec['packages']), ('tarball', 'mrtrix-3.0.8-macos', ()))
        self.assertEqual(spec['commands'], COMMANDS)
        self.assertEqual(PINNED['url'], 'https://github.com/MRtrix3/mrtrix3/releases/download/3.0.8/mrtrix3-macos-3.0.8.tar.gz')
        self.assertEqual((PINNED['size'], PINNED['sha256']),
                         (64751520, '91fb80ffc9f26136d094e4a5310d051f52db9cb5faa35ad3f3e6156598090c39'))

    def test_linux_keeps_the_conda_profile(self):
        with patch.object(runtime.platform, 'system', return_value='Linux'):
            spec = runtime.profile_spec('mrtrix')
        self.assertNotIn('kind', spec)
        self.assertEqual(spec['directory'], 'mrtrix-3.0.8')
        self.assertIn('mrtrix3=3.0.8=h07fc403_0', spec['packages'])

    def test_plan_names_the_tarball_and_the_new_prefix(self):
        plan = runtime.install_plan(('mrtrix',), self.root)
        self.assertEqual(plan['prefixes']['mrtrix'], str(self.prefix.resolve()))
        self.assertEqual(plan['tarballs']['mrtrix']['url'], PINNED['url'])


class InstallTests(MrtrixTarballBase):
    def test_install_renames_the_top_level_folder_and_writes_the_receipt(self):
        with self.download(), self.fake_tools():
            result = runtime.provision(('mrtrix',), root=self.root)
        self.assertEqual(result['external_dependencies'],
                         {'mrtrix_bin_dir': str(self.prefix.resolve() / 'bin'),
                          'preprocessing_mrtrix_bin_dir': str(self.prefix.resolve() / 'bin')})
        self.assertTrue((self.prefix / 'bin/mrconvert').is_file())
        self.assertTrue(os.access(self.prefix / 'bin/mrconvert', os.X_OK))
        self.assertFalse((self.root / 'downloads/mrtrix3-macos-3.0.8.tar.gz.part').exists())
        self.assertEqual([path.name for path in self.root.glob('.mrtrix-unpack-*')], [])
        receipt = json.loads((self.root / 'mrtrix-3.0.8-macos.json').read_text())
        self.assertEqual((receipt['kind'], receipt['profile'], receipt['kind_detail'], receipt['status']),
                         ('slicerdsurfer_managed_runtime', 'mrtrix', 'tarball', 'ready'))
        self.assertEqual((receipt['url'], receipt['sha256'], receipt['host_machine']),
                         (PINNED['url'], sha256(self.source), 'arm64'))
        self.assertEqual(set(receipt['command_sha256']), set(COMMANDS))
        self.assertEqual(receipt['package_versions'], {'mrtrix3': '3.0.8'})

    def test_sha256_mismatch_stops_before_anything_is_unpacked(self):
        other = Path(self.temp.name) / 'other.tar.gz'
        build_tarball(other, extra=[regular('mrtrix3/extra.txt')])
        with patch.dict(runtime.MRTRIX_TARBALL, size=other.stat().st_size), self.download(other):
            # same size, other bytes: the size check passes, the SHA-256 check must stop it
            with patch.dict(runtime.MRTRIX_TARBALL, sha256='0' * 64):
                result = self.provision_failure()
        self.assertIn('checksum mismatch', result)
        self.assertFalse(self.prefix.exists())
        self.assertEqual(list((self.root / 'downloads').glob('*')), [])

    def test_size_mismatch_stops(self):
        with patch.dict(runtime.MRTRIX_TARBALL, size=runtime.MRTRIX_TARBALL['size'] + 1), self.download():
            self.assertIn('size mismatch', self.provision_failure())
        self.assertFalse(self.prefix.exists())

    def provision_failure(self):
        with self.fake_tools():
            with self.assertRaises(RuntimeError) as context:
                runtime.provision(('mrtrix',), root=self.root)
        return str(context.exception)

    def unsafe(self, extra, top='mrtrix3'):
        archive = Path(self.temp.name) / 'unsafe.tar.gz'
        build_tarball(archive, extra=extra, top=top)
        self.pin(archive)
        with self.download(archive):
            message = self.provision_failure()
        self.assertIn('Unsafe archive member', message)
        self.assertFalse(self.prefix.exists())
        self.assertEqual([path.name for path in self.root.glob('.mrtrix-unpack-*')], [])
        return message

    def test_unsafe_archive_members_stop(self):
        self.unsafe([regular('/etc/escape')])
        self.unsafe([regular('mrtrix3/../../escape')])
        self.unsafe([regular('other/file')])
        self.unsafe([link('mrtrix3/bin/out', '/etc/passwd')])
        self.unsafe([link('mrtrix3/bin/out', '../../../../etc/passwd')])
        self.unsafe([link('mrtrix3/out', '../outside')])
        self.unsafe([link('mrtrix3/bin/out', '../../../outside', hard=True)])
        fifo = tarfile.TarInfo('mrtrix3/fifo')
        fifo.type = tarfile.FIFOTYPE
        self.unsafe([(fifo, None)])

    def test_python_without_a_tar_filter_refuses_to_unpack(self):
        """R-N1 (guard): no lexical-only fallback; the archive is not opened and nothing is written."""
        without_filter = SimpleNamespace(open=tarfile.open)  # a tarfile module that has no data_filter
        destination = Path(self.temp.name) / 'unpacked'
        destination.mkdir()
        with patch.object(runtime, 'tarfile', without_filter), self.assertRaisesRegex(RuntimeError, 'no tarfile extraction filter'):
            runtime._safe_extract(self.source, destination, 'mrtrix3')
        self.assertEqual(list(destination.iterdir()), [])
        with patch.object(runtime, 'tarfile', without_filter), self.download(), self.fake_tools():
            self.assertIn('no tarfile extraction filter', self.provision_failure())
        self.assertFalse(self.prefix.exists())

    def test_links_inside_the_folder_are_allowed(self):
        archive = Path(self.temp.name) / 'links.tar.gz'
        build_tarball(archive, extra=[link('mrtrix3/bin/mrcat', 'mrconvert'), link('mrtrix3/lib/libx', '../bin/mrconvert')])
        self.pin(archive)
        with self.download(archive), self.fake_tools():
            runtime.provision(('mrtrix',), root=self.root)
        self.assertTrue((self.prefix / 'bin/mrcat').is_symlink())

    def test_archive_without_one_top_level_folder_stops(self):
        archive = Path(self.temp.name) / 'flat.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            member, body = regular('mrtrix3x')
            tar.addfile(member, io.BytesIO(body))
        self.pin(archive)
        with self.download(archive):
            self.assertIn('Unsafe archive member', self.provision_failure())

    def test_missing_command_stops_verification(self):
        archive = Path(self.temp.name) / 'incomplete.tar.gz'
        build_tarball(archive, skip_commands=True)
        self.pin(archive)
        with self.download(archive):
            self.assertIn('Runtime is incomplete', self.provision_failure())

    def test_wrong_version_stops_verification(self):
        with self.download(), self.fake_tools(version='== mrconvert 3.0.7 =='):
            with self.assertRaisesRegex(RuntimeError, r'does not report 3\.0\.8'):
                runtime.provision(('mrtrix',), root=self.root)

    def test_version_text_with_a_longer_number_is_not_the_pin(self):
        with self.download(), self.fake_tools(version='== mrconvert 13.0.81 =='):
            with self.assertRaisesRegex(RuntimeError, r'does not report 3\.0\.8'):
                runtime.provision(('mrtrix',), root=self.root)

    def test_verify_prefix_tarball_branch_needs_no_conda_meta(self):
        with self.download(), self.fake_tools():
            runtime.provision(('mrtrix',), root=self.root)
            record = runtime.verify_prefix('mrtrix', self.prefix)
        self.assertFalse((self.prefix / 'conda-meta').exists())
        self.assertEqual(record['functional_checks'], ['mrconvert:synthetic_image_roundtrip'])

    def test_reuse_is_verified_and_skips_download(self):
        with self.download(), self.fake_tools():
            runtime.provision(('mrtrix',), root=self.root)
        with patch.object(runtime.urllib.request, 'urlopen') as download, self.fake_tools():
            result = runtime.provision(('mrtrix',), root=self.root)
        download.assert_not_called()
        self.assertEqual(result['status'], 'completed')

    def test_reuse_with_a_changed_command_checksum_stops(self):
        with self.download(), self.fake_tools():
            runtime.provision(('mrtrix',), root=self.root)
        (self.prefix / 'bin/tckgen').write_text('#!/bin/sh\necho changed\n')
        with self.fake_tools(), self.assertRaisesRegex(RuntimeError, 'checksum changed'):
            runtime.provision(('mrtrix',), root=self.root)

    def test_unmanaged_folder_is_never_replaced(self):
        self.prefix.mkdir(parents=True)
        (self.prefix / 'keep.txt').write_text('mine')
        with self.download(), self.fake_tools(), self.assertRaisesRegex(RuntimeError, 'unmanaged environment'):
            runtime.provision(('mrtrix',), root=self.root)
        self.assertEqual((self.prefix / 'keep.txt').read_text(), 'mine')

    def test_symlink_prefix_is_refused(self):
        target =Path(self.temp.name) / 'elsewhere'
        target.mkdir()
        self.prefix.symlink_to(target, target_is_directory=True)
        with self.download(), self.fake_tools(), self.assertRaisesRegex(ValueError, 'symlink'):
            runtime.provision(('mrtrix',), root=self.root)

    def test_repair_keeps_the_previous_folder_aside(self):
        with self.download(), self.fake_tools():
            runtime.provision(('mrtrix',), root=self.root)
            (self.prefix / 'marker.txt').write_text('old')
            runtime.provision(('mrtrix',), root=self.root, repair=True)
        aside = list(self.root.glob('mrtrix-3.0.8-macos.replaced-*'))
        self.assertEqual(len(aside), 1)
        self.assertEqual((aside[0] / 'marker.txt').read_text(), 'old')
        self.assertFalse((self.prefix / 'marker.txt').exists())


class MigrationTests(MrtrixTarballBase):
    def old_conda_prefix(self):
        old = self.root / 'mrtrix-3.0.8'
        (old / 'bin').mkdir(parents=True)
        (old / 'bin/mrconvert').write_text('old conda binary')
        (self.root / 'mrtrix-3.0.8.json').write_text(json.dumps(
            {'kind': 'slicerdsurfer_managed_runtime', 'profile': 'mrtrix', 'status': 'ready'}))
        return old

    def install(self, config):
        with self.download(), self.fake_tools():
            return runtime.provision(('mrtrix',), root=self.root, user_config=config)

    def test_migration_repoints_only_paths_into_the_old_managed_prefix(self):
        old = self.old_conda_prefix()
        config = {'external_dependencies': {'mrtrix_bin_dir': str(old / 'bin'),
                                            'preprocessing_mrtrix_bin_dir': '/opt/my-mrtrix/bin'}}
        result = self.install(config)
        new_bin = str(self.prefix.resolve() / 'bin')
        self.assertEqual(result['external_dependencies'], {'mrtrix_bin_dir': new_bin})
        self.assertEqual(result['migration']['repointed'], ['mrtrix_bin_dir'])
        self.assertEqual(result['migration']['kept_user_paths'], ['preprocessing_mrtrix_bin_dir'])
        self.assertIn('can be deleted by hand', result['migration']['note'])

    def test_unset_keys_are_set_and_the_old_prefix_is_never_deleted(self):
        old = self.old_conda_prefix()
        result = self.install({'external_dependencies': {}})
        self.assertEqual(set(result['external_dependencies']), {'mrtrix_bin_dir', 'preprocessing_mrtrix_bin_dir'})
        self.assertEqual((old / 'bin/mrconvert').read_text(), 'old conda binary')
        self.assertTrue((self.root / 'mrtrix-3.0.8.json').is_file())

    def test_no_old_prefix_means_no_migration_entry(self):
        result = self.install({'external_dependencies': {'mrtrix_bin_dir': '/opt/my-mrtrix/bin'}})
        self.assertNotIn('migration', result)
        self.assertIn('mrtrix_bin_dir', result['external_dependencies'])

    def test_old_folder_without_a_receipt_is_not_treated_as_managed(self):
        (self.root / 'mrtrix-3.0.8/bin').mkdir(parents=True)
        result = self.install({'external_dependencies': {'mrtrix_bin_dir': str(self.root / 'mrtrix-3.0.8/bin')}})
        self.assertNotIn('migration', result)


if __name__ == '__main__':
    unittest.main()

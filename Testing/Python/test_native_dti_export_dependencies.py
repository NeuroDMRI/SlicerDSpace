"""Preparation must cover the format conversion used by native DTI outputs."""
from pathlib import Path
import unittest
from shared.environment import EnvironmentDoctorConfig,build_dependency_specs

class NativeDTIExportDependencyTest(unittest.TestCase):
    def test_tensor_export_routes_declare_installable_nrrd_reader(self):
        for target in ('microstructure','microstructure_freewater','microstructure_dki','microstructure_noddi','tractography'):
            with self.subTest(target=target):
                config=EnvironmentDoctorConfig(targets=(target,),repo_root=Path(__file__).resolve().parents[2],tractography_backend='dti_det')
                deps=build_dependency_specs(config)
                matches=[d for d in deps if d.python_import=='nrrd']
                self.assertEqual(len(matches),1)
                self.assertEqual(matches[0].python_install_requirement,'pynrrd')
                self.assertIn(target,matches[0].targets)

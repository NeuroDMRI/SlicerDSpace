"""Native bundle reading and fresh native Teem DTI timing on real, isolated data."""

from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import qt
import slicer
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def run():
    destination = Path(os.environ['SLICERDSPACE_SMOKE_ROOT'])
    try:
        sys.path.insert(0, os.environ['SLICERDSPACE_EXTENSION_MODULE_DIR'])
        from shared.module_logic import SharedLogicMixin
        from shared.pipeline import find_subject_root, ensure_native_dti_bundle
        configuration = json.loads(Path(os.environ['SLICERDSPACE_TASKS_CONFIG']).read_text())
        class ReaderLogic(SharedLogicMixin):
            log = staticmethod(print)
        logic = ReaderLogic()
        records = []
        for case in configuration.get('qc_rejection_cases', []):
            from shared.auto_results import automatic_result_candidates
            from shared.result_scene import recorded_qc_summary
            task = dict(name=case['product'], status='completed', qc_path=case['qc_path'])
            manifest = {'tasks': [task]}
            summary = recorded_qc_summary(manifest)
            selected, skipped, _ = automatic_result_candidates(manifest)
            assert summary['issues'] == 1 and not selected and skipped, (case, summary, selected)
            records.append(dict(module='slicer.qc_rejection.' + case['product'], status='completed',
                                qc_path=case['qc_path'], scope='Legacy green execution rejected by current scientific-QC/result-selection policy'))
            manifest_path = destination / ('rejected_' + case['product'] + '.json')
            manifest_path.write_text(json.dumps(manifest))
            # Use a real bundle as the previously loaded preview. These are
            # failure-path scene tests, not a reclassification of its anatomy.
            source = Path(configuration['cases'][0]['files'][0])
            name = 'FailedPreview_' + case['product']
            assert logic._load_fiber_bundle_node(source, name)
            automatic = slicer.util.getNode(name)
            automatic.SetAttribute('SlicerDDSurfer.Run', str(manifest_path))
            automatic.SetAttribute('SlicerDDSurfer.Product', case['product'])
            automatic.SetAttribute('SlicerDDSurfer.SourcePath', str(source))
            automatic.SetAttribute('SlicerDDSurfer.AutomaticLoaded', 'true')
            identity = automatic.GetID()
            manual = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', 'ManualReviewData')
            manual.SetAttribute('SlicerDDSurfer.Run', str(manifest_path))
            manual.SetAttribute('SlicerDDSurfer.Product', case['product'])
            manual.SetAttribute('SlicerDDSurfer.SourcePath', str(source))
            manual_id = manual.GetID()
            result = logic.loadKeyResultsFromManifest(manifest_path, automatic=True)
            assert not result['loaded'] and result['skipped']
            assert slicer.mrmlScene.GetNodeByID(identity) is None
            assert slicer.mrmlScene.GetNodeByID(manual_id) is manual
            slicer.mrmlScene.RemoveNode(manual)
            records.append(dict(module='slicer.qc_retraction.' + case['product'], status='completed',
                                automatic_node_removed=True, manual_node_preserved=True,
                                scope='Real Slicer failed-QC retraction without a per-file rejection path'))
        for case in configuration.get('qc_acceptance_cases', []):
            from shared.auto_results import automatic_result_candidates
            from shared.result_scene import recorded_qc_summary
            task = dict(name=case['product'], status='completed', qc_path=case['qc_path'])
            manifest = {'tasks': [task]}
            summary = recorded_qc_summary(manifest)
            selected, _, _ = automatic_result_candidates(manifest)
            assert summary['passed'] == 1 and summary['issues'] == 0 and len(selected) == case['expected_files']
            records.append(dict(module='slicer.qc_acceptance.' + case['product'], status='completed',
                                qc_path=case['qc_path'], selected_files=len(selected)))
            rejection = next((item for item in configuration.get('qc_rejection_cases', [])
                              if item['product'] == case['product']), None)
            if rejection:
                manifest_path = destination / ('manual_ownership_' + case['product'] + '.json')
                manifest_path.write_text(json.dumps(manifest))
                manual = logic.loadKeyResultsFromManifest(manifest_path)
                identities = [identity for entry in manual['loaded'] for identity in entry['node_ids']]
                assert identities, case
                repeated = logic.loadKeyResultsFromManifest(manifest_path, automatic=True)
                assert not repeated['loaded'] and repeated['reused']
                assert all(slicer.mrmlScene.GetNodeByID(identity).GetAttribute('SlicerDDSurfer.AutomaticLoaded') != 'true'
                           for identity in identities)
                task['qc_path'] = rejection['qc_path']
                manifest_path.write_text(json.dumps(manifest))
                logic.loadKeyResultsFromManifest(manifest_path, automatic=True)
                assert all(slicer.mrmlScene.GetNodeByID(identity) is not None for identity in identities)
                logic.clearResultsFromManifest(manifest_path)
                records.append(dict(module='slicer.qc_manual_ownership.' + case['product'], status='completed',
                                    manual_nodes_preserved=len(identities),
                                    scope='Automatic reuse never claims manual-node ownership; later failed QC preserves those nodes'))
        for case in configuration['cases']:
            paths = [Path(p) for p in case['files']]
            started = time.monotonic()
            lines = 0
            maximum_delta = 0.
            for path in paths:
                if path.suffix in {'.trk', '.tck'}:
                    import nibabel as nib
                    tracks = list(nib.streamlines.load(str(path)).streamlines)
                    expected_lines = len(tracks)
                    original_points = np.concatenate(tracks)
                else:
                    reader = vtk.vtkXMLPolyDataReader() if path.suffix == '.vtp' else vtk.vtkPolyDataReader()
                    reader.SetFileName(str(path))
                    reader.Update()
                    original = reader.GetOutput()
                    expected_lines = original.GetNumberOfLines()
                    original_points = vtk_to_numpy(original.GetPoints().GetData())
                assert logic._load_fiber_bundle_node(path, path.stem), str(path)
                node = slicer.util.getNode(path.stem)
                assert node and node.IsA('vtkMRMLFiberBundleNode'), str(path)
                loaded = node.GetPolyData()
                assert loaded.GetNumberOfLines() == expected_lines > 0
                actual_points = vtk_to_numpy(loaded.GetPoints().GetData())
                assert actual_points.shape == original_points.shape, str(path)
                delta = float(np.max(np.abs(actual_points.astype(float) - original_points.astype(float))))
                maximum_delta = max(maximum_delta, delta)
                assert delta <= .001, (str(path), delta)
                lines += loaded.GetNumberOfLines()
                slicer.mrmlScene.RemoveNode(node)
            records.append(dict(module='slicer.load.' + case['backend'], status='completed',
                                subject_id=case['subject_id'], runtime_seconds=time.monotonic() - started,
                                files=len(paths), streamlines=lines, point_coordinates_exact=maximum_delta == 0.,
                                maximum_coordinate_delta_mm=maximum_delta, coordinate_tolerance_mm=.001,
                                scope='SlicerDSpace FiberBundle loader plus independent reader comparison; excludes Slicer startup and tract extraction'))
        if configuration.get('preview_cst'):
            slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
            bounds = []
            for index, path in enumerate(configuration['preview_cst']):
                name = 'QC_CST_' + str(index)
                assert logic._load_fiber_bundle_node(Path(path), name)
                bundle = slicer.util.getNode(name)
                bounds.append(bundle.GetPolyData().GetBounds())
                display = bundle.GetDisplayNode()
                display.SetVisibility(True)
                display.SetScalarVisibility(False)
                display.SetColor(*((.24, .46, .65) if index == 0 else (.78, .44, .25)))
                display.SetOpacity(.8)
            view = slicer.app.layoutManager().threeDWidget(0).threeDView()
            node = view.mrmlViewNode()
            node.SetBackgroundColor(1, 1, 1)
            node.SetBackgroundColor2(1, 1, 1)
            node.SetBoxVisible(False)
            node.SetAxisLabelsVisible(False)
            node.SetOrientationMarkerType(0)
            slicer.util.resetThreeDViews()
            camera = slicer.modules.cameras.logic().GetViewActiveCameraNode(node).GetCamera()
            minimum = np.asarray(bounds)[:, (0, 2, 4)].min(axis=0)
            maximum = np.asarray(bounds)[:, (1, 3, 5)].max(axis=0)
            center = (minimum + maximum) / 2
            camera.SetFocalPoint(*center)
            camera.ParallelProjectionOn()
            camera.SetParallelScale(float(np.max(maximum - minimum) * .65))
            for name, direction in (('front', (0, -1, .05)), ('oblique', (.8, -.6, .2))):
                camera.SetPosition(*[center[i] + direction[i] * 500 for i in range(3)])
                camera.SetViewUp(0, 0, 1)
                camera.SetClippingRange(1, 2000)
                slicer.app.processEvents()
                view.renderWindow().Render()
                capture = vtk.vtkWindowToImageFilter()
                capture.SetInput(view.renderWindow())
                capture.ReadFrontBufferOff()
                capture.Update()
                pixels = vtk_to_numpy(capture.GetOutput().GetPointData().GetScalars())
                nonwhite = float(np.mean(np.any(pixels[:, :3] < 245, axis=1)))
                assert nonwhite > .002, ('Blank CST preview', name, nonwhite)
                writer = vtk.vtkPNGWriter()
                writer.SetFileName(str(destination / f'cst_{name}.png'))
                writer.SetInputConnection(capture.GetOutputPort())
                writer.Write()
                records.append(dict(module='slicer.preview.cst.' + name, status='completed',
                                    nonwhite_pixel_fraction=nonwhite, path=str(destination / f'cst_{name}.png'),
                                    scope='Rendered real streamlines; no expert anatomical certification'))
        if not configuration.get('skip_fresh_dti'):
            raw = find_subject_root(Path(configuration['subject_root']), '100307', generate_missing_mask=False)
            raw = replace(raw, subject_root=destination / 'fresh_dti')
            started = time.monotonic()
            bundle = ensure_native_dti_bundle(raw, fit_method='wls', reuse=False, log=print)
            seconds = time.monotonic() - started
            import nibabel as nib
            image = nib.load(str(bundle.fa_nii_path))
            source = nib.load(str(raw.dwi_path))
            fa = np.asarray(image.dataobj)
            assert image.shape == source.shape[:3] and np.allclose(image.affine, source.affine, atol=1e-4)
            assert np.isfinite(fa).all() and fa.min() >= -1e-5 and fa.max() <= 1.00001
            records.append(dict(module='microstructure.native_dti_fresh', subject_id='100307', status='completed',
                                runtime_seconds=seconds, shape=list(source.shape), source=str(raw.dwi_path),
                                source_record=str(bundle.bundle_dir / 'native_dti_timing.json'),
                                scope='Fresh real full-volume native Teem WLS fitting and six scalar conversions, no cache reuse'))
        result = dict(status='passed', records=records, slicer_version=slicer.app.applicationVersion,
                      scientific_scope='Geometry/integrity and scoped elapsed measurements; not anatomical certification')
        code = 0
    except Exception:
        result = dict(status='failed', error=traceback.format_exc())
        code = 2
    (destination / 'bundle_products.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    slicer.app.exit(code)


qt.QTimer.singleShot(3000, run)

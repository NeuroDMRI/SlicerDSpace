#!/usr/bin/env python3
"""Fresh complete DKI calls on explicitly labeled real native-grid ROIs.

No original image is edited. ROI timings must never be scaled to full brain.
Both routes run unchanged DIPY WLS; only metric evaluation granularity differs.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import shutil
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    import dipy
    import nibabel as nib
    import numpy as np
    from microstructure.dki import DkiConfig,run_dki
    from shared.diffusion_io import diffusion_image
    from shared.pipeline import RawInputBundle

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--side',type=int,default=24)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    if args.repeats<1 or args.side<2:
        parser.error('Positive repeat count and ROI side >= 2 required')
    args.output.mkdir(parents=True,exist_ok=False)
    report=args.output/'measurements.json'
    record=dict(status='running',scope='Complete WLS DKI stage on fixed real native-grid ROIs, not full brain',
                implementation='Unchanged DIPY WLS and metrics; array-level evaluation only',
                environment=dict(python=sys.version,numpy=np.__version__,dipy=dipy.__version__,platform=platform.platform()),
                implementation_sha256={name:digest(ROOT/name) for name in ('microstructure/dki.py','scripts/benchmark_dki_dense_metrics.py')},
                repeats=args.repeats,roi_side=args.side,cases=[])
    def save():
        report.write_text(json.dumps(record,indent=2)+'\n')
    save()
    try:
        for source in args.input:
            folder=args.output/source.parent.name
            folder.mkdir()
            inputs=folder/'input'
            inputs.mkdir()
            paths={name:source/name for name in ('data.nii.gz','bvals','bvecs','nodif_brain_mask.nii.gz')}
            before={name:digest(path) for name,path in paths.items()}
            mask_image=nib.load(paths['nodif_brain_mask.nii.gz'])
            mask=np.asanyarray(mask_image.dataobj)
            coordinates=np.argwhere(mask>0)
            center=np.rint((coordinates.min(axis=0)+coordinates.max(axis=0))/2).astype(int)
            start=np.maximum(0,np.minimum(center-args.side//2,np.asarray(mask.shape)-args.side))
            stop=np.minimum(start+args.side,mask.shape)
            selection=tuple(slice(int(lo),int(hi)) for lo,hi in zip(start,stop))
            with diffusion_image(paths['data.nii.gz'],folder) as image:
                data=image.dataobj[selection+(slice(None),)]
                affine=image.affine.copy()
                affine[:3,3]=(image.affine@np.r_[start,1])[:3]
                nib.save(nib.Nifti1Image(np.asarray(data),affine),inputs/'data.nii.gz')
            nib.save(nib.Nifti1Image(mask[selection],affine),inputs/'nodif_brain_mask.nii.gz')
            for name in ('bvals','bvecs'):
                shutil.copyfile(source/name,inputs/name)
            raw=RawInputBundle(folder,source.parent.name,inputs,inputs/'data.nii.gz',inputs/'bvals',inputs/'bvecs',inputs/'nodif_brain_mask.nii.gz')
            timings={'reference':[],'optimized':[]}
            expected=None
            checks=[]
            qc_counts=[]
            for repeat in range(args.repeats):
                order=('reference','optimized') if repeat%2==0 else ('optimized','reference')
                for method in order:
                    output=folder/(method+str(repeat))
                    began=time.perf_counter()
                    result=run_dki(DkiConfig(raw.subject_id,raw,output,force_recompute=True,
                                            dense_metrics=method=='optimized'))
                    elapsed=time.perf_counter()-began
                    timings[method].append(elapsed)
                    products={}
                    for key,path in result.outputs.items():
                        if key in ('manifest','qc','progress'):
                            continue
                        image=nib.load(path)
                        products[key]=(image.affine.copy(),np.asanyarray(image.dataobj))
                    if expected is None:
                        expected=products
                    exact=all(np.array_equal(expected[key][0],value[0]) and np.array_equal(expected[key][1],value[1],equal_nan=True) for key,value in products.items())
                    checks.append(exact)
                    qc=json.loads(result.qc_path.read_text())
                    qc_counts.append((qc['fitted_voxels'],qc['valid_voxels'],qc['invalid_voxels']))
                    if not exact:
                        raise ValueError('Dense metric arrays differ; optimization not accepted')
                    print(source.parent.name,method,repeat,elapsed,flush=True)
            unchanged=before=={name:digest(path) for name,path in paths.items()}
            if not unchanged or len(set(qc_counts))!=1:
                raise ValueError('Source or QC decisions changed')
            seconds={name:statistics.median(values) for name,values in timings.items()}
            record['cases'].append(dict(subject=raw.subject_id,source=str(source.resolve()),source_sha256=before,
                source_unchanged=unchanged,start_ijk=start.tolist(),stop_ijk=stop.tolist(),mask_voxels=int(mask[selection].sum()),
                fitted_voxels=qc_counts[0][0],valid_voxels=qc_counts[0][1],all_output_arrays_exact=all(checks),
                raw_seconds=timings,median_seconds=seconds,speed_ratio=seconds['reference']/seconds['optimized'],
                seconds_saved=seconds['reference']-seconds['optimized']))
            save()
        record['status']='passed'
    except BaseException as error:
        record.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        save()


if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""Freeze a scheduler-only candidate without replacing active scientific code."""

import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.benchmark_cohort_workflows import save, sha

SCHEDULER_FILES=('shared/cohort_schedule.py','shared/cohort_policy.py','shared/cohort_profile.py',
                 'shared/cohort_planning.py','shared/cohort_resources.py','shared/compute_schedule.py')
CACHE_MARKER_FILES=('shared/pipeline.py','microstructure/pipeline.py')


def cache_normalised_ast(path):
    class Marker(ast.NodeTransformer):
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if isinstance(node.func,ast.Name) and node.func.id=='mark_reused':
                node.args=[]
                node.keywords=[]
            return node
    return ast.dump(Marker().visit(ast.parse(path.read_text())),include_attributes=False)


def freeze(base, destination, code_root=ROOT):
    for name in CACHE_MARKER_FILES:
        if cache_normalised_ast(base/name)!=cache_normalised_ast(code_root/name):
            raise ValueError('Scientific code changed beyond cache-marker arguments: '+name)
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(base,destination,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in (*SCHEDULER_FILES,*CACHE_MARKER_FILES):
        shutil.copy2(code_root/name,destination/name)
    old={str(path.relative_to(base)):sha(path) for path in base.rglob('*') if path.is_file() and '__pycache__' not in path.parts}
    current={str(path.relative_to(destination)):sha(path) for path in destination.rglob('*') if path.is_file()}
    changed=sorted(key for key,value in current.items() if old.get(key)!=value)
    if set(changed)-set(SCHEDULER_FILES)-set(CACHE_MARKER_FILES) or old.keys()-current.keys():
        raise ValueError('Unexpected non-scheduler change in candidate')
    core=[key for key in old if key.startswith(('microstructure/','anatomy/','tractography/','connectome/','preprocessing/'))
          and key not in CACHE_MARKER_FILES]
    return dict(snapshot=str(destination),snapshot_hashes=current,changed_files=changed,
                byte_identical_scientific_files=len(core),scientific_files_sha256={name:old[name] for name in core},
                cache_marker_ast_equal=True,base_snapshot=str(base))


def campaign(source, output, candidate, *, route, reuse_reference=False):
    record=deepcopy(json.loads(source.read_text()))
    rows=[deepcopy(row) for row in record['trials'] if row['route']==route and row['policy'] in ('subject','adaptive')]
    if len(rows)!=2 or {row['policy'] for row in rows}!={'subject','adaptive'}:
        raise ValueError('A matched subject/adaptive template is required')
    for row in rows:
        if reuse_reference and row['policy']=='subject':
            if row['status']!='completed' or row.get('qc_status')!='passed':
                raise ValueError('Only a completed independently audited reference can be reused')
            row['inherited_evidence']={name:sha(Path(row['output'])/name)
                for name in ('independent_qc.json','scientific_fingerprints.json')}
        else:
            row={key:row[key] for key in ('id','route','policy','threads','repeat')}
            row.update(id=row['id']+'-mature-v2',status='pending',snapshot=candidate['snapshot'],
                       snapshot_hashes=candidate['snapshot_hashes'])
        if row['policy']=='subject':
            reference=row
        else:
            adaptive=row
    for key in ('driver_pid','started_at','ended_at','inputs_unchanged','execution_harness'):
        record.pop(key,None)
    record.update(status='prepared',trials=[reference,adaptive],reference_policy='subject',
                  candidate_provenance=candidate,selected_from=dict(path=str(source),sha256=sha(source)))
    record['notes'].append('Scheduler-v2 comparison; numerical methods unchanged; completed reference retained when available')
    output.mkdir(parents=True,exist_ok=False)
    save(output/'campaign.json',record)
    return output/'campaign.json'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-campaign',type=Path,required=True)
    parser.add_argument('--long-campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=json.loads(args.base_campaign.read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    candidate=freeze(Path(source['snapshot']),args.output/'candidate_runtime')
    save(args.output/'candidate_provenance.json',candidate)
    paths=[campaign(args.base_campaign,args.output/'dki-v2',candidate,route='dki',reuse_reference=True),
           campaign(args.long_campaign,args.output/'tracts-v2',candidate,route='tracts'),
           campaign(args.long_campaign,args.output/'mkcurve-v2',candidate,route='mkcurve')]
    print(json.dumps(dict(status='prepared',campaigns=[str(path) for path in paths],
                          changed_files=candidate['changed_files'],
                          unchanged_scientific_files=candidate['byte_identical_scientific_files']),indent=2))


if __name__=='__main__':
    main()

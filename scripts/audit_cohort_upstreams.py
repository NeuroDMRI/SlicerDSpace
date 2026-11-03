#!/usr/bin/env python3
"""Record official workflow references and verify the Nipype resource-sort contract."""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request


def fetch(url):
    request=urllib.request.Request(url,headers={'User-Agent':'SlicerDSpace-upstream-review'})
    return urllib.request.urlopen(request,timeout=30).read()


def methods(source):
    tree=ast.parse(source)
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='MultiProcPlugin')
    return {node.name:node for node in cls.body if isinstance(node,ast.FunctionDef)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Existing upstream evidence is immutable')
    api='https://api.github.com/repos/nipy/nipype'
    release=json.loads(fetch(api+'/releases/latest'))
    main=json.loads(fetch(api+'/commits/master'))
    sources=[]
    for ref in (release['tag_name'],main['sha']):
        url=f'https://raw.githubusercontent.com/nipy/nipype/{ref}/nipype/pipeline/plugins/multiproc.py'
        raw=fetch(url)
        parsed=methods(raw.decode())
        sources.append(dict(ref=ref,url=url,sha256=hashlib.sha256(raw).hexdigest(),
            selected_methods={name:hashlib.sha256(ast.dump(parsed[name],include_attributes=False).encode()).hexdigest()
                              for name in ('_prerun_check','_check_resources','_sort_jobs')},
            sort_contract=ast.unparse(parsed['_sort_jobs'])))
    contract='(self.procs[item].mem_gb, self.procs[item].n_procs)'
    if any(contract not in row['sort_contract'] for row in sources):
        raise ValueError('Upstream sort contract changed; review before adapting')
    report=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),
        latest_release=release['tag_name'],release_published_at=release['published_at'],sources=sources,
        references=[
            dict(project='Pydra',url='https://nipype.github.io/pydra/tutorial/2-advanced-execution.html',
                 mechanism='ConcurrentFutures workers; separate execution from task parameters'),
            dict(project='Pydra',url='https://nipype.github.io/pydra/explanation/hashing-caching.html',
                 mechanism='Task input identity, isolated outputs and resumable completed results'),
            dict(project='Slurm',url='https://slurm.schedmd.com/sched_config.html',
                 mechanism='Protect the head expected start while backfilling'),
            dict(project='Python',url='https://docs.python.org/3.12/library/graphlib.html',
                 mechanism='Directly reused TopologicalSorter for model readiness'),
            dict(project='Python',url='https://docs.python.org/3.12/library/concurrent.futures.html',
                 mechanism='Directly reused futures completion notifications')],
        scope='Official source audit and selected mechanism adaptation; Nipype/Pydra/Slurm engines are not runtime dependencies; no speedup measurement')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('status','latest_release','scope')},indent=2))


if __name__=='__main__':
    main()

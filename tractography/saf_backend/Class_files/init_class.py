import os
import re
from copy import deepcopy

from Support.make_full_path import make_full_path
from Support.Supp_gz_fileparts import Supp_gz_fileparts
from Support.verb_msg import verb_msg
from Support.sys_exec import sys_exec
from Support.shell_quote import shell_quote


class init_class:
    def __init__(self, parsed_input):
        self.fod = ''
        self.fs_dir = ''
        self.work_dir = ''
        self.logt = []
        self.slab = 'cortex'
        self.surfseed = False
        self.tckgen = ''
        self.noseed = ''
        self.sl_weights = ''
        self.maxlen = 90
        self.final_init = ''
        self.remesh_seeds = False
        self.notrunc = False
        self.fa = ''
        self.wrks = ''
        self.optmodules = []
        self.scalar2surf = []
        self.fastx = True
        self.pialfilter = False
        self.gwg_hard = False
        self.gg_margin = 0
        self.mrtrix_prefix = ''
        self.mrtrix_bin_dir = ''
        self.outf = -1
        self.force = False
        self.dgn = False
        self.quiet = False
        self.version = '1.0.0'

        arguments_bool = {'dgn', 'notrunc', 'force', 'quiet', 'pialfilter', 'remesh_seeds', 'gwg_hard', 'surfseed', 'fastx'}
        arguments_multi = {'optmodules', 'scalar2surf'}
        arguments_path = {'fod', 'fa', 'noseed', 'sl_weights'}

        args = list(parsed_input)
        while args:
            if any(isinstance(a, init_class) for a in args):
                idx = [i for i, a in enumerate(args) if isinstance(a, init_class)][0]
                old = args.pop(idx)
                self.__dict__.update(deepcopy(old.__dict__))
                self.logt = []
                self.final_init = ''
                self.force = True
                continue

            iname = args.pop(0)
            if iname not in self.__dict__:
                raise ValueError('wrong input provided')
            ival = []
            while args and (not isinstance(args[0], str) or args[0] not in self.__dict__):
                val = args.pop(0)
                if isinstance(val, list):
                    ival.extend(val)
                else:
                    ival.append(val)

            if iname in arguments_path:
                ival = make_full_path(ival)

            if len(ival) == 0:
                if iname not in arguments_bool:
                    raise ValueError(f'{iname} requires values to proceed')
                setattr(self, iname, True)
            elif len(ival) == 1:
                val = ival[0]
                if iname in arguments_bool:
                    if val in ('on', 1):
                        val = True
                    elif val in ('off', 0):
                        val = False
                if iname in arguments_multi:
                    getattr(self, iname).append(val)
                else:
                    setattr(self, iname, val)
            else:
                if iname not in arguments_multi:
                    raise ValueError(f'{iname} does not take multiple values')
                getattr(self, iname).extend(ival)

        for mvi in arguments_multi:
            if isinstance(getattr(self, mvi), list) and any(x == '' for x in getattr(self, mvi)):
                setattr(self, mvi, [])

        self._workdir()
        self._sortFS()
        self._n_wrks()
        self._start_log()

        if not self.fod or not self.fs_dir:
            raise AssertionError('FOD file and FreeSurfer dir must be provided')
        if self.tckgen and self.noseed:
            raise AssertionError('Either "tckgen" or "noseed" must be empty')

        if self.dgn:
            self.quiet = False

        if self.maxlen is not None and self.tckgen:
            m = re.search(r'-maxlength\s+(\d+)', self.tckgen)
            if m:
                tckgen_maxlen = int(m.group(1))
                if tckgen_maxlen < self.maxlen:
                    self.maxlen = None

    def _n_wrks(self):
        if not self.wrks:
            self.wrks = os.cpu_count() or 1
        elif isinstance(self.wrks, str):
            self.wrks = int(self.wrks)

    def _workdir(self):
        if self.outf == -1:
            self.work_dir = os.path.join(Supp_gz_fileparts(self.fod)[0], 'SAF_output')
        else:
            if not os.path.dirname(str(self.outf)):
                self.work_dir = os.path.join(Supp_gz_fileparts(self.fod)[0], str(self.outf))
            else:
                self.work_dir = str(self.outf)
        os.makedirs(self.work_dir, exist_ok=True)
        os.chdir(self.work_dir)

    def _start_log(self):
        fa = self.fa if self.fa else 'not provided'
        self.logt = [
            ['Log for Python SAF_pipeline.py run on:', _now_str()],
            ['Version:', self.version],
            ['', ''],
            ['FreeSurfer directory:', self.fs_dir],
            ['FOD file:', self.fod],
            ['FA file:', fa],
            ['Output directory:', self.work_dir],
            ['', ''],
        ]
        self.logt.append(['Surface space:', 'native dMRI'])
        if not self.noseed:
            self.logt.extend([
                ['Seed from surface mesh vertices for tractography:', self.surfseed],
                ['Surface label for seeding:', self.slab],
                ['Additional tckgen options:', self.tckgen],
            ])
        else:
            self.logt.append(['Tractography file:', self.noseed])
        if not self.sl_weights:
            self.logt.append(['Streamline weights:', 'not provided'])
        else:
            self.logt.append(['Streamline weights:', self.sl_weights])
        self.logt.extend([
            ['MRtrix command prefix:', self.mrtrix_prefix],
            ['MRtrix bin directory:', self.mrtrix_bin_dir],
            ['Fast mesh-streamline intersection detection:', self.fastx],
            ['Extra margin during GG:', self.gg_margin],
            ['Hard 2-point intersection during GWG:', self.gwg_hard],
            ['Do not truncate streamline at the WSM:', self.notrunc],
            ['Apply pial filtering:', self.pialfilter],
            ['Diagnostics:', self.dgn],
            ['Force overwrite:', self.force],
            ['Quiet:', self.quiet],
            ['', ''],
        ])
        if not self.scalar2surf:
            self.logt.append(['Scalars provided for projection on the surface:', 'none'])
        else:
            self.logt.append(['Scalars provided for projection on the surface:', ', '.join(self.scalar2surf)])
        if not self.optmodules:
            self.logt.append(['Optional modules:', 'none'])
        else:
            self.logt.append(['Optional modules:', ', '.join(self.optmodules)])
        self.logt.append(['', ''])

    def _sortFS(self):
        def_dir = os.path.join(Supp_gz_fileparts(self.fod)[0], 'FS_reconall')

        if not os.path.exists(self.fs_dir) and not os.path.exists(def_dir):
            raise RuntimeError('fs_dir not provided correctly')
        elif not os.path.exists(self.fs_dir):
            self.fs_dir = def_dir
        elif not os.path.isdir(self.fs_dir) and os.path.exists(def_dir):
            raise RuntimeError('fs_dir points to a file but a default freesurfer directory found. Provide correct directory')
        elif os.path.isdir(self.fs_dir) and os.path.exists(def_dir):
            self.fs_dir = make_full_path(self.fs_dir)
        elif not os.path.isdir(self.fs_dir) and not os.path.exists(def_dir):
            verb_msg(f'Running Freesurfer recon-all on {self.fs_dir}... will take a while ;)', self)
            sys_exec(
                "recon-all -subject FS_reconall "
                f"-sd {shell_quote(Supp_gz_fileparts(self.fod)[0])} "
                f"-i {shell_quote(self.fs_dir)} -all",
                self,
            )
            self.fs_dir = def_dir


def _now_str():
    import datetime
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

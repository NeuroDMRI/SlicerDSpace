import os

from Support.sys_exec import sys_exec


class tx_class:
    def __init__(self, PS):
        self.t1 = os.path.join(PS.work_dir, 'brain_T1.nii.gz')
        self.in_dwi = ''
        self.affine = ''
        self.warp = ''
        self.iwarp = ''

        if not os.path.exists(self.t1):
            brain_mgz = os.path.join(PS.fs_dir, 'mri', 'brain.mgz')
            sys_exec(["mrconvert", brain_mgz, self.t1], PS)

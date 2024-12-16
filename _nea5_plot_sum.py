# -*- coding: utf-8 -*-

import os
import sys
from glob import glob

from imaging_fly import ImagingFly, ImagingFlyGroup
from _0_constants import DEFAULT_DATE, DEFALUT_STIM

""" NOTE:
    plot summary figures
"""

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: plot [fly|geno] stim\n   fly: 221020-M1, geno: CX1013, stim: OL_grating, OL_dot, CL_bar")
        fly_name = DEFAULT_DATE
        stim_name = DEFALUT_STIM
    else:
        fly_name, stim_name = sys.argv[1], sys.argv[2]
    if fly_name.startswith("2"):
        fly = ImagingFly(fly_name, stim_name)
    else:
        fly = ImagingFlyGroup(fly_name, stim_name)
    if fly.load():
        # fly.plot_single()
        fly.plot_summary()

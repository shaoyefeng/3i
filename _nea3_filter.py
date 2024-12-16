# -*- coding: utf-8 -*-

import os
import sys
import shutil
from glob import glob

from _0_constants import EXP_TYPE, SKIP_PLOT, DEFAULT_DATE
from _0_function_analysis import cmd_to_time_glob, get_ext_exp_type
from _0_function_imaging_exp import ImagingTrial

""" NOTE:
    filter incomplete/aborted/slow trials
"""

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: filter date\n    date: 221020")
        cmd = DEFAULT_DATE
    else:
        cmd = sys.argv[1]
    time_glob = cmd_to_time_glob(cmd)
    print("filter", time_glob)
    for f in glob(time_glob):
        exp_type = get_ext_exp_type(f, EXP_TYPE)
        trial = ImagingTrial(f, exp_type, for_filter=True)
        if trial.invalid_type and trial.invalid_type != "invalid":
        #     os.makedirs(trial.invalid_path, exist_ok=True)
            print(f, trial.invalid_path)
            shutil.move(f, trial.invalid_path)

# -*- coding: utf-8 -*-

import os
import sys
from glob import glob

from _0_constants import EXP_TYPE, SKIP_PLOT, DEFAULT_DATE
from _0_function_analysis import cmd_to_time_glob, get_ext_exp_type
# from twop_exp import TwopExp
from _0_function_imaging_exp import ImagingTrial

""" NOTE:
    plot all figures
"""

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: plot [time_glob|time|date|fly|'recent'] ['video']\n    time: 221020_163927, date: 221020, fly: 221020-M1")
        cmd = DEFAULT_DATE
    else:
        cmd = sys.argv[1]
    last_roi_info = {}
    time_glob = cmd_to_time_glob(cmd)
    if len(sys.argv) > 2:
        cmd2 = sys.argv[2]
    else:
        cmd2 = None
    print("plot", time_glob)
    for f in glob(time_glob):
        exp_type = get_ext_exp_type(f, EXP_TYPE)
        if SKIP_PLOT and ImagingTrial.is_data_exist(f, exp_type):
            print("skip plot", f)
            continue
        if os.path.exists(f + "/stim.txt"):
            # try:
                if cmd2 == "video":
                    print("video", f)
                    ImagingTrial(f, exp_type).export_video()
                elif cmd2 == "hot":
                    ImagingTrial(f, exp_type).plot_time_hot()
                else:
                    print("plot", f)
                    ImagingTrial(f, exp_type).plot_pr()
                    # ImagingTrial(f, exp_type).plot_all()
                    # TwopExp(f, exp_type).plot_all()
            # except:
            #     print("    error")
            #     import traceback
            #     print(traceback.format_exc())

# -*- coding: utf-8 -*-

import os
import sys
import datetime
import shutil
from glob import glob
from os.path import join as pj

LOCAL_DATA_PATH = r"C:\fictrac\data" #r"E:\NJ"  #
REMOTE_DATA_PATH = r"\\192.168.1.38\nj\Imaging_data"

""" NOTE:
    from
        for fictrac data
            under LOCAL_DATA_PATH: [date]-[fly]-FT_[info] (e.g. 221020-M1-FT_CX1013G)
        for imaging data
            under LOCAL_DATA_PATH: [date]-[fly]_[trial] (e.g. 221020-M1_001), [date]-[fly]_TS[trial] (e.g. 221020-M1_TS000)

    to REMOTE_DATA_PATH: [date]/[fictrac & imaging data]
"""

def main():
    if len(sys.argv) < 2:
        print("Usage: copy_data [date|'today']\n    date: 221020")
    cmd = sys.argv[1]
    if cmd == "today":
        now = datetime.datetime.now()
        date_str = now.strftime("%y%m%d")
    else:
        date_str = cmd
    copy_data(date_str)

def copy_data(date_str):
    g = pj(LOCAL_DATA_PATH, date_str+"*")  # E:\NJ\221020*, C:\fictrac\data\221020*
    dst = pj(REMOTE_DATA_PATH, date_str)
    print("copy", g, "to", dst)
    os.makedirs(dst, exist_ok=True)
    for f in glob(g):
        print("    ", f)
        ff = pj(dst, os.path.basename(f))
        if os.path.exists(ff):
            print("    already exists, skip")
            continue
        shutil.copytree(f, ff)
        print("    ok")

if __name__ == '__main__':
    main()

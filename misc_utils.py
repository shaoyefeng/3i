import os
import json
from glob import glob
from datetime import datetime, timedelta

def load_dict(filename):
    if not os.path.exists(filename):
        return None
    # print("load_dict %s" % filename)
    f = open(filename, "r")
    j = json.load(f)
    f.close()
    return j

def save_dict(filename, obj):
    f = open(filename, "w")
    json.dump(obj, f, indent=4)
    f.close()
    print("save_dict %s" % filename)

def format_time(t):
    if t < 60:
        return "%.2f" % t
    return "%d:%.2f" % (t//60, t%60)

def str_to_time(s):
    return datetime.strptime(s, "%y%m%d_%H%M%S")

def str_time_diff(s1, s2):
    return (str_to_time(s2) - str_to_time(s1)).seconds

def mp_process_glob(cb, file_glob, pool_size=0):
    mp_process_files(cb, glob(file_glob), pool_size)

def mp_process_files(cb, files, pool_size=0):
    if pool_size:
        from multiprocessing import Pool
        p = Pool(pool_size)
        p.map(cb, files)
        p.close()
    else:
        for f in files:
            print("mp_process", f)
            cb(f)

def lim_dir_a(dir1, pi=180):
    dir1[dir1 > pi] -= 2 * pi
    dir1[dir1 < -pi] += 2 * pi
    return dir1

def lim_dir(dir1, pi=180):
    if dir1 > pi:
        dir1 -= 2*pi
    elif dir1 < -pi:
        dir1 += 2*pi
    return dir1
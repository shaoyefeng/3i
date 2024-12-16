
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from scipy.stats import circstd

from _0_function_imaging_exp import ImagingTrial, TimeSeqPool, FIG_SUM_NAMES2, STIM_PRE, STIM_POST, STIM_ON_FOLD, STIM_ON

# plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.sans-serif'] = ['Arial']

# NOTE: based on data/*/*/*.pickle data
""" group structure
-Group1
    -Fly1
    -Fly2
    -Fly3
        -Trial1
        -Trial2
            TimeSeq1
            TimeSeq2
"""
class ImagingFly(object):
    def __init__(self, fly_name, stim_name=""):
        print("fly", fly_name)
        self.fly_name = fly_name
        self.stim_name = stim_name
        pickle_glob = "data/*/%s*/*.pickle" % fly_name
        self.pickle_l = []
        if stim_name:
            for p in glob(pickle_glob):
                f = open(p, "rb")
                data = pickle.load(f)
                f.close()
                if "ft_data" not in data:
                    print("ft_data not found !!!", p)
                elif data["ft_data"]["config"].get("stim_name") == stim_name:
                    self.pickle_l.append(p)
        else:
            self.pickle_l = glob(pickle_glob)

    def __len__(self):
        return len(self.pickle_l)

    def load(self):
        # print(self.pickle_l)
        cycles = []
        trial_l = []
        fold_len_d = np.zeros(2000)
        for p in self.pickle_l:
            trial = ImagingTrial(None, None, p)
            if not trial.ima_data or not trial.ima_data.get("2p_info"):
                continue
            if not trial.split_time_seq():
                continue
            trial_l.append(trial)
            fold_len_d[trial.timep_cycles.fold_len] += 1
        need_fold_len = int(np.argmax(fold_len_d)) #4 if self.stim_name == "OL_grating" else 12

        self.trial_l = []
        for trial in trial_l:
            if trial.timep_cycles.fold_len != need_fold_len:
                print("fold_len:", trial.timep_cycles.fold_len, "!=", need_fold_len)
                continue
            self.trial_l.append(trial)
            cycles.extend(trial.timep_cycles.time_seq_l)
        if len(self.trial_l) == 0:
            print("no trials!")
            return False
        self.timep_stim_pre = TimeSeqPool([self.fly_name, STIM_PRE], [t.times_stim_pre for t in self.trial_l])
        self.timep_stim_on = TimeSeqPool([self.fly_name,  STIM_ON], [t.times_stim_on for t in self.trial_l])
        self.timep_stim_post = TimeSeqPool([self.fly_name, STIM_POST], [t.times_stim_post for t in self.trial_l])
        self.timep_cycles = TimeSeqPool([self.fly_name, STIM_ON_FOLD], cycles, trial.timep_cycles.n_cycles, need_fold_len)
        return True

    def plot_single(self):
        for t in self.trial_l:
            t.plot_all()

    def plot_summary(self):
        geno = self.trial_l[0].exp_type
        img_path = "img/_Fly/%s/%s/%s_%s" % (geno, self.stim_name, self.fly_name, self.stim_name)
        print("plot_summary:", img_path)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        title_l = [self.fly_name, self.stim_name, "(%d trials)" % len(self)]
        self.timep_stim_pre.plot_summary(FIG_SUM_NAMES2, title_l, img_path)
        self.timep_stim_on.plot_summary(FIG_SUM_NAMES2, title_l, img_path)
        self.timep_stim_post.plot_summary(FIG_SUM_NAMES2, title_l, img_path)
        self.timep_cycles.plot_summary(FIG_SUM_NAMES2, title_l, img_path)


class ImagingFlyGroup(object):
    def __init__(self, geno_type, stim_name):
        print("group", geno_type, stim_name)
        self.geno_type = geno_type
        self.stim_name = stim_name
        self.fly_l = []
        for f in glob("data/*/*%s" % geno_type):
            fly_name = os.path.basename(f)
            fly = ImagingFly(fly_name, stim_name)
            if len(fly) == 0:
                continue
            self.fly_l.append(fly)
            print(fly.pickle_l)

    def load(self):
        pres, posts, ons, cycles = [], [], [], []
        fly_l = []
        fold_len_d = np.zeros(2000)
        for fly in self.fly_l:
            if not fly.load():
                continue
            # TODO: filter by speed
            fly_l.append(fly)
            fold_len_d[fly.timep_cycles.fold_len] += 1
        need_fold_len = int(np.argmax(fold_len_d))
        self.fly_l = []
        for fly in fly_l:
            if fly.timep_cycles.fold_len != need_fold_len:
                print("fold_len:", fly.timep_cycles.fold_len, "!=", need_fold_len)
                continue
            self.fly_l.append(fly)
            pres.extend(fly.timep_stim_pre.time_seq_l)
            posts.extend(fly.timep_stim_post.time_seq_l)
            ons.extend(fly.timep_stim_on.time_seq_l)
            cycles.extend(fly.timep_cycles.time_seq_l)
        if len(self.fly_l) == 0:
            return False
        self.times_stim_pre = TimeSeqPool([self.geno_type, self.stim_name, STIM_PRE], pres)
        self.times_stim_on = TimeSeqPool([self.geno_type, self.stim_name, STIM_ON], ons)
        self.times_stim_post = TimeSeqPool([self.geno_type, self.stim_name, STIM_POST], posts)
        self.timep_cycles = TimeSeqPool([self.geno_type, self.stim_name, STIM_ON_FOLD], cycles, fly.timep_cycles.n_cycles)
        return True

    def plot_single(self, single_trial=False):
        for f in self.fly_l:
            single_trial and f.plot_single()
            f.plot_summary()

    def plot_summary(self):
        img_path = "img/_Group/%s/%s_%s" % (self.geno_type, self.geno_type, self.stim_name)
        print("plot_summary:", img_path)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        title_l = [self.geno_type, self.stim_name, "(%d flies, %d trials)" % (len(self.fly_l), sum([len(f) for f in self.fly_l]))]
        self.times_stim_pre.plot_summary(FIG_SUM_NAMES2, title_l, img_path)
        self.times_stim_on.plot_summary(FIG_SUM_NAMES2, title_l, img_path)
        self.times_stim_post.plot_summary(FIG_SUM_NAMES2, title_l, img_path)
        self.timep_cycles.plot_summary(FIG_SUM_NAMES2, title_l, img_path)




import os
import pickle
import numpy as np
from glob import glob

# NOTE: data/*/220627_F3/220627_181001.pickle {"config":{}, "x_df":DataFrame}
# NOTE: img/*/220627_F3/220627_181001/ (time_0.png...)
# NOTE: img/*_sum220803/220627_F3_220627_181001.png
from _0_constants import LOAD_CACHE, IMG_ROOT, DATA_ROOT


class BaseExp(object):
    def __init__(self, exp_folder, pickle_path=None, for_filter=False):
        self.data = None
        if pickle_path:  # BaseExp(None, None, "*.pickle")
            self.data = pickle.load(open(pickle_path, "rb"))
            self.exp_folder = self.data["exp_folder"]
            # self.exp_type = self.data["exp_type"]
        else:
            self.exp_folder = exp_folder
            # self.exp_type = exp_type
        # self.parent_img = IMG_ROOT + self.exp_type + "/"
        # self.parent_data = DATA_ROOT + self.exp_type + "/"

        self.exp_name = os.path.basename(self.exp_folder)
        if self.exp_folder.endswith(".pickle"):  # init from data
            self.exp_name = self.exp_name[:-7]
        self.fly_name = os.path.basename(os.path.dirname(self.exp_folder))

        # self.data_path = os.path.join(self.exp_folder, self.exp_name + ".pickle")
        # self.img_path = os.path.join(self.parent_img, self.fly_name, self.exp_name)
        # self.img_sum_path = self.img_path + "_sum"
        # self.img_sum_path = "img/%s_sum%s/%s-%s.png" % (exp_type, SUM_LIST[0], self.fly_name, self.exp_name)

        if for_filter:
            self.invalid_type = None
            self.load_for_filter()
            return
        if not pickle_path:
            self.load_data()
        # self.proc_raw_data()

    @staticmethod
    def is_data_exist(exp_folder, exp_type):
        fly_name = os.path.basename(os.path.dirname(exp_folder))
        parent_data = "data/" + exp_type + "/"
        exp_name = os.path.basename(exp_folder)
        data_path = os.path.join(parent_data, fly_name, exp_name + ".pickle")
        return os.path.exists(data_path)

    def load_data(self):
        if LOAD_CACHE and os.path.exists(self.data_path):
            print("load %s from data folder." % self.exp_name)
            self.data = pickle.load(open(self.data_path, "rb"))
        else:
            print("load %s from raw fictrac.dat & stim.txt." % self.exp_name)
            self.load_raw_data()
            if self.data is None:
                print("load %s failed." % self.exp_name)
                return
            self.data["exp_folder"] = self.exp_folder
            # self.data["exp_type"] = self.exp_type
            # os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
            self.data_path = os.path.join(self.exp_folder, self.data['ft_data']['config']['stim_name'] + ".pickle")
            pickle.dump(self.data, open(self.data_path, "wb"))

    def load_for_filter(self):
        pass

    def load_raw_data(self):
        pass

    def proc_raw_data(self):
        pass

    @staticmethod
    def is_exist(exp_folder, exp_type):
        e = BaseExp(exp_folder, exp_type, load=False)
        return os.path.exists(e.data_path)# and os.path.exists(e.img_sum_path)

class BaseExpCollection(object):
    def __init__(self, exp_glob, col_name, exp_type, idx=None):
        self.exp_glob = exp_glob
        self.exp_type = exp_type
        self.col_name = col_name
        self.parent_img = "img/" + exp_type + "/" + col_name + "/"

        if idx is None:
            g = glob(exp_glob)
        else:
            g = np.array(glob(exp_glob))[idx]
        self.exp_l = []
        end_trial = 0
        for i, f in enumerate(g):
            exp = self.create_exp(f)
            if exp.stim_df is not None:
                end_trial = exp.trial_no_add(end_trial)
                self.exp_l.append(exp)

    def create_exp(self, exp_folder):
        return BaseExp(exp_folder, self.exp_type)

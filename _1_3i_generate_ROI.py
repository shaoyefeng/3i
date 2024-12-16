# -*- coding: utf-8 -*-

import os
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from _0_function_roi import draw_roi_png, define_roi_no_show
from _0_function_analysis import calc_all_roi_F, split_z_slices, process_move_file, process_motion_correction

plt.ioff()

data_path = r"H:\Data\R22A07XSS00730\Pharmacology\Control"
imaging_rate = 11.5

# process_move_file(data_path)
# process_motion_correction(data_path)

for date_path in glob(data_path+'/*'):
    for fly_path in glob(date_path+'/*'):
        path_glob = glob(fly_path+'/*')
        trial_glob = [dI for dI in path_glob if os.path.isdir(dI)]
        for i, trial_path in enumerate(trial_glob):
            mmap_path = glob(trial_path+"/*.mmap")[0]
            print("process roi in", mmap_path)
            roi_file = trial_path + "/roi.npy"
            bg_file = trial_path + "/i_std.png"

            if i == 0:
                temp = None
            else:
                temp = last_roi_info.get(i - 1)

            # temp = None
            fly_name = os.path.basename(fly_path)
            ui = define_roi_no_show(temp, bg_file)
            plt.show()
            last_roi_info = {i: roi_file}

            draw_roi_png(trial_path)
            # chs = split_z_slices(tif_path)
            chs = split_z_slices(mmap_path)
            plt.close()
            for j, ch in enumerate(chs):
                calc_all_roi_F(roi_file, ch, trial_path)

            df_path = os.path.join(trial_path, 'F.csv')
            df = pd.read_csv(df_path)
            names = df.columns
            ratio_df = pd.DataFrame([])
            ratio_df['frame'] = df.index
            ratio_df['time'] = ratio_df['frame'] / imaging_rate
            for g in names[1:]:
                f = df[g] - df['0']
                f0 = np.average(f.iloc[-50:-1])
                ratio_df['ROI_' + str(g)] = (f - f0) / f0
            ratio_df.to_csv(df_path.replace('F.csv', 'ratio.csv'), index=False)



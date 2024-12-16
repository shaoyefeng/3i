import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import numpy as np

from _0_constants import *
from _0_function_analysis import plot_PhotoStim, merge_all_trials

imaging_rate = 11.5
data_path = r"H:\Data\R24A08XSS01553"
ROI_l = ['ROI_1']
stim_l = ['2p_500_ctrl', '2p_100', '2p_200', '2p_300', '2p_400', '2p_500']
compart = 'somaL_'



for ROI in ROI_l:
    for i, stim in enumerate(stim_l):
        for date_path in glob(data_path+'/*'):
            for fly_path in glob(date_path+'/*'):
                fly = (os.path.basename(fly_path)).split('_')[-1]
                path_glob = glob(fly_path+'/*')
                trial_glob = [dI for dI in path_glob if os.path.isdir(dI) and (compart+stim) in dI]
                if trial_glob:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    plt.subplots_adjust(left=0.10, right=0.99, bottom=0.1, top=0.94)
                    fig.suptitle(ROI)
                    fly_df = pd.DataFrame([])
                    for trial_path in trial_glob:
                        mean_df = merge_all_trials(trial_path, imaging_rate)
                        fly_df = pd.concat([fly_df, mean_df], ignore_index=True)
                    sns.lineplot(data=fly_df, x='time', y=ROI, label=stim, errorbar='se')
                    plot_PhotoStim(ax, imaging_rate, merged=True)

                    result_path = data_path.replace('Data', 'Result')
                    each_fly_path = os.path.join(result_path, 'each_fly_merged')
                    if not os.path.exists(each_fly_path):
                        os.makedirs(each_fly_path)
                    fig_path = os.path.join(each_fly_path, '%s_%s_%s%s.png' % (fly, stim, compart, ROI))
                    plt.savefig(fig_path, dpi=600)
                    plt.close()
                    print(fig_path)
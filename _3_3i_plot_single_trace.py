import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

from _0_function_analysis import plot_PhotoStim

imaging_rate = 11.5
data_path = r"H:\Data\R22A07XSS00730\Pharmacology\Control"
ROI_l = ['ROI_1']


for date_path in glob(data_path+'/*'):
    for fly_path in glob(date_path+'/*'):
        fly = (os.path.basename(fly_path)).split('_')[-1]
        path_glob = glob(fly_path+'/*')
        trial_glob = [dI for dI in path_glob if os.path.isdir(dI)]
        if trial_glob:
            for trial_path in trial_glob:
                trial_df_path = os.path.join(trial_path, 'ratio.csv')
                trial_df = pd.read_csv(trial_df_path)
                fig, ax = plt.subplots(figsize=(20, 5))
                plt.subplots_adjust(left=0.04, right=0.99, bottom=0.1, top=0.94)
                # fig.suptitle(stim)
                for ROI in ROI_l:
                    if ROI in trial_df.columns:
                        sns.lineplot(data=trial_df, x='time', y=ROI, label=ROI, errorbar='se')
                plot_PhotoStim(ax, imaging_rate)

                result_path = data_path.replace('Data', 'Result')
                each_fly_path = os.path.join(result_path, fly)
                if not os.path.exists(each_fly_path):
                    os.makedirs(each_fly_path)
                fig_path = os.path.join(each_fly_path, '%s_%s.png' % (fly, os.path.basename(trial_path)))
                plt.savefig(fig_path, dpi=600)
                plt.close()
                print(fig_path)
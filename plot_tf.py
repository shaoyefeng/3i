import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

from _0_function_analysis import plot_PhotoStim

fps = 7
fly_path = r"F:\Project\Physiology\3i\Data\MBON11XVC91\231212\MBON11XVC91_fly3"
ROI_l = ['ROI_1', 'ROI_2', 'ROI_3']
stim_l = ['2p_500_', '2p_500tf']

for ROI in ROI_l:
    fig, ax = plt.subplots(figsize=(20, 5))
    plt.subplots_adjust(left=0.04, right=0.99, bottom=0.1, top=0.94)
    fig.suptitle(ROI)
    for stim in stim_l:
        fly = (os.path.basename(fly_path)).split('_')[-1]
        path_glob = glob(fly_path + '/*')
        trial_glob = [dI for dI in path_glob if os.path.isdir(dI) and stim in dI]
        total_df = pd.DataFrame([])
        for i, trial_path in enumerate(trial_glob):
            df_path = os.path.join(trial_path, 'ratio.csv')
            df = pd.read_csv(df_path)
            total_df = pd.concat([total_df, df], ignore_index=True)
        sns.lineplot(data=total_df, x='time', y=ROI, label=stim)

    plot_PhotoStim(ax, fps)
    result_path = fly_path.replace('Data', 'Result')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    fig_path = os.path.join(result_path, '500_%s.png' % ROI)
    plt.savefig(fig_path)
    print(fig_path)
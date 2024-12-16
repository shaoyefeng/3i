import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from glob import glob

from _0_function_analysis import plot_PhotoStim, merge_all_trials
from _0_constants import *

data_path = r"F:\Project\Physiology\3i\Data\MBON11XVC91"
ROI_l = ['ROI_1', 'ROI_2', 'ROI_3']
# stim_l = ['2p_300', '2p_400', '2p_500']
stim_l = ['2p_100', '2p_200', '2p_300', '2p_400', '2p_500']
compart = ''

cmap = cm.get_cmap('rainbow', len(stim_l))
colors = np.linspace(1, 0, len(stim_l))

for ROI in ROI_l:
    fig, ax = plt.subplots(figsize=(20, 5))
    plt.subplots_adjust(left=0.04, right=0.99, bottom=0.1, top=0.94)
    fig.suptitle(ROI)
    for i, stim in enumerate(stim_l):
        total_df = pd.DataFrame([])
        for date_path in glob(data_path+'/*'):
            for fly_path in glob(date_path+'/*'):
                fly = (os.path.basename(fly_path)).split('_')[-1]
                path_glob = glob(fly_path+'/*')
                trial_glob = [dI for dI in path_glob if os.path.isdir(dI) and (compart+stim) in dI]
                if trial_glob:
                    fly_df = pd.DataFrame([])
                    for trial_path in trial_glob:
                        df_path = os.path.join(trial_path, 'ratio.csv')
                        df = pd.read_csv(df_path)
                        # #  mean ROI
                        # df['ROI'] = df[['ROI_1', 'ROI_2']].mean(axis=1)
                        fly_df = pd.concat([fly_df, df], ignore_index=True)
                    mean_df = fly_df.groupby('frame').agg('mean')
                    total_df = pd.concat([fly_df, mean_df], ignore_index=True)
        sns.lineplot(data=total_df, x='time', y=ROI, label=stim, color=cmap(colors[i]), errorbar='se')
    plot_PhotoStim(ax)

    result_path = data_path.replace('Data', 'Result')
    each_fly_path = os.path.join(result_path, 'ROI')
    if not os.path.exists(each_fly_path):
        os.makedirs(each_fly_path)
    fig_path = os.path.join(each_fly_path, '%s%s.png' % (compart, ROI))
    plt.savefig(fig_path, dpi=600)
    plt.close()
    print(fig_path)
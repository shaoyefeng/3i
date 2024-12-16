import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

from _0_constants import *
from _0_function_analysis import plot_PhotoStim, merge_all_trials

data_path = r"F:\Project\Physiology\3i\Data\MBON11_compartment"
stim_l = ['2p_100', '2p_300', '2p_500']
compart_l = ['whole_', 'y1_', 'ped_']
ROI = 'ROI_1'

colors = ['r', 'g', 'b']
# cmap = cm.get_cmap('rainbow', len(stim_l))
# colors = np.linspace(1, 0, len(stim_l))

for stim in stim_l:
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.subplots_adjust(left=0.10, right=0.99, bottom=0.1, top=0.94)
    fig.suptitle(stim)
    for i, compart in enumerate(compart_l):
        total_df = pd.DataFrame([])
        for date_path in glob(data_path+'/*'):
            for fly_path in glob(date_path+'/*'):
                fly = (os.path.basename(fly_path)).split('_')[-1]
                path_glob = glob(fly_path+'/*')
                trial_glob = [dI for dI in path_glob if os.path.isdir(dI) and (compart+stim) in dI]
                if trial_glob:
                    fly_df = pd.DataFrame([])
                    for trial_path in trial_glob:
                        single_stim_df = merge_all_trials(trial_path)
                        fly_df = pd.concat([fly_df, single_stim_df], ignore_index=True)
                    mean_df = fly_df.groupby('frame').agg('mean')
                    mean_df['frame'] = mean_df.index
                    mean_df['time'] = mean_df['frame'] / fps
                    total_df = pd.concat([total_df, mean_df], ignore_index=True)
        sns.lineplot(data=total_df, x='time', y=ROI, label=compart, color=colors[i], errorbar='se')
    plot_PhotoStim(ax, merged=True)
    result_path = data_path.replace('Data', 'Result')
    each_fly_path = os.path.join(result_path, 'compartment')
    if not os.path.exists(each_fly_path):
        os.makedirs(each_fly_path)
    fig_path = os.path.join(each_fly_path, '%s_%s.png' % (ROI, stim))
    plt.savefig(fig_path, dpi=600)
    plt.close()
    print(fig_path)
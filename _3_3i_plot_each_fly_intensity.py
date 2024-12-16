import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from glob import glob

from _0_function_analysis import plot_PhotoStim, merge_all_trials
from _0_constants import *

imaging_rate = 11.5
data_path = r"H:\Data\R22A07XSS00730\Pharmacology\PTX"
ROI_l = ['ROI_1']
# ROI_l = ['ROI_1', 'ROI_2', 'ROI_3', 'ROI_4', 'ROI_5', 'ROI_6', 'ROI_7', 'ROI_8']
stim_l = ['2p_300']
# compart_l = ['somaL1', 'somaL2', 'somaR1', 'somaR2', 'tdtomatoL', 'tdtomatoR']
compart_l = ['somaL', 'somaR', 'somaL_NoLaser']

cmap = cm.get_cmap('rainbow', len(stim_l))
colors = np.linspace(1, 0, len(stim_l))

result_path = data_path.replace('Data', 'Result')
each_fly_path = os.path.join(result_path, 'each_fly_intensity')
if not os.path.exists(each_fly_path):
    os.makedirs(each_fly_path)

for compart in compart_l:
    for ROI in ROI_l:
        for date_path in glob(data_path+'/*'):
            for fly_path in glob(date_path+'/*'):
                fig, ax = plt.subplots(figsize=(10, 5))
                plt.subplots_adjust(left=0.10, right=0.99, bottom=0.1, top=0.94)
                fig.suptitle(ROI)
                fly = (os.path.basename(fly_path)).split('_')[-1]
                path_glob = glob(fly_path+'/*')
                for i, stim in enumerate(stim_l):
                    trial_glob = [dI for dI in path_glob if os.path.isdir(dI) and (compart+'_'+stim) in dI]
                    if trial_glob:
                        fly_df = pd.DataFrame([])
                        for trial_path in trial_glob:
                            single_stim_df = merge_all_trials(trial_path, imaging_rate)
                            fly_df = pd.concat([fly_df, single_stim_df], ignore_index=True)

                        if stim == '2p_500_ctrl':
                            color = 'k'
                        else:
                            color = cmap(colors[i])
                        if ROI in fly_df.columns:
                            sns.lineplot(data=fly_df, x='time', y=ROI, label=stim, errorbar='se', color=color)
                plot_PhotoStim(ax, imaging_rate, merged=True)

                fig_path = os.path.join(each_fly_path, '%s_%s_%s.png' % (fly, compart, ROI))
                plt.savefig(fig_path, dpi=600)
                print(fig_path)
                plt.close()

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from glob import glob

from _0_function_analysis import plot_PhotoStim, merge_all_trials
from _0_constants import *

imaging_rate = 11.5
data_path = r"H:\Data\R22A07XSS00730\Pharmacology\PTX\100uM"
ROI_l = ['ROI_1']
stim_l = ['2p_400']
compart_l = ['somaL', 'somaR']
drug = 'PTX'

result_path = data_path.replace('Data', 'Result')
each_fly_path = os.path.join(result_path, 'each_fly_intensity_Pharmacology')
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
                    drug_trial_glob = [dI for dI in path_glob if os.path.isdir(dI) and '%s_%s_%s' % (compart, drug, stim) in dI]
                    trial_glob = [dI for dI in path_glob if os.path.isdir(dI) and (compart+'_'+stim) in dI]
                    if trial_glob:
                        fly_df = pd.DataFrame([])
                        drug_fly_df = pd.DataFrame([])
                        for trial_path in trial_glob:
                            single_stim_df = merge_all_trials(trial_path, imaging_rate, if_ratio=False)
                            fly_df = pd.concat([fly_df, single_stim_df], ignore_index=True)
                        for drug_trial_path in drug_trial_glob:
                            drug_single_stim_df = merge_all_trials(drug_trial_path, imaging_rate, if_ratio=False)
                            drug_fly_df = pd.concat([drug_fly_df, drug_single_stim_df], ignore_index=True)

                        sns.lineplot(data=fly_df, x='time', y=ROI, label=stim, errorbar='se', color='k')
                        sns.lineplot(data=drug_fly_df, x='time', y=ROI, label=stim+'_'+drug, errorbar='se', color='r')

                plot_PhotoStim(ax, imaging_rate, merged=True)

                # plt.show()
                fig_path = os.path.join(each_fly_path, 'F_%s_%s_%s.png' % (fly, compart, drug))
                plt.savefig(fig_path, dpi=600)
                print(fig_path)
                plt.close()

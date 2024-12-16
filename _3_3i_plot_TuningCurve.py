import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from glob import glob
import numpy as np

from _0_function_analysis import plot_PhotoStim

fps = 7
stim_t_l = np.array([19, 79, 139, 199])
data_path = r"F:\Project\Physiology\3i\Data\P1XVC91"
ROI_l = ['ROI_1', 'ROI_2', 'ROI_3']
stim_l = ['2p_300', '2p_400', '2p_500']
# stim_l = ['2p_100', '2p_200', '2p_300', '2p_400', '2p_500']

cmap = cm.get_cmap('rainbow', len(stim_l))
colors = np.linspace(1, 0, len(stim_l))


for ROI in ROI_l:
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.subplots_adjust(left=0.10, right=0.99, bottom=0.1, top=0.94)
    fig.suptitle(ROI)
    total_index_df = pd.DataFrame([])
    for i, stim in enumerate(stim_l):
        for date_path in glob(data_path+'/*'):
            for fly_path in glob(date_path+'/*'):
                fly = (os.path.basename(fly_path)).split('_')[-1]
                path_glob = glob(fly_path+'/*')
                trial_glob = [dI for dI in path_glob if os.path.isdir(dI) and stim in dI]
                if trial_glob:
                    fly_index_l = []
                    fly_df = pd.DataFrame([])
                    for trial_path in trial_glob:
                        df_path = os.path.join(trial_path, 'F.csv')
                        df = pd.read_csv(df_path)
                        names = df.columns
                        dff_df = pd.DataFrame([])
                        dff_df['frame'] = df.index
                        for g in names[1:]:
                            dff_df[g] = df[g] - df['0']
                        all_stim_df = pd.DataFrame([])
                        for stim_t in stim_t_l:
                            stim_df = dff_df.iloc[int(stim_t-10):int(stim_t+20)]
                            stim_df = stim_df.reset_index(drop=True)
                            ratio_df = pd.DataFrame([])
                            ratio_df['frame'] = stim_df.index
                            ratio_df['time'] = ratio_df['frame'] / fps
                            for g in names[1:]:
                                f = stim_df[g]
                                f0 = np.average(f.iloc[:10])
                                ratio_df['ROI_' + str(g)] = (f - f0) / f0
                            all_stim_df = pd.concat([all_stim_df, ratio_df])
                        single_stim_df = all_stim_df.groupby('frame').agg('mean')
                        single_stim_df['frame'] = single_stim_df.index
                        fly_df = pd.concat([fly_df, single_stim_df], ignore_index=True)
                    mean_df = fly_df.groupby('frame').agg('mean')
                    mean_df['frame'] = mean_df.index
                    integral = np.trapz(mean_df[ROI].iloc[10:20])
                    response_index = round(integral, 2)
                    fly_index_l.append(response_index)
                    fly_index_df = pd.DataFrame([])
                    fly_index_df['response_index'] = fly_index_l
                    # fly_index_df['genotype'] = genotype
                    fly_index_df['stim'] = int(stim.split('_')[-1])
                    total_index_df = pd.concat([total_index_df, fly_index_df], ignore_index=True)
    sns.lineplot(data=total_index_df, x='stim', y='response_index', marker='o', err_style="bars", errorbar='se')

    result_path = data_path.replace('Data', 'Result')
    each_fly_path = os.path.join(result_path, 'TuningCurve')
    if not os.path.exists(each_fly_path):
        os.makedirs(each_fly_path)
    fig_path = os.path.join(each_fly_path, '%s.png' % ROI)
    plt.savefig(fig_path, dpi=600)
    plt.close()
    print(fig_path)
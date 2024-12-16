import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from glob import glob

from _0_function_analysis import plot_PhotoStim, merge_all_trials
from _0_constants import *

imaging_rate = 11.5
data_path = r"H:\Data\R24A08XSS00730"
ROI = 'ROI_1'
# stim_l = ['2p_100', '2p_200', '2p_300', '2p_400', '2p_500']
stim_l = ['2p_500']

cmap = cm.get_cmap('rainbow', len(stim_l))
colors = np.linspace(1, 0, len(stim_l))


# Intensity tuning

# fig, ax = plt.subplots(figsize=(10, 5))
# plt.subplots_adjust(left=0.10, right=0.99, bottom=0.1, top=0.94)
# for i, stim in enumerate(stim_l):
#     stim_df = pd.DataFrame([])
#     for date_path in glob(data_path + '/*'):
#         for fly_path in glob(date_path + '/*'):
#             fly = (os.path.basename(fly_path)).split('_')[-1]
#             path_glob = glob(fly_path + '/*')
#             trial_glob = [dI for dI in path_glob if os.path.isdir(dI) and stim in dI]
#             if trial_glob:
#                 fly_df = pd.DataFrame([])
#                 for trial_path in trial_glob:
#                     single_stim_df = merge_all_trials(trial_path, imaging_rate)
#                     fly_df = pd.concat([fly_df, single_stim_df], ignore_index=True)
#                 mean_fly_df = fly_df.groupby('frame', as_index=False).agg('mean')
#                 stim_df = pd.concat([stim_df, mean_fly_df], ignore_index=True)
#
#     sns.lineplot(data=stim_df, x='time', y=ROI, label=stim, errorbar='se', color=cmap(colors[i]))
# plot_PhotoStim(ax, imaging_rate, merged=True)
#
# result_path = data_path.replace('Data', 'Result')
# each_fly_path = os.path.join(result_path, 'intensity_tuning')
# if not os.path.exists(each_fly_path):
#     os.makedirs(each_fly_path)
# fig_path = os.path.join(each_fly_path, 'intensity_tuning.png')
# plt.savefig(fig_path, dpi=600)
# plt.close()
# print(fig_path)


# Across trial

trial_l = ['', ' - 1', ' - 2', ' - 3', ' - 4']

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 8))
plt.subplots_adjust(left=0.10, right=0.99, bottom=0.1, top=0.94)
axes = axes.flatten()

for i, stim in enumerate(stim_l):
    ax = axes[i]
    ax.set_title(stim)
    for j, trial_i in enumerate(trial_l):
        trial = stim + trial_i
        trial_df = pd.DataFrame([])
        for date_path in glob(data_path + '/*'):
            for fly_path in glob(date_path + '/*'):
                fly = (os.path.basename(fly_path)).split('_')[-1]
                trial_path = glob(fly_path + '/*' + trial)[0]
                fly_df = merge_all_trials(trial_path, imaging_rate)
                trial_df = pd.concat([trial_df, fly_df], ignore_index=True)

        sns.lineplot(ax=ax, data=trial_df, x='time', y=ROI, label='trial_'+str(j+1), errorbar='se')
        plot_PhotoStim(ax, imaging_rate, merged=True)

result_path = data_path.replace('Data', 'Result')
each_fly_path = os.path.join(result_path, 'Across_trial')
if not os.path.exists(each_fly_path):
    os.makedirs(each_fly_path)
fig_path = os.path.join(each_fly_path, 'Across_trial.png')
plt.savefig(fig_path, dpi=600)
plt.close()
print(fig_path)


# # Across Repeat
# repeat_l = [0, 1, 2, 3]
#
# fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 8))
# plt.subplots_adjust(left=0.10, right=0.99, bottom=0.1, top=0.94)
# axes = axes.flatten()
# for i, stim in enumerate(stim_l):
#     ax = axes[i]
#     ax.set_title(stim)
#     stim_df = pd.DataFrame([])
#     for repeat_i in repeat_l:
#         stim_t = stim_t_l[repeat_i]
#         repeat_df = pd.DataFrame([])
#         for date_path in glob(data_path + '/*'):
#             for fly_path in glob(date_path + '/*'):
#                 path_glob = glob(fly_path + '/*')
#                 trial_glob = [dI for dI in path_glob if os.path.isdir(dI) and stim in dI]
#                 fly_df = pd.DataFrame([])
#                 for trial_path in trial_glob:
#                     # Function merge_all_trials
#                     trial_df_path = os.path.join(trial_path, 'F.csv')
#                     trial_df = pd.read_csv(trial_df_path)
#                     names = trial_df.columns
#                     dff_df = pd.DataFrame([])
#                     dff_df['frame'] = trial_df.index
#                     for g in names[1:]:
#                         dff_df[g] = trial_df[g] - trial_df['0']
#                         stim_df = dff_df.iloc[int(stim_t - 10):int(stim_t + 20)]
#                         stim_df = stim_df.reset_index(drop=True)
#                         ratio_df = pd.DataFrame([])
#                         ratio_df['frame'] = stim_df.index
#                         ratio_df['time'] = ratio_df['frame'] / imaging_rate
#                         for g in names[1:]:
#                             f = stim_df[g]
#                             f0 = np.average(f.iloc[:10])
#                             ratio_df['ROI_' + str(g)] = (f - f0) / f0
#
#                         fly_df = pd.concat([fly_df, ratio_df], ignore_index=True)
#                 mean_fly_df = fly_df.groupby('frame', as_index=False).agg('mean')
#                 repeat_df = pd.concat([repeat_df, mean_fly_df], ignore_index=True)
#
#         sns.lineplot(ax=ax, data=repeat_df, x='time', y=ROI, label='repeat_'+str(repeat_i+1), errorbar='se')
#         plot_PhotoStim(ax, imaging_rate, merged=True)
#
# result_path = data_path.replace('Data', 'Result')
# each_fly_path = os.path.join(result_path, 'Across_repeat')
# if not os.path.exists(each_fly_path):
#     os.makedirs(each_fly_path)
# fig_path = os.path.join(each_fly_path, 'Across_repeat.png')
# plt.savefig(fig_path, dpi=600)
# plt.close()
# print(fig_path)
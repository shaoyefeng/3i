# -*- coding: utf-8 -*-

import h5py
import os
import cv2
import json
import shutil
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
from glob import glob
from scipy.stats import circmean, pearsonr
import seaborn as sns
from tifffile import tifffile

from _0_constants import *
from _0_function_roi import roi_contours_to_points
from _0_function_motion_correction import load_mmap
from _0_function_motion_correction import cm_motion_correction

plt.ioff()


def process_motion_correction(data_path):
    for date_path in glob(data_path + '/*'):
        for fly_path in glob(date_path + '/*'):
            path_glob = glob(fly_path + '/*')
            trial_glob = [dI for dI in path_glob if os.path.isdir(dI)]
            for i, trial_path in enumerate(trial_glob):
                tif_path = glob(trial_path + "/*.tif")[0]
                mmap_path = cm_motion_correction(tif_path, os.path.dirname(tif_path))
                merge_z_slices(mmap_path)
                print(mmap_path)
    print('motion correction end')


def process_move_file(data_path):
    for date_path in glob(data_path + '/*'):
        for fly_path in glob(date_path + '/*'):
            for tif_path in glob(fly_path + '/*.tif'):
                tif = os.path.basename(tif_path)
                trial_path = tif_path.replace('.tif', '')
                if not os.path.exists(trial_path):
                    os.makedirs(trial_path)
                    new_tif_path = os.path.join(trial_path, tif)
                    shutil.move(tif_path, new_tif_path)
                    print(new_tif_path)
    print('move file end')


def merge_all_trials(trial_path, imaging_rate, if_ratio=True):
    trial_df_path = os.path.join(trial_path, 'F.csv')
    trial_df = pd.read_csv(trial_df_path)
    names = trial_df.columns
    dff_df = pd.DataFrame([])
    dff_df['frame'] = trial_df.index
    for g in names[1:]:
        dff_df[g] = trial_df[g] - trial_df['0']
    all_stim_df = pd.DataFrame([])
    for stim_t in stim_t_l:
        stim_df = dff_df.iloc[int(stim_t - 10):int(stim_t + 30)]
        stim_df = stim_df.reset_index(drop=True)
        ratio_df = pd.DataFrame([])
        ratio_df['frame'] = stim_df.index
        ratio_df['time'] = ratio_df['frame'] / imaging_rate
        for g in names[1:]:
            f = stim_df[g]
            f0 = np.average(f.iloc[:10])
            if if_ratio:
                ratio_df['ROI_' + str(g)] = (f - f0) / f0
            else:
                ratio_df['ROI_' + str(g)] = f
        # stim_df = stim_df.reset_index(drop=True)
        # stim_df['frame'] = stim_df.index
        all_stim_df = pd.concat([all_stim_df, ratio_df])
    single_stim_df = all_stim_df.groupby('frame').agg('mean')
    single_stim_df['frame'] = single_stim_df.index
    return single_stim_df


def plot_PhotoStim(ax, imaging_rate, merged=False):
    if merged:
        stim_t_l = np.array([0, 10, 39]) / imaging_rate
    else:
        stim_t_l = np.array([0, 19, 79, 139, 199, 299]) / imaging_rate
    for stim_t in stim_t_l:
        if stim_t != stim_t_l[0] and stim_t != stim_t_l[-1]:
            ax.axvline(stim_t, color='r', linestyle='dashed')
    # for stim in stim_l:
    #     rect_visual_left_bot = (stim, -1)
    #     rect_visual = plt.Rectangle(rect_visual_left_bot, 1/imaging_rate, 2, color='r', alpha=0.3)
    #     ax.add_patch(rect_visual)
    ax.set_xticks(stim_t_l)
    ax.set_ylabel('delta_F / F')
    plt.legend()


def plot_stim(ax0, stim_type, pre_delay, stim_time):
    # if 'dot' or 'bar' in stim_type:
    if not 1:
        dot_x = np.array([pre_delay, pre_delay + stim_time * 0.5, pre_delay + stim_time])
        dot_y = [stim_start, stim_end, stim_start]
        sns.lineplot(x=dot_x, y=dot_y, ax=ax0, color="r", alpha=0.2)

    elif 'RDM' in stim_type or '_R' in stim_type:
        grating_x = np.array([pre_delay, pre_delay + stim_time])
        grating_y = [stim_start, stim_end]
        sns.lineplot(x=grating_x, y=grating_y, ax=ax0, color="r", alpha=0.2)

    elif '_L' in stim_type:
        grating_x = np.array([pre_delay, pre_delay + stim_time])
        grating_y = [stim_end, stim_start]
        sns.lineplot(x=grating_x, y=grating_y, ax=ax0, color="r", alpha=0.2)

    elif stim_type == 'OL_dot':
        dot_x = np.arange(pre_delay, pre_delay + stim_time, 4)
        dot_y = np.tile((stim_start, stim_end), int(len(dot_x) / 2))
        sns.lineplot(x=dot_x, y=dot_y, ax=ax0, color="r", alpha=0.2)

    elif stim_type == 'OL_bar':
        dot_x = np.arange(pre_delay, pre_delay + stim_time, 4)
        dot_y = np.tile((stim_start, stim_end), int(len(dot_x) / 2))
        sns.lineplot(x=dot_x, y=dot_y, ax=ax0, color="r", alpha=0.2)

    elif stim_type == 'looming':
        looming_x = np.linspace(pre_delay, pre_delay + stim_time, len(wid))
        sns.lineplot(x=looming_x, y=looming_size, ax=ax0, color="r", alpha=0.5)
        # sns.lineplot(x=looming_x + stim_time * 0.5, y=looming_size, ax=ax0, color="r", alpha=0.5)

    rect_visual_left_bot = (pre_delay, stim_start)
    rect_visual = plt.Rectangle(rect_visual_left_bot, stim_time, stim_end - stim_start, color='r', alpha=0.05)
    ax0.add_patch(rect_visual)

    # for vline in np.array([pre_delay + stim_time * 0.25, pre_delay + stim_time * 0.5, pre_delay + stim_time * 0.75]):
    #     plt.axvline(vline, color='black', lw=0.5)

    ax0.set_ylim(stim_start, stim_end)
    ax0.set_yticks([stim_start, 0, stim_end])
    ax0.set_ylabel('')


def load_dataclass(data_class):
    ima_data = data_class.data['ima_data']['delta_F']
    ima_fps = data_class.data['ima_data']['2p_info']['frameRate']
    pre_delay = data_class.data['ft_data']['config']['pre_duration']
    stim_duration = data_class.data['ft_data']['config']['stim_duration']
    post_delay = data_class.data['ft_data']['config']['post_duration']
    ima_df = pd.DataFrame(data=ima_data[:], columns=np.arange(ima_data.shape[1]) + 1)
    # ima_df = pd.DataFrame(data=ima_data[:], columns=['delta_F'])
    ratio_df = pd.DataFrame([])
    ratio_df['frame'] = ima_df.index
    ratio_df['time'] = ratio_df['frame'] / ima_fps
    names = ima_df.columns

    # pre_delay baseline
    baseline_index_start = 0
    baseline_index_end = round(pre_delay * ima_fps)

    # # post_delay baseline
    # baseline_index_start = round((pre_delay + stim_duration) * ima_fps)
    # baseline_index_end = round((pre_delay + stim_duration + post_delay) * ima_fps)

    for g in names:
        f0 = np.average(ima_df[g].iloc[baseline_index_start:baseline_index_end])
        ratio_df['ROI_' + str(g)] = (ima_df[g] - f0) / f0
    return ratio_df


def plot_imaging(data_class, ratio_df, fig_path):
    pre_delay = data_class.data['ft_data']['config']['pre_duration']
    stim_time = data_class.data['ft_data']['config']['stim_duration']
    post_delay = data_class.data['ft_data']['config']['post_duration']
    stim_type = data_class.data['ft_data']['config']['stim_name']

    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 8))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.06, top=0.95)
    fig.suptitle(stim_type)
    ax0 = ax.twinx()
    plot_stim(ax0, stim_type, pre_delay, stim_time)

    ## Single ROI
    # sns.lineplot(ax=ax, data=ratio_df, x='time', y='ROI_1', lw=0.5)

    # Multi ROI
    for ROI in ratio_df.columns[2:]:
        sns.lineplot(ax=ax, data=ratio_df, x='time', y=ROI, label=ROI)

    ## Merged ROI
    # merged_df = pd.DataFrame([])
    # for ROI in ratio_df.columns[2:]:
    #     roi_df = pd.DataFrame([])
    #     roi_df['ROI'] = ratio_df[ROI]
    #     roi_df['time'] = ratio_df['time']
    #     merged_df = pd.concat([merged_df, roi_df], ignore_index=True)
    # sns.lineplot(ax=ax, data=merged_df, x='time', y='ROI')

    ax.set_xlim(0, pre_delay + stim_time + post_delay)
    ax.set_xticks([0, pre_delay, pre_delay + stim_time * 0.5, pre_delay + stim_time])
    ax.set_ylabel('ratio_F')
    # plt.show()
    plt.savefig(fig_path, dpi=600)
    plt.close()


def plot_fob(data_class, fig_path):
    pre_delay = data_class.data['ft_data']['config']['pre_duration']
    stim_time = data_class.data['ft_data']['config']['stim_duration']
    post_delay = data_class.data['ft_data']['config']['post_duration']
    stim_type = data_class.data['ft_data']['config']['stim_name']
    fob_df = data_class.data['ft_data']['stim_df']

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 8))
    # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 8))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.06, top=0.95)
    fig.suptitle(stim_type)
    ax0 = ax.twinx()
    plot_stim(ax0, stim_type, pre_delay, stim_time)
    sns.lineplot(data=fob_df, x='cur_t', y='va', ax=ax, errorbar='sd')
    ax.set_xlim(0, pre_delay + stim_time + post_delay)
    ax.set_xticks([0, pre_delay, pre_delay + stim_time * 0.5, pre_delay + stim_time])
    ax.set_xlabel('time(s)')
    ax.set_ylabel('va')

    # plt.show()
    plt.savefig(fig_path, dpi=600)
    plt.close()


def find_close_time(find_t, t_l, MAX_DELAY):
    for r, t in t_l:
        delay = find_t - t
        # print(delay)
        if abs(delay) < MAX_DELAY:
            print("find", r, delay)
            return r
    return None


def get_ext_exp_type(f, EXP_TYPE="2P"):
    if EXP_TYPE == "2P":
        parent = os.path.dirname(f)
        geno = parent.split("_")[-1]
        # geno = GENO_MAP.get(geno, geno)
        return "2P_" + geno
    return EXP_TYPE


def is_valid_date_str(x):
    if not x.startswith("2"):
        return False
    tt = x.split("_")
    if len(tt) == 2 and tt[1][0] in ["0", "1", "2"] and len(tt[0]) == 6 and len(tt[1]) == 6:
        return True
    return False


def cmd_to_time_glob(cmd):
    if cmd == "today":
        now = datetime.datetime.now()
        date_str = now.strftime("%y%m%d")
        time_glob = os.path.join(ROOT, date_str, "*", "*")
    elif cmd == "recent":  # most recent
        date_l = os.listdir(ROOT)
        date_str = sorted(filter(lambda x: x.startswith("2"), date_l))[-1]
        time_glob = os.path.join(ROOT, date_str, "*", "*")
        time_str_l = [os.path.basename(t) for t in glob(time_glob)]
        time_str = sorted(filter(is_valid_date_str, time_str_l))[-1]
        time_glob = os.path.join(ROOT, date_str, "*", time_str)
    elif len(cmd) < 7:  # date
        date_str = cmd
        time_glob = os.path.join(ROOT, date_str, "*", "*")
    elif len(cmd) < 25:
        if len(cmd.split("-")) > 1:  # fly
            date_str = cmd.split("-")[0]
            time_glob = os.path.join(ROOT, date_str, cmd + "*", "*")
        else:  # time
            date_str = cmd.split("_")[0]
            time_glob = os.path.join(ROOT, date_str, "*", cmd)
            if len(glob(time_glob)) == 0:
                ft_t = datetime.datetime.strptime(cmd, "%y%m%d_%H%M%S") - datetime.timedelta(days=1)
                date_str = datetime.datetime.strftime(ft_t, "%y%m%d")
                time_glob = os.path.join(ROOT, date_str, "*", cmd)
        # "\\192.168.1.38\nj\Imaging_data\221020\221020-M1-FT_CX1013G\221020_163927"
    else:
        time_glob = cmd
    return time_glob


def start_file(path):
    os.startfile(os.path.abspath(path))


def get_stim_range(on_stim, merge_inter=20):
    d = np.diff(on_stim.astype(int))
    on = np.nonzero(d == 1)[0]
    off = np.nonzero(d == -1)[0]
    ret = []
    last_o, last_f = -1000, -1000
    for o, f in zip(on, off):
        if last_o >= 0:
            if o - last_f < merge_inter:
                last_f = f
                continue
            else:
                ret.append([last_o, last_f])
        last_o = o
        last_f = f
    if len(ret) == 0 or ret[-1][-1] != last_f:
        ret.append([last_o, last_f])
    return ret


def load_tif(fname):
    from tifffile import tifffile
    return tifffile.TiffFile(fname).asarray()


def calc_avg_frame(m, parent):
    # if os.path.exists(parent + "/i_std.png"):
    #     return
    avg_frame = np.mean(m, axis=0)
    eq = norm_img(avg_frame)
    cv2.imwrite(parent + "/i_avg.png", eq)
    std_frame = np.std(m, axis=0)
    eq2 = norm_img(std_frame)
    cv2.imwrite(parent + "/i_std.png", eq2)
    max_frame = np.max(m, axis=0)
    eq3 = norm_img(max_frame)
    cv2.imwrite(parent + "/i_max.png", eq3)
    # eqc = np.transpose([eq3, eq2, eq], (1, 2, 0))
    # cv2.imwrite(parent + "/i_all.png", eqc)

    # cluster_roi(m, parent)
    # show_cluster_roi(m, parent)
    # th2, res = cv2.threshold(eq3.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imwrite(parent + "/i_test.png", res)


def cluster_roi(m, parent):
    eq3 = norm_img(np.max(m, axis=0)) * 2  # NOTE: m: 1400*128*128
    th2, img = cv2.threshold(eq3.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))
    p_l = np.nonzero(img)
    n = len(p_l[0])
    cluster_m = np.zeros(img.shape, dtype=np.uint8)
    # for i in range(n):
    #     print(i)
    #     for j in range(n):
    #         cor_map[i, j] = cor(m[:, p_l[0][i], p_l[1][i]], m[:, p_l[0][j], p_l[1][j]])
    X = m[:, p_l[0], p_l[1]].T  # NOTE: n*1400

    from pyclustering.cluster.kmeans import kmeans
    from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
    from pyclustering.utils.metric import type_metric, distance_metric
    metric = distance_metric(type_metric.USER_DEFINED, func=lambda a, b: 1 - cor(a, b))
    initial_centers = kmeans_plusplus_initializer(X, 10).initialize()
    kmeans_instance = kmeans(X, initial_centers, metric=metric)
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    pred = np.zeros((X.shape[0]))
    cf = []
    for i, c in enumerate(clusters):
        pred[c] = i + 1
        cf.extend(c)

    plot_fast_cor_map(X, cf)
    # plt.show()
    plt.savefig(parent + "/i_cor.png")

    # from sklearn.cluster import KMeans, DBSCAN
    # estimator = KMeans(n_clusters=20, max_iter=500)
    # estimator = DBSCAN(eps=0.8, min_samples=50, metric=lambda a, b: 1-cor(a, b))

    # estimator.fit(X)
    # pred = estimator.labels_

    cluster_m[p_l[0], p_l[1]] = pred
    cv2.imwrite(parent + "/i_test.png", norm_img(cluster_m))

    for i in range(int(np.max(pred))):
        show_largest_for_roi(m, cluster_m, i)
        plt.savefig(parent + "/i_largest_%d.png" % i)
    # plt.imshow(cluster_m)


def plot_cor_map(X, cf):
    n = len(cf)
    cor_map = np.zeros((n, n))
    for i, cf1 in enumerate(cf):
        print(i, "/", n)
        for j, cf2 in enumerate(cf):
            cor_map[i][j] = cor(X[cf1], X[cf2])
    plt.imshow(cor_map, cmap="bwr")
    plt.colorbar()


def plot_fast_cor_map(X, cf):
    from scipy.stats import zscore
    Xz = zscore(X, axis=1)
    n = len(cf)
    cor_map = np.zeros((n, n))
    for i, cf1 in enumerate(cf):
        print(i, "/", n)
        for j, cf2 in enumerate(cf):
            cor_map[i][j] = np.dot(Xz[cf1], Xz[cf2]) / n
    plt.imshow(cor_map, cmap="bwr")
    plt.colorbar()


def show_cluster_roi(m, parent):
    cluster_m = cv2.imread(parent + "/i_test.png")
    for k, i in enumerate(np.unique(cluster_m)):
        show_largest_for_roi(m, cluster_m, i)
        plt.savefig(parent + "/i_largest_%d.png" % k)


def show_largest_for_roi(m, cluster_m, label):
    p_l = np.nonzero(cluster_m == label)
    X = m[:, p_l[0], p_l[1]]
    n = X.shape[1]
    # if n < 1600:
    #     cor_map = np.zeros((n, n))
    #     for i in range(n):
    #         print(i, "/", n)
    #         for j in range(n):
    #             cor_map[i][j] = cor(X[:, i], X[:, j])
    #     plt.imshow(cor_map)
    #     plt.colorbar()
    #     plt.show()
    f = np.argmax(np.sum(X, axis=1))
    plt.figure(figsize=(6, 3))
    plt.title(str(f))
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(m[f])
    axs[0].scatter(p_l[1], p_l[0], alpha=1, color="r", s=1)
    axs[1].imshow(m[f])


def norm_img(img):
    pmax = np.percentile(img, 99)
    pmin = np.percentile(img, 1)
    r_norm = 255.0 * (img - pmin) / (pmax - pmin)
    np.clip(r_norm, 0, 255, r_norm)
    return r_norm.astype(np.uint8)


def load_roi(roi_file, shape):
    if roi_file.endswith(".zip"):
        return load_roi_zip(roi_file, shape)
    elif roi_file.endswith(".npy"):
        roi = np.load(roi_file, allow_pickle=True)
        names = np.arange(len(roi))
        xy = [r[:, 0, :] for r in roi]
        return names, roi_contours_to_points(xy, shape), xy


def load_roi_zip(roizip, shape, read_roi_zip=None):
    from read_roi import read_roi_zip  # read_roi_file
    rois = read_roi_zip(roizip)

    xy = []
    names = []
    for d in rois.values():
        if d["type"] == "polygon":
            xy.append(np.array(tuple(zip(d["x"], d["y"]))))
            names.append(d["name"])
        elif d["type"] == "rectangle":
            l, t, w, h = d["left"], d["top"], d["width"], d["height"]
            xy.append(np.array([[l, t], [l + w, t], [l + w, t + h], [l, t + h]]))
            names.append(d["name"])

    return names, roi_contours_to_points(xy, shape), xy


def calc_all_roi_F1(roizip, m, parent):
    print("Do calc F...")
    names, points, contours = load_roi(roizip, m[0].shape)
    ret = [[] for i in range(len(names))]
    for i, mi in enumerate(m):
        for j, xyj in enumerate(points):
            ret[j].append(float(mi[xyj].mean()))
    # plot_lines(ret, names)
    # plot_rois(contours, m)
    plt.savefig(parent + "/roi.png")

    F = pd.DataFrame(np.array(ret).T, columns=names)
    F.to_csv(parent + "/F.csv", index=False)
    plot_lines([F[n] for n in names], names)
    plt.savefig(parent + "/F.png")

    zs_csv = parent + "/zscore.csv"
    zs = get_zscore(F)
    zs.to_csv(zs_csv, index=False)

    # dFF_csv = parent + "/dFF.csv"
    # dFF = get_dFF(F)
    # dFF.to_csv(dFF_csv, index=False)

    ndFF_csv = parent + "/ndFF.csv"
    ndFF = get_ndFF(F)
    ndFF.to_csv(ndFF_csv, index=False)

    if ndFF.shape[1] == 11:
        plot_hot(ndFF.iloc[:, -8:], parent + "/ndFF_hot", 1)
    else:
        plot_hot(ndFF.iloc[:, -16:], parent + "/ndFF_hot", 1)
    # print(dFF_csv)
    return zs_csv


def calc_all_roi_F(roizip, m, parent, plot_frames=-1, ax=None):
    print("Do calc F...")
    names, points, contours = load_roi(roizip, m[0].shape)
    ret = [[] for i in range(len(names))]
    for i, mi in enumerate(m):
        for j, xyj in enumerate(points):
            ret[j].append(float(mi[xyj].mean()))
    # plot_lines(ret, names)
    # plot_rois(contours, m)
    # plt.savefig(parent + "/roi.png")

    F = pd.DataFrame(np.array(ret).T, columns=names)
    F.to_csv(parent + "/F.csv", index=False)

    # plt.figure()
    # plot_lines([F[n] for n in names], names)
    # plt.savefig(parent + "/F.png")

    # zs = get_zscore(F)
    # zs.to_csv(parent + "/zscore.csv", index=False)

    # ndFF = get_ndFF(F)
    # ndFF.to_csv(parent + "/ndFF.csv", index=False)

    # if ndFF.shape[1] == 11:
    #     plot_hot(ndFF.iloc[:plot_frames, -8:], parent + "/ndFF_hot", 1, ax=ax)
    # else:
    #     plot_hot(ndFF.iloc[:plot_frames, -16:], parent + "/ndFF_hot", 1, ax=ax)


COLORS = ["k", "r", "g", "b", "y", "c", "m", "gray", "pink", "springgreen", "deepskyblue", "yellow", ]


def plot_lines(lines, names, ylim=None, xlim_r=1):
    n = len(lines)
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(30, 12), dpi=300)
    plt.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.05, hspace=0)
    for i, r in enumerate(lines):
        ax = axes[n - i - 1]
        ax.plot(r, c=COLORS[i % 12])
        ax.set_ylabel(names[i], rotation=0, fontsize=6)
        if ylim is not None:
            ax.set_ylim(ylim)
        if xlim_r is not None:
            ax.set_xlim((0, xlim_r * len(r)))


def plot_pva(ax, dff, unwrap=False, c="k", alpha=1, offset=0):
    # max_idx = np.argmax(dff.T, axis=0) + offset
    pv_dir, pv_len = calc_pva(dff)
    if unwrap:
        pv_dir = unwrap_dir(pv_dir)
    pi = dff.shape[1] / 2
    plot_angle(ax, (np.array(pv_dir) + np.pi) * pi / np.pi + offset, c, pi, alpha=alpha)


def plot_hot(df, save_name, img_rate, is_PB=False, ax=None):
    ax0 = ax
    if ax0 is None:
        plt.figure(figsize=(24, 2), dpi=300)
        plt.subplots_adjust(left=0.02, right=0.99)
        ax = plt.gca()
    dff = df.to_numpy()
    ax.pcolor(dff.T, cmap="Blues")
    if is_PB:
        plot_pva(ax, dff[:, :9])
        plot_pva(ax, dff[:, 9:], offset=9)
    else:
        plot_pva(ax, dff, alpha=0.3)
    # frames = len(dff)
    # seconds = int(frames / img_rate)
    # labels = np.linspace(0, seconds, 9)
    # ax.set_xticks(labels * img_rate, labels)
    # plt.colorbar()
    if ax0 is None:
        plt.savefig(save_name + ".png")

    # plt.figure(figsize=(24, 2), dpi=300)
    # plt.subplots_adjust(left=0.02, right=0.99)
    # if is_PB:
    #     plot_pva(plt.gca(), dff[:, :9], unwrap=True)
    #     plot_pva(plt.gca(), dff[:, 9:], unwrap=True, c="gray", offset=9)
    # else:
    #     plot_pva(plt.gca(), dff, unwrap=True)
    # plt.xticks(labels * img_rate, labels)
    # plt.xlim(0, seconds * img_rate)
    # plt.savefig(save_name + "_unwrap.png")


def real_heading(heading, scr_width_angle=240):
    ret = lim_dir_l(np.array(heading))
    return ret / 360 * scr_width_angle


def lim_dir(dir1, pi=np.pi):
    if dir1 > pi:
        dir1 -= 2 * pi
    elif dir1 < -pi:
        dir1 += 2 * pi
    return dir1


def lim_dir_l(dir_l, pi=np.pi):
    dir1 = dir_l.copy()
    dir1[dir1 > pi] -= 2 * pi
    dir1[dir1 < -pi] += 2 * pi
    return dir1


def unwrap_dir(v, pi=np.pi):
    ret = []
    li = v[0]
    offset = 0
    for i in v:
        i += offset
        d = i - li
        if d > pi:
            offset -= 2 * pi
            i -= 2 * pi
        elif d < -pi:
            offset += 2 * pi
            i += 2 * pi
        ret.append(i)
        li = i
    return ret


def unwrap_dir_win(v, rate, sec=10):
    win = int(sec * rate)
    ret = []
    for i in range(0, len(v), win):
        u = unwrap_dir(v[i:i + win])
        ret.extend([d - u[0] for d in u])
    return ret


def plot_angle(ax, al, c, pi=np.pi, xs=None, lw=1, vertical=False, alpha=1):
    # al = unwrap_dir(al)  #[lim_dir(a-2) for a in al]
    if xs is None:
        xs = range(len(al))
    last = 0
    start = 0
    for i, a in enumerate(al):
        if abs(a - last) > pi * 1.5:
            if vertical:
                ax.plot(al[start:i], xs[start:i], c=c, lw=lw, alpha=alpha)
                ax.scatter(al[i], xs[i], c=c, s=0.1)
            else:
                ax.plot(xs[start:i], al[start:i], c=c, lw=lw, alpha=alpha)
                ax.scatter(xs[i], al[i], c=c, s=0.1)
            start = i
        last = a
    if vertical:
        ax.plot(al[start:], xs[start:len(al)], c=c, lw=lw, alpha=alpha)
    else:
        ax.plot(xs[start:len(al)], al[start:], c=c, lw=lw, alpha=alpha)


def plot_scatter(ax, al, c, xs=None):
    if xs is None:
        xs = range(len(al))
    ax.scatter(xs, al, c=c, s=0.1)


def rotate_unit_vec(rad):
    return np.array([np.cos(rad), np.sin(rad)])


def calc_pva(f):
    # pv_len = np.max(f, axis=1)
    # pv_dir = (np.argmax(f, axis=1) + 0.5) / 8 * 2 * np.pi - np.pi

    vs = np.array([rotate_unit_vec(2 * np.pi * (r + 0.5) / f.shape[1] - np.pi) for r in range(f.shape[1])])
    pv = f.dot(vs)
    pv_len = np.sqrt(np.sum(pv ** 2, axis=1))
    pv_dir = np.arctan2(pv[:, 1], pv[:, 0])
    return pv_dir, pv_len


def unify_sample(f, ts, fps, n):  # no interp
    idx = []
    inter = 1.0 / fps
    i = 0
    for j in range(n):
        t = j * inter
        if ts[i + 1] < t:
            while i < len(ts) - 1 and ts[i + 1] < t:
                i += 1
        if i + 1 >= len(f):
            break
        if t < ts[i]:
            idx.append(i)
            continue
        if i >= len(ts) - 1:
            break
        if t - ts[i] > ts[i + 1] - t:
            idx.append(i + 1)
        else:
            idx.append(i)
    return f[idx], idx


def calc_offset(ft, pv):
    n = min(len(ft), len(pv))
    o = ft[:n] - pv[:n]
    # o[o < 0] += 2*np.pi
    o[o > np.pi] -= 2 * np.pi
    o[o < -np.pi] += 2 * np.pi
    return o


def plot_slide_cor(ax, ft, pv, rate):
    for win_sec in [10]:  # 4, 16, 64
        win = int(win_sec * rate)
        ax.plot(*slide_cor(ft, pv, win, rate))


def circular_std(o):
    o = o[~np.isnan(o)]
    # s = np.sin(o)
    # c = np.cos(o)
    # return np.sqrt(-2 * np.log(np.sqrt(np.sum(s) ** 2 + np.sum(c) ** 2) / np.linalg.norm(o)))
    from scipy.stats import circstd
    return circstd(o)


def format_time(t):
    if t < 60:
        return "%.2f" % t
    return "%d:%.2f" % (t // 60, t % 60)


def non_nan(s):
    return s[~np.isnan(s)]


def load_ft_bar(bar_name):
    if not os.path.exists(bar_name):
        return None
    ft_info = {}
    if bar_name.endswith(".mat"):
        import scipy.io as sio
        bar_pos = sio.loadmat(bar_name)["bar_position"][0]
        p = np.min(np.nonzero(np.isnan(bar_pos)))
        return (bar_pos[:p] - 200) / 1400 * 2 * np.pi  # 200~1600
    else:
        f = open(bar_name, "r")
        m = []
        screen_start, screen_width = 64, 1648  # projector
        # screen_start, screen_width = 0, 256  # LED
        with_gap = False
        for line in f.readlines():
            if len(line) <= 1 or line[0] == "#" or line[0] == " ":
                print("ft_bar", line)
                if line.find("with_gap:1") >= 0:
                    with_gap = True
                if line.startswith("#screen_range"):
                    a = line.split(":")[-1].split("-")
                    screen_start, screen_width = int(a[0]), int(a[1])
                elif line.startswith("#stim_type"):
                    for a in line[1:-1].split():
                        k, v = a.split(":")
                        if v.find(",") > 0:
                            ft_info[k] = v
                        else:
                            ft_info[k] = float(v)
                    if ft_info.get("screen1") is not None:
                        screen_start = 0
                        scr1 = ft_info["screen1"].split(",")
                        scr2 = ft_info["screen2"].split(",")
                        scr1w = int(scr1[1]) - int(scr1[0])
                        scr2w = int(scr2[1]) - int(scr2[0])
                        screen_width = scr1w + scr2w
                continue
            t = line[:-1].split()
            m.append([float(tt) for tt in t])
        m = np.array(m)
        return bar_pos_to_angle(m[:, -2], screen_start, screen_width, with_gap), bar_pos_to_angle(m[:, -1],
                                                                                                  screen_start,
                                                                                                  screen_width), m[:,
                                                                                                                 1], ft_info


def bar_pos_to_angle(bp, screen_start, screen_wid, with_gap=False):
    if with_gap:
        return 2 * np.pi - ((bp - screen_start) * 1.5 * np.pi / screen_wid + np.pi / 4)
    else:
        return (bp - screen_start) * 2 * np.pi / screen_wid  # 64~1712


def load_NI_h5(NI_name, exp_info=None):
    ni = h5py.File(NI_name, "r")

    if "side_camera" in ni["AI"]:
        ft_out = ni['AI']["side_camera"][:, 0]
    elif "ai7" in ni["AI"]:
        ft_out = ni['AI']["ai7"][:, 0]
    elif "UVLED" in ni["AI"]:
        ft_out = ni['AI']["UVLED"][:, 0]
    FT_frame = find_raising_edge(ft_out).astype(int)

    frame_out = ni["DI"]["FrameOut"][:, 0]

    # pd_info_raw = ni["AI"]["photodiode"][:, 0]  # (1640499frame,) 5000Hz
    # ft_range = get_stim_range(pd_info_raw > 0.3)
    # PD_start, PD_end = int(ft_range[0][0]), int(ft_range[0][1])
    # plt.plot(pd_info_raw, "r")
    # plt.plot(frame_out/8, "g")
    # plt.plot(ft_out/12, "b")
    # plt.title("PD(%d, %d), FT(%d, %d)" % (PD_start, PD_end, FT_frame[0], FT_frame[-1]))
    # plt.show()

    if exp_info and exp_info.get("steps", 0) > 1:
        volume_frame = find_raising_edge(frame_out)
        volume_step = exp_info["steps"] + exp_info["flybackFrames"]
        dFF_frame = volume_frame[int(exp_info["steps"] / 2)::volume_step].astype(int)
    else:
        dFF_frame = find_raising_edge(frame_out)

    return {"rate": SYNC_RATE, "FT_frame": FT_frame, "dFF_frame": dFF_frame}


def load_NI_h5_old(NI_name, exp_info, FT_l, show_raise=False):
    import h5py
    ni = h5py.File(NI_name, "r")

    pd_info_raw = ni["AI"]["photodiode"][:, 0]  # (1640499frame,) 5000Hz
    ft_range = get_stim_range(pd_info_raw > 0.14)
    PD_start, PD_end = int(ft_range[0][0]), int(ft_range[0][1])

    if "UVLED" in ni["AI"]:
        ft_out = ni['AI']["UVLED"][:, 0]
        FT_frame = find_raising_edge(ft_out).astype(int)
    else:
        ft_out = None
        FT_frame = np.linspace(PD_start, PD_end, FT_l).astype(int)

    frame_out = ni["DI"]["FrameOut"][:, 0]

    # plt.plot(pd_info_raw, "r")
    # plt.plot(frame_out/8, "g")
    # plt.plot(ft_out/12, "b")
    # plt.show()

    volume_frame = find_raising_edge(frame_out)
    volume_step = exp_info["steps"] + exp_info["flybackFrames"]
    dFF_frame = volume_frame[int(exp_info["steps"] / 2)::volume_step].astype(int)

    if show_raise:
        for i in [0]:  # , -1]:
            plt.figure()
            idx_raise = np.arange(FT_frame[i] - int(SYNC_RATE / 8), FT_frame[i] + int(SYNC_RATE))
            plt.plot(idx_raise, frame_out[idx_raise] / 8, "g")
            plt.plot(idx_raise, pd_info_raw[idx_raise], "r")
            if ft_out is not None:
                plt.plot(idx_raise, ft_out[idx_raise] / 12, "b")
            plt.scatter(dFF_frame, np.zeros((len(dFF_frame),)), c="g")
            plt.scatter(FT_frame, np.zeros((len(FT_frame),)), c="b", marker="x")
            plt.xlim(idx_raise[0], idx_raise[-1])
            ticks = np.linspace(idx_raise[0], idx_raise[-1], 10)
            plt.xticks(ticks, ["%.2f" % t for t in ticks / SYNC_RATE])
        plt.show()
    return {"rate": SYNC_RATE, "length": len(frame_out), "PD_start": PD_start, "PD_end": PD_end, "FT_frame": FT_frame,
            "dFF_frame": dFF_frame}


def rolling_mean(s, w):
    ret = pd.Series(s).rolling(w).mean()
    for i in range(w):
        ret[i] = ret[w - 1]
    return ret


def get_part(fname):
    if fname.find("-PB-") >= 0 or fname.find("_PB_") >= 0:
        return "PB"
    elif fname.find("_FB_") >= 0:
        return "FB3"
    elif fname.startswith("PF"):
        return "FB3"
    else:
        return "EB"


def get_dFF_by_baseline_range(F, baseline_range, bg_idx=0):
    names = F.columns
    bg = F[names[bg_idx]].copy()
    for g in names:
        F[g] = F[g] - bg
        baseline = np.mean(F[g][baseline_range[0]:baseline_range[1]])
        F[g] = (F[g] - baseline) / baseline
    return F


def get_dFF(F):
    names = F.columns
    bg = F[names[0]]
    for g in names[1:]:
        f = F[g] - bg
        baseline = np.mean(f[f <= np.percentile(f, 5)])  # NOTE: F as the mean of the lower 5% (MaimonG_Nat17)
        if baseline <= 1:
            baseline = 1
        print(g, "baseline", baseline)
        F[g] = (f - baseline) / baseline
    return F


def get_ndFF(F):
    F = F.copy()
    names = F.columns
    bg = F[names[0]]
    for g in names[1:]:
        f = F[g] - bg
        f0 = np.mean(f[f <= np.percentile(f, 5)])  # NOTE: lower 5% (MaimonG_bioRxiv20)
        fm = np.mean(f[f >= np.percentile(f, 97)])  # NOTE: top 3% (MaimonG_bioRxiv20)
        # if baseline <= 1:
        #     baseline = 1
        F[g] = (f - f0) / (fm - f0)
    return F


def get_zscore(F):
    from scipy.stats import zscore
    zs = zscore(F, axis=0)
    return pd.DataFrame(zs, columns=F.columns)


def zscore1(xs):
    m, s = np.nanmean(xs), np.nanstd(xs)
    return (xs - m) / s


def scale_x(xs, rmin, rmax):
    r_dot = np.array(xs)
    r0, r1 = np.nanmin(r_dot), np.nanmax(r_dot)
    return rmin + (r_dot - r0) * (rmax - rmin) / (r1 - r0)


def max_dist(xs, ys):
    from scipy import spatial
    pts = np.vstack([xs, ys]).T
    candidates = pts[spatial.ConvexHull(pts).vertices]
    dist_mat = spatial.distance_matrix(candidates, candidates)
    # i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
    return dist_mat.max()


def cov1(xs, ys):
    n = len(xs)
    mx, my = np.mean(xs), np.mean(ys)
    return (xs - mx).dot(ys - my) / (n - 1)


def cor1(xs, ys):
    n = len(xs)
    return zscore1(xs).dot(zscore1(ys)) / (n - 1)


def cor2(xs, ys):
    return cov1(xs, ys) / np.std(xs) / np.std(ys)


def cir_cor(xs, ys):  # circstat circ_corrcc
    mx, my = circmean(xs), circmean(ys)
    num = np.sin(xs - mx).dot(np.sin(ys - my))
    den = np.sqrt(np.sum(np.sin(xs - mx) ** 2) * np.sum(np.sin(ys - my) ** 2))
    return num / den


def auto_cor(xs):
    xm = xs - xs.mean()
    xn = np.sum(xm ** 2)
    return np.correlate(xm, xm, "same") / xn


def cor(x, y):
    # return np.dot(x, y)/np.count_nonzero(y>0.5)
    r1 = np.corrcoef(x, y)[0][1]
    r, p = pearsonr(x, y)
    print("corr", r1, "pearsonr", r, p)
    return r


def cor_equal_len(x, y):
    if len(x) > len(y):
        return cor(down_sample(x, len(y)), y)
    else:
        return cor(x, down_sample(y, len(x)))


def slide_cor(x, y, win, rate):
    n = min(len(x), len(y))
    rx, ry = [], []
    for i in range(0, n - win, int(rate)):
        c = cor(unwrap_dir(x[i:i + win]), unwrap_dir(y[i:i + win]))
        rx.append(i + win / 2.0)
        ry.append(c)
    return rx, ry


def down_sample(x, n):
    x = np.array(x, dtype=np.float)
    index_arr = np.linspace(0, len(x) - 1, num=n, dtype=np.float)
    index_floor = np.array(index_arr, dtype=np.int)
    index_ceil = index_floor + 1
    index_rem = index_arr - index_floor

    val1 = x[index_floor]
    val2 = x[index_ceil % len(x)]
    interp = val1 * (1.0 - index_rem) + val2 * index_rem
    assert (len(interp) == n)
    return interp


def bin_data(xs, ys, start, end, step):
    bin_l = np.arange(start, end, step)
    n = len(bin_l)
    bt = []
    bd = []
    for i in range(n):
        bd.append([])
    for x, y in zip(xs, ys):
        bin_idx = (x - start) / step
        if not np.isnan(bin_idx):
            bin_idx = int(bin_idx)
            if 0 <= bin_idx < n:
                bt.append((bin_l[bin_idx], y))
                bd[bin_idx].append(y)
    return bin_l, bt, bd


def up_sample2(x, in_frame, out_frame, return_idx=False):
    ret = []
    idx = []
    i = 0
    in_len = len(in_frame)
    x_len = len(x)
    while out_frame[0] >= in_frame[i]:
        i += 1
    for o in out_frame:
        i1 = in_frame[i]
        i0 = in_frame[i - 1]
        if o == i1:
            ret.append(x[i])
            idx.append(i)
            continue
        elif o > i1:
            if i < in_len - 1:
                i += 1
            else:
                ret.append(x[i])
                idx.append(i)
                continue
            i1 = in_frame[i]
            i0 = in_frame[i - 1]

        if o < i0:
            ret.append(x[i - 1])
            idx.append(i - 1)
        else:
            if i >= x_len:
                ret.append(x[-1])
                idx.append(x_len - 1)
            else:
                ret.append(x[i - 1] + (x[i] - x[i - 1]) * (o - i0) / (i1 - i0))
                # ret.append(x[i] if o-i0 > i1-o else x[i-1])
                idx.append(i if o - i0 > i1 - o else i - 1)
    # plt.plot(in_frame, x, "b.-")
    # plt.plot(out_frame, ret, "r.-")
    # plt.show()
    if return_idx:
        return np.array(ret), np.array(idx)
    return np.array(ret)


def down_sample2(x, in_frame, out_frame, circ=False):
    steph = int((out_frame[1] - out_frame[0]) / 2)
    i = 0
    ret = []
    last = np.nan
    for o in out_frame:
        if i > len(in_frame):
            ret.append(last)
        else:
            in_x = []
            while i < len(in_frame) and o + steph > in_frame[i]:
                if o - steph < in_frame[i]:
                    in_x.append(x[i])
                i += 1
            if len(in_x):
                if circ:
                    ret.append(circmean(in_x))
                else:
                    ret.append(np.mean(in_x))
                last = ret[-1]
            else:
                if i == len(in_frame):
                    ret.append(np.nan)
                else:
                    ret.append(last)
    return np.array(ret)


def up_sample2d(x, in_frame, out_frame):
    ret = []
    for c in range(x.shape[1]):
        ret.append(up_sample2(x[:, c], in_frame, out_frame))
    return np.vstack(ret).T


def down_sample2d(x, in_frame, out_frame):
    ret = []
    for c in x.shape[1]:
        ret.append(down_sample2(x[:, c], in_frame, out_frame))
    return np.hstack(ret)


def view_img_seq(imgs, shifts_rig=None):
    global g_total_frame, g_frame, g_is_input_begin, g_input_int
    g_total_frame, h, w = imgs.shape
    g_frame = 0
    g_is_input_begin = False
    g_input_int = 0

    def plot_one_frame(f):
        global g_frame
        g_ax.cla()
        g_ax.imshow(imgs[f].astype(int), cmap=plt.cm.gray, norm=NoNorm())
        # g_ax.set_xlabel("%02d:%02.2f" % (t_sec / 60, t_sec % 60))
        g_ax.set_xlabel(str(f))
        if shifts_rig is not None:
            g_ax.set_title(str(shifts_rig[g_frame]))
        g_ax.grid(True)
        g_ax.set_xticks(np.linspace(0, imgs[f].shape[1], 13))
        g_frame = f

    def on_slider(val):
        plot_one_frame(int(val))

    def onkey(event):
        print(event.key)
        global g_frame, g_is_input_begin, g_input_int
        if event.key == "left":
            g_frame -= 1
        elif event.key == "right":
            g_frame += 1
        elif event.key == "enter":
            g_frame = g_input_int
            g_input_int = 0
        elif event.key in list([*"1234567890"]):
            g_input_int = int(event.key) + g_input_int * 10
            print("input: %d" % g_input_int)
            return
        else:
            g_input_int = 0
            return
        if g_frame >= g_total_frame:
            g_frame = g_total_frame - 1
        if g_frame < 0:
            g_frame = 0
        g_slider.set_val(g_frame)
        event.canvas.draw()

    from matplotlib.widgets import Slider
    fig, g_ax = plt.subplots(figsize=(w / 25, h / 25))
    plt.subplots_adjust(top=0.95, bottom=0.1)
    g_slider = Slider(plt.axes([0.1, 0.03, 0.8, 0.03]), "", valmin=0, valmax=g_total_frame - 1, valfmt="%d", valinit=0)
    g_slider.on_changed(on_slider)
    plot_one_frame(0)
    fig.canvas.mpl_connect('key_press_event', onkey)


def load_exp_xml(xml):
    if not os.path.exists(xml):
        return None

    def get_value(ss, key):
        p = ss.find(key + "=")
        return ss[p:].split("\"")[1] if p >= 0 else None

    s = open(xml, "r").readlines()
    ret = {}
    for ss in s:
        if ss.find("ThorZPiezo") >= 0:
            ret["steps"] = int(get_value(ss, "steps"))
            ret["stepSizeUM"] = float(get_value(ss, "stepSizeUM"))
            ret["startPos"] = float(get_value(ss, "startPos"))
        elif ss.find("<Streaming") >= 0:
            ret["frames"] = int(get_value(ss, "frames"))
            ret["zFastEnable"] = int(get_value(ss, "zFastEnable"))
            ret["flybackFrames"] = int(get_value(ss, "flybackFrames"))
        elif ss.find("<PMT") >= 0:
            ret["gainA"] = float(get_value(ss, "gainA"))
        elif ss.find("<Pockels") >= 0:
            if ss.find("maskEnable") > 0:
                ret["pockels"] = float(get_value(ss, "start"))
        elif ss.find("<LSM") >= 0:
            ret["name"] = get_value(ss, "name")
            ret["pixelX"] = int(get_value(ss, "pixelX"))
            ret["pixelY"] = int(get_value(ss, "pixelY"))
            ret["pixelSizeUM"] = float(get_value(ss, "pixelSizeUM"))
            ret["frameRate"] = float(get_value(ss, "frameRate"))  # 167.569
    if ret.get("zFastEnable"):
        ret["volume_rate"] = ret["frameRate"] / (ret["steps"] + ret["flybackFrames"])
    else:
        ret["volume_rate"] = ret["frameRate"]
    return ret


def find_raising_edge(s, th=0.5, min_inter=10):
    d = np.diff(s.astype(float))
    ret = np.nonzero(d > th)[0]
    return ret[np.concatenate([[True], np.diff(ret) > min_inter])]


def write_video(path, m, cvt=cv2.COLOR_GRAY2BGR, fps=FT_RATE, need_time=True):
    h, w = m[0].shape[:2]
    output_video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"DIVX"), fps, (w, h))
    for i, mi in enumerate(m):
        img_bgr = cv2.cvtColor(mi, cvt)
        if need_time:
            cv2.putText(img_bgr, format_time(i / fps), (6, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)
        output_video.write(img_bgr)
    output_video.release()


def one_z_slice(fname, n, i):
    import tifffile
    with tifffile.TiffFile(fname) as tffl:
        input_arr = tffl.asarray()  # 25200*128*128
        tn = int(len(input_arr) / n)
        ch = input_arr[i:tn * n:n]
        fname = fname + "_ch%d.tif" % i
        write_video(fname + "_ch%d.avi" % i, norm_img(ch), fps=30)
        # tifffile.imwrite(fname, ch)
        calc_avg_frame(ch, os.path.dirname(fname))
        return fname


def merge_tif_files(files, output):
    import tifffile
    ch = []
    for fname in files:
        tffl = tifffile.TiffFile(fname)
        input_arr = tffl.asarray()
        ch.append(input_arr)
        tffl.close()
    ch = np.array(ch)
    write_video(output + ".avi", norm_img(ch), fps=30)
    tifffile.imwrite(output, ch)
    calc_avg_frame(ch, os.path.dirname(output))


def get_slice_n(fname):
    exp_xml = os.path.dirname(fname) + "/Experiment.xml"
    exp_info = load_exp_xml(exp_xml)
    if exp_info.get("zFastEnable") and fname.find("tif_ch") < 0:
        n = exp_info["steps"]
    else:
        n = 1
    fps = exp_info["volume_rate"]
    if n > 20:  # Z_stack
        n = 1
        fps = 30
    return n, fps


def split_z_slices(fname):
    print("split z", fname)
    # import tifffile
    # with tifffile.TiffFile(fname) as tffl:
    #     input_arr = tffl.asarray()

    input_arr = load_mmap(fname)
    # input_arr = load_tif(fname)

    chs = []
    tn = len(input_arr)
    chs.append(input_arr[0:tn:1])
    return chs


def merge_z_slices(fname):
    # import tifffile
    # with tifffile.TiffFile(fname) as tffl:
    #     input_arr = tffl.asarray()  # 25200*128*128
    # # write_video(fname + ".avi", norm_img(input_arr))
    # # plt.figure()
    # # a = input_arr[0].flatten()
    # # plt.hist(a[a>0.01], bins=50)
    # # plt.savefig(fname + "_hist.png")

    input_arr = load_mmap(fname)
    # input_arr = load_tif(fname)

    tn = len(input_arr)
    ch = input_arr[0:tn:1]
    # avg_frame = np.mean(ch0, axis=0)
    # w = 255.0 / np.max(avg_frame)
    # print(w)
    # eq = ch0  # * w*w
    # eq[eq > 0] = 255

    # tifffile.imwrite(fname + "_ch%d.tif" % i, eq)
    # write_video(fname + "_ch%d.avi"%i, norm_img(eq))
    # cv2.imwrite(fname.split('.')[0] + "_avg%d.png" % i, norm_img(np.mean(eq, axis=0)))

    # if method == "mean":
    #     avg_tif = np.mean(chs, axis=0)
    # else:
    #     avg_tif = np.max(chs, axis=0)
    # if n > 1:
    #     tifffile.imwrite(fname + "_avg.tif", avg_tif)
    # write_video(fname.split('.')[0] + ".avi", norm_img(avg_tif), fps=fps)

    # pix = avg_tif.flatten()
    # plt.figure()
    # plt.hist(pix[pix > 1], range=(1, 2000), bins=100)
    # plt.savefig(fname + "_hist.png")
    avg_tif = ch
    calc_avg_frame(avg_tif, os.path.dirname(fname))
    return fname + "_avg.tif"


def merge_info(date_folder):
    info_l = []
    for f in glob(date_folder + "/*/info.txt"):
        c = json.load(open(f, "r"))
        print(f, c)
        exp_name = os.path.basename(os.path.dirname(f))
        t = exp_name.split("-")
        pair = "-".join(t[1:-1])
        trail = t[-1]
        stim = t[-2]
        part = t[-3]
        fly = t[2]
        info_l.append([pair, fly, trail, part, stim, c["cor_unwrap"], c["cor_cir"]])
    df = pd.DataFrame.from_records(info_l, columns=["pair", "fly", "trail", "part", "stim", "cor_unwrap", "cor_cir"])
    import seaborn as sns
    for x in ["cor_unwrap", "cor_cir"]:
        plt.figure()
        sns.set_theme(style="darkgrid")
        sns.catplot(x=x, y="stim", hue="fly", col="part", data=df)
        plt.xlim(-1, 1)
        plt.tight_layout()
        plt.savefig(date_folder + "/" + x)


def smooth_angle(s, win):
    s = np.array(s)
    hw = int(win / 2)
    s1 = np.concatenate([[s[0]] * hw, s, [s[-1]] * hw])
    ret = []
    for i in range(len(s)):
        m = circmean(s1[i:i + win])
        if m > np.pi:
            m -= np.pi * 2
        ret.append(m)
    # m0 = ret[0]
    # for i in range(hw):
    #     ret.insert(0, m0)
    return np.array(ret)


def diff_angle(a1, a2):
    return lim_dir_l(a1 - a2)


def calc_bout(s):  # 00011100111100
    d = np.diff(s.astype(int))  # 00100-01000-0
    start = np.nonzero(d > 0)[0]  # [2, 7]
    end = np.nonzero(d < 0)[0]  # [5, 11]
    if len(start) < len(end):
        start = np.concatenate([[-1], start])
    elif len(start) > len(end):
        end = np.append(end, len(d) - 1)
    return tuple(zip(start + 1, end + 1))  # [[3, 6], [8, 12]


def proc_ft_pv(fname):
    c = json.load(open(fname, "r"))
    t_range, heading, pv, pvl, speed, vx, vy, vx, pv2, pvl2 = c
    n = len(heading)
    ts = np.linspace(t_range[0], t_range[1], n)
    # plot_lines(c[:7], "heading pv pvl speed vx vy vz".split())
    pvs = smooth_angle(pv, 10)
    pvs2 = smooth_angle(pv2, 10)

    fig, axs = plt.subplots(3, 1, figsize=(8, 4), sharex=True, dpi=300)
    # plt.subplots_adjust(left=0.02, right=0.99)
    axs[0].plot(ts, unwrap_dir(np.array(heading)), c="m", lw=0.5)
    axs[0].plot(ts, unwrap_dir(pvs), "k--", lw=0.5)
    axs[0].plot(ts, unwrap_dir(pv), c="k", lw=0.5)
    # axs[0].plot(ts, unwrap_dir(pvs2), c="gray", lw=0.5)
    print(cor(unwrap_dir(np.array(heading)), unwrap_dir(pvs)))
    # axs[0].plot(ts, unwrap_dir(pv2), c="gray", lw=0.5)
    # axs[1].plot(ts, -np.array(heading), c="m", lw=0.5)
    axs[1].plot(ts, pv, c="k", lw=0.5)
    # axs[1].plot(ts, pvs, c="g", lw=0.5)
    axs[1].plot(ts, pv2, c="gray", lw=0.5)
    axs[2].plot(ts, zscore1(speed), c="m", lw=0.5)
    axs[2].plot(ts, zscore1(pvl), c="k", lw=0.5)
    axs[2].plot(ts, zscore1(pvl2), c="gray", lw=0.5)
    plt.tight_layout()
    # plt.show()
    plt.savefig(fname + ".png")


def rename_folder(folder):
    # for f in os.listdir(folder):
    #     if f.startswith("F"):
    #         idx = int(f.split("-")[-1])
    #         shutil.move(os.path.join(folder, f), os.path.join(folder, "F0-%d" % (idx)))
    # if f.startswith("F13_FB_PFNd_CX1015_OPCLS_"):
    #     idx = int(f.split("_")[-1])
    #     if idx <= 10:
    #         shutil.move(os.path.join(folder, f, "zscore.csv.pickle"), os.path.join(folder, f, "zscore.no_dot.pickle"))
    # shutil.move(os.path.join(folder, f), os.path.join(folder, f.replace("F13_FB_PFNd_CX019_OPCLS", "F13_FB_PFNd_CX1015_OPCLS")))
    # return
    for f in os.listdir(folder):
        if f.startswith("F10_FB_PFNv_CX1018G7b_OPCLS_"):  # and not f.endswith("OPCLS"):
            f1 = "_TMP_" + f
            # print(f1)
            shutil.move(os.path.join(folder, f), os.path.join(folder, f1))
    for f in os.listdir(folder):
        if f.startswith("_TMP_"):
            # pr, n = f[5:].split("-")
            # n = int(n)
            # if n > 36:
            #     shutil.move(os.path.join(folder, f), os.path.join(folder, "F7-%d" % (n-36)))
            # else:
            #     shutil.move(os.path.join(folder, f), os.path.join(folder, "F6-%d" % (n)))

            pos = f.rfind("_")
            pr, n = f[5:pos + 1], int(f[pos + 1:])
            # if n > 4:
            #     n -= 1
            #     f1 = pr.replace("F6", "F7") + "%03d" % (n - 7)
            # else:
            f1 = pr + "%03d" % (n + 1)  # .replace("CX1013", "CX019")
            # print(f1)
            shutil.move(os.path.join(folder, f), os.path.join(folder, f1))


def copy_to_all(date_folder, all_folder, fname="zscore.csv.pickle.png"):
    d = os.path.basename(date_folder)
    for f in os.listdir(date_folder):
        # for ff in ["zscore.csv.pickle.png", "zscore.csv.pickle.png"]
        png = os.path.join(date_folder, f, fname)
        if os.path.exists(png):
            print("copy", f)
            shutil.copy2(png, os.path.join(all_folder, d + "_" + f + ".png"))


def png_merge(png_ll, name, cols=0):
    if isinstance(png_ll[0], str):
        n = len(png_ll)
        if cols == 0:
            cols = int(np.sqrt(n) + 0.5)
        rows = n / cols
        if rows * cols < n:
            cols += 1
        png_ll1 = []
        for i in range(0, n, cols):
            png_ll1.append(png_ll[i: i + cols])
        png_ll = png_ll1
    cols = max([len(l) for l in png_ll])
    rows = len(png_ll)
    h, w, _ = cv2.imread(png_ll[0][0]).shape
    png_all = np.zeros((h * rows, w * cols, 3))
    for i, png_l in enumerate(png_ll):
        for j, png in enumerate(png_l):
            img = cv2.imread(png)
            if img is None:
                continue
            h1, w1, _ = img.shape
            if h1 != h or w1 != w:
                continue
            png_all[i * h:i * h + h, j * w:j * w + w] = img
    cv2.imwrite(name, png_all)


def get_zs_pickle_l(name, pk_name="zscore.csv.pickle"):  # 220112_1_0
    root = r"\\192.168.1.15\nj\Imaging_data"
    root = r"\\192.168.1.15\lqt\LQTdata\NJ"
    d = name.split("_")
    if len(d) == 3:
        if d[2] == "0":
            return glob(root + r"\%s\\*F%s_*OPCLS\%s" % (d[0], d[1], pk_name))
        return glob(root + r"\%s\\*F%s_%03d\%s" % (d[0], d[1], int(d[2]), pk_name))
    elif len(d) == 2:
        return glob(root + r"\%s\\*F%s_*\%s" % (d[0], d[1], pk_name))
    elif len(d) == 1:
        return glob(root + r"\%s\\*\%s" % (d[0], pk_name))


def correct_ft_lost_idx(ft_ts1, ft_ts2):
    # NOTE: ts1[fictrac camera ts], ts2[camera trigger ts]
    i = 0
    idx_re = []
    for t1 in ft_ts1:
        if i < len(ft_ts2):
            t2 = ft_ts2[i]
            if t1 - t2 > 2 / FT_RATE:  # 0.04 for FT fps 50
                print("    correct lost idx at:", t1, t2)
                while t1 > t2 and i < len(ft_ts2) - 1:
                    i += 1
                    t2 = ft_ts2[i]
            else:
                i += 1
        idx_re.append(i - 1)
    return idx_re


def load_imaging_data(exp_folder):
    # NOTE: 2p_info, rate, FT_frame, dFF_frame, dFF_df, zscore_df
    roi_file_l = glob(exp_folder + "/*/roi.npy")
    if len(roi_file_l) <= 0:
        print("roi file not found! [need run _0_function_roi.py]")
        return {}
    if len(roi_file_l) > 1:
        print("found multiple roi files! use", roi_file_l[0])
    roi_file = roi_file_l[0]
    parent = os.path.dirname(roi_file)

    tif_file = os.path.join(parent, "Image_scan_1_region_0_0.tif_avg.tif")
    if not os.path.exists(tif_file):
        tif_file = os.path.join(parent, "Image_scan_1_region_0_0.tif")
    # if not os.path.exists(tif_file):
    #     tif_file = os.path.join(parent, "ChanA_001_001_001_001.tif")
    # tif_data = load_tif(tif_file)
    # draw_roi_png(parent)
    # calc_all_roi_F(roi_file, tif_data, parent)

    # dff_df = pd.read_csv(parent + "/ndFF.csv").to_numpy()[:, 1:]
    # (500frame, 10roi) 5.383Hz self.data["2p_info"]["frameRate"]
    # zscore_df = pd.read_csv(parent + "/zscore.csv").to_numpy()[:, 1:]
    F_df = pd.read_csv(parent + "/F.csv")
    names = F_df.columns
    bg = F_df[names[0]]
    for g in names[1:]:
        F_df[g] = F_df[g] - bg
    F_df = F_df.to_numpy()[:, 1:]
    data = {"F_df": F_df}

    ts_file_l = glob(exp_folder + "/*/Episode001.h5")
    if len(ts_file_l) <= 0:
        print("ts file not found!")
        return {}
    else:
        ts_file = ts_file_l[0]
        data["2p_info"] = load_exp_xml(os.path.join(parent, "Experiment.xml"))
        frame_info = load_NI_h5(ts_file, data["2p_info"])
        data["rate"] = frame_info["rate"]
        data["FT_frame"] = frame_info["FT_frame"]
        data["dFF_frame"] = frame_info["dFF_frame"]

    # dff_frames = len(data["dFF_frame"])
    # if dff_frames != len(dff_df):
    #     print("!!! dFF frames =! images")
    #     data["dFF_df"] = dff_df[:dff_frames]
    #     data["zscore_df"] = zscore_df[:dff_frames]
    return data


def correct_ft_frame(stim_df, ft_frame_ni, rate):
    # correct_ft_frame(self.data["stim_df"], self.data["FT_frame"], self.data["rate"])
    idx = stim_df["cnt"].tolist()  # NOTE: cnt not match FT_frame (but match .dat:ft_ts)
    cnt0 = idx[0]
    # if (stim_df["ts"][1] - stim_df["ts"][0]) == 0:
    ft_ts1 = stim_df["cur_t"].to_numpy()  # NOTE: matlab ts
    # else:
    #     ft_ts1 = stim_df["ts"].to_numpy()  # NOTE: camera ts
    # NOTE: error ~0.002s after 3000 frames(50s), trigger additional ~12 frames at termination

    ft_frame = ft_frame_ni[cnt0:]
    ft_ts2 = (ft_frame - ft_frame[0]) / rate
    ft_ts1 = ft_ts1 - ft_ts1[0]
    # lost_frame = len(ft_ts2) - len(ft_ts1)
    # print("delay frame:", cnt0)
    # if lost_frame > 15:
    #     print("!!! lost frame: %d frames" % lost_frame)

    idx_real = np.array(correct_ft_lost_idx(ft_ts1, ft_ts2)) + cnt0

    stim_df["FT_frame"] = ft_frame_ni[idx_real]  # NOTE: exception if ft not within dFF

# if __name__ == '__main__':
#     # ft, pv = json.load(open(r"D:\exp_2p\data\EPG\210823\CX1001-7F-F6-PB-reverseCLOSED-7+\ft_pv.txt", "r"))
#     # print(cor(ft, pv))
#     # print(cor(unwrap_dir(ft), unwrap_dir(pv)))
#     # print(cir_cor(ft, pv))
#     # rename_folder(r"\\192.168.1.15\lqt\LQTdata\NJ\211228")
#     # rename_folder(r"\\192.168.1.15\nj\Imaging_data\220129")
#     # rename_folder(r"D:\exp_2p\anatomy\img")
#     # copy_to_all(r"\\192.168.1.15\nj\Imaging_data\220112", r"D:\exp_2p\data\all")
#     # copy_to_all(r"\\192.168.1.15\nj\FoB_data\211216", r"D:\exp_2p\data_FoB\all", "stim.txt.png")
#
#     # plot_fictrac(r"\\192.168.1.15\nj\FoB_data\211230\F1-1")
#     # plot_fictrac(r"\\192.168.1.15\nj\FoB_data\211216\F6-1")
#     # plot_fictrac(glob(r"\\192.168.1.15\nj\FoB_data\211215\F0-*"))
#     # plot_fictrac(glob(r"\\192.168.1.15\nj\FoB_data\211216\F7-*"))
#     # plot_fictrac(glob(r"\\192.168.1.15\lqt\LQTdata\NJ\211129\F5-*"))
#     # d = "220112"
#     # plot_scatter_line(get_zs_pickle_l("220112_1_0"), "220112_1_0")
#     # for n in range(10):
#     #     plot_scatter_line(glob(r"\\192.168.1.15\nj\Imaging_data\%s\\*F%d_*\zscore.csv.pickle"%(d,n)), "%s_F%d"%(d,n))
#         # for i, f in enumerate(glob(r"\\192.168.1.15\nj\Imaging_data\%s\\*F%d_*\zscore.csv.pickle"%(d,n))):
#         #     plot_scatter_line([f], "%s_F%d_%d" % (d, n, i))
#         #     plt.close("all")
#     # plot_scatter_line(glob(r"\\192.168.1.15\nj\Imaging_data\%s\\*F*\zscore.csv.pickle"%d), "%s"%d)
#     # png_merge(get_zs_pickle_l("211102", "i_std.png"), "img/211102.png")
#     # merge_tif_files(glob(r"D:\exp_2p\data_2p\220915_GG2p\220915-1\ChanA_001_001_001_*.tif")[1:], r"D:\exp_2p\data_2p\220915_GG2p\220915-1\Image.tif")
#     # import sys
#     # one_z_slice(sys.argv[1], 1, 0)
#     # one_z_slice(sys.argv[1], 5, 2)
#     # merge_z_slices(sys.argv[1], 5, "max")
#     # load_NI_h5(r"\\192.168.1.38\nj\Imaging_data\221025\221025-M4-FT_CX1013\221025_214414\221025_M4_TS029\Episode001.h5")
#
#     for f in glob(r"\\192.168.1.38\nj\Imaging_data\221129\221129-M2-FT_CX019\*\*\Experiment.xml"):
#         exp = load_exp_xml(f)
#         print(os.path.basename(os.path.dirname(os.path.dirname(f))), exp["gainA"], exp["pockels"])

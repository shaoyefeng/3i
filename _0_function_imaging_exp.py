
import os
import numpy as np
import pandas as pd
import matplotlib.colorbar as pcb
import matplotlib.pyplot as plt
from glob import glob
from scipy.stats import circstd

from _0_constants import SYNC_RATE, FT_RATE, UNIFY_FPS, FILTER_MIN_DURATION, FILTER_MIN_SPEED, ROOT, FILTER_MAX_TSPEED, \
    IS_FOB
from _0_function_base_exp import BaseExp
from _0_function_FoB import load_fob_dat
from _0_function_analysis import correct_ft_frame, load_imaging_data, calc_pva, up_sample2, \
    unwrap_dir, smooth_angle, lim_dir_l, real_heading, scale_x, diff_angle, cir_cor, slide_cor, down_sample2, \
    up_sample2d, down_sample2d, unify_sample, max_dist, load_exp_xml, load_NI_h5, calc_bout
from nea_video import write_stim_video, crop_video, stack_video, write_ft_vel_video, write_ima_video

from plot_utils import save_fig, scatter_lines_with_err_band, plot_lines_with_err_band, plot_angle, rolling_mean, \
    COLOR_V, COLOR_VS, COLOR_PVA, COLOR_PVM, COLOR_TI, COLOR_AV, COLOR_VF, COLOR_CW, COLOR_CCW, BG_TYPE, DOT_TYPE, \
    corr, plot_cross_corr, COLOR_HEADING, cross_corr_win, bin_2d, bin_2d_yz, corr2, nancorr2, set_pi_tick, \
    fold_by_fix_len, corr_slide_win, norm_by_percentile, plot_stim_schema, ax_imshow, ax_scatter, ax_set_ylim, \
    ax_set_xlim, ax_set_xticklabels, ax_set_xlabel, ax_plot, ax_set_ylabel1, COLOR_CUE, plot_legend, ax_set_ylabel2, \
    is_cl_stim, fold_bouts_by_fix_len

# plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.sans-serif'] = ['Arial']

PLOT_TIME = True
PLOT_SUM = True
PLOT_STIM_ON_FOLD = True

PLOT_IMA = False
PLOT_FT = False
PLOT_STIM_PRE = False
PLOT_STIM_ON = False
PLOT_STIM_POST = False

PLOT_COLOR_BAR = True

FIG_TIME_NAMES = ["dFF", "zscore", "PVA", "v", "av", "vf", "TI", "offset"]
FIG_SUM_NAMES = ["trial", "fold", "relation_cue_av", "tune_roi_heading", "tune_vs_vf_pvm",
                 "traj", "relation_cue_pva", "relation_v_pvm", "xcorr_pvm_vf", "roi_cor_matrix"] #tune_roi_heading_offset
FIG_TIME_NAMES_FT = ["av", "vf", "TI"]
FIG_SUM_NAMES_FT = ["fold", "relation_cue_av", "traj", "trial"]

NEED_SCATTER = False
NEED_FOLD_CUE_SINGLE = True
VERTICAL = True
FIG_PR_NAMES = ["time_zscore", "time_PVA", "time_av", "time_vf",
                "sum_fold", "sum_relation_cue_av", "sum_traj", "sum_trial",
                "sum_tune_roi_heading", "sum_roi_cor_matrix", "sum_relation_cue_pva", "sum_relation_v_pvm"]
FIG_PR_NAMES_FT = ["time_av", "time_vf",
                   "sum_fold", "sum_relation_cue_av", "sum_traj", "sum_trial"
                   "", "", "", ""]

STIM_ON = "2_ON"
STIM_ON_FOLD = "2_ON_FOLD"
STIM_PRE = "1_PRE"
STIM_POST = "3_POST"

# NOTE:
""" single trial structure
|-----|-PRE-|-----ON-----|-POST-|-----|
|-----|-----------FT----------- |-----|
|-----------------IMA----------- -----|
"""
class TimeSeq(object):
    def __init__(self, name, data, ts):
        self.name = name
        # data = {"fps": self.unify_fps, "beh": DataFrame, "dff": DataFrame, "zscore": DataFrame,
        #  "PVA": array, "PVM": array, "norm_PVM": array}
        self.data = data
        self.ts = ts
        self.info = {}  # mean_v
        self.calc_info()

    def __len__(self):
        return len(self.ts)

    def calc_info(self):
        self.info["mean_v"] = np.nanmean(self["v"])

    def __getitem__(self, item):
        ret = self.data.get(item)
        if ret is None:
            ret = self.data.get("beh", {}).get(item)
            if ret is None:
                ret = self.info.get(item)
        return ret

    def to_pool(self, config_d=None):
        return TimeSeqPool(self.name, config_d, [self])

class TimeSeqPool(object):
    def __init__(self, name, config_d, time_seq_l, n_cycles=0, fold_len=0):
        self.name = name
        self.config_d = config_d
        self.time_seq_l = time_seq_l
        self.n_cycles = n_cycles
        self.fold_len = fold_len
        print("TimeSeqPool:", name, len(self.time_seq_l), [len(ts) for ts in self.time_seq_l])
        self.r_dot_name = "r_dot"
        self.is_cl = is_cl_stim(self.config_d)
        if self.is_cl:
            self.r_dot_name = "r_bar"
        self.fps = time_seq_l[0]["fps"]

    def plot_info(self, ax, key, color="b"):
        # plot_info(ax, "mean_v")
        ys = [t.info[key] for t in self.time_seq_l]
        ax.plot(ys, "-.", c=color)
        xs = [t.name for t in self.time_seq_l]
        ax.set_xticklabels(xs)
        ax.set(ylabel=key)
        return {"fig": "line", "color": color, "y": ys, "ylabel": key}

    def plot_info_2d(self, ax, key, cols, vrange=None, cmap="jet"):
        # plot_info_2d(ax, "mean_v", 12) # for OL_dot
        zs = np.array([t.info[key] for t in self.time_seq_l])
        if len(zs)//cols > 0:
            print("info_2d not aligned !!!")
            zs = zs[:(len(zs)//cols*cols)]
        z = zs.reshape([-1, cols])
        if vrange is None:
            im = ax.imshow(z, cmap=cmap)
        else:
            im = ax.imshow(z, vmin=vrange[0], vmax=vrange[1], cmap=cmap)
        PLOT_COLOR_BAR and plt.colorbar(im, ax=ax, orientation="horizontal", shrink=0.5)
        ax.set_yticklabels([])
        ax.set_title("%s: %.2f" % (key, np.mean(zs)), fontsize=20)
        return {"fig": "hot", "cmap": cmap, "z": z, "zlabel": key}

    def plot_data_mean(self, ax, key, color, alpha_single=None):  # fold
        # plot_data_mean(ax, "r_dot", "k")
        # plot_data_mean(ax, "vf", COLOR_VF)
        # plot_data_mean(ax, "av", COLOR_AV)
        ts_l = [t.ts-self.time_seq_l[0].ts[0] for t in self.time_seq_l]  # start from time 0
        ys_l = [t[key] for t in self.time_seq_l]
        # cut by shortest one
        xs, ys, es = plot_lines_with_err_band(ax, ys_l, color, xs_l=ts_l, alpha_single=alpha_single)
        if len(xs):
            ax.set(xlabel="time", ylabel=key, xlim=(xs[0], xs[-1]))
        return {"fig": "line", "x": xs, "y": ys, "es": es, "color": color, "xlabel": "time", "ylabel": key}

    def plot_data_relation(self, ax, key1, key2, xrange, yrange, need_cw=False):
        # plot_data_relation(ax, "r_dot", "av", (-180, 180), (-5, 5))
        # plot_data_relation(ax, "v", "PVM", (0, 5), (0, 3))
        xl = [t[key1] for t in self.time_seq_l]
        xs = np.concatenate(xl)
        yl = [t[key2] for t in self.time_seq_l]
        ys = np.concatenate(yl)
        scatter_lines_with_err_band(ax, xs, ys, "k", xrange, alpha=1, alpha_single=0, alpha_fill=0.3)
        sub_figs = []
        if need_cw:
            r_dot_l = [t[key1] for t in self.time_seq_l]
            xl_cw, yl_cw = [], []
            xl_ccw, yl_ccw = [], []
            for x, y, r_dot in zip(xl, yl, r_dot_l):
                d_r_dot = r_dot.diff()
                xl_cw.extend(r_dot[d_r_dot > 0])
                xl_ccw.extend(r_dot[d_r_dot < 0])
                yl_cw.extend(y[d_r_dot > 0])
                yl_ccw.extend(y[d_r_dot < 0])
            xl_cw = np.array(xl_cw)
            xl_ccw = np.array(xl_ccw)
            yl_cw = np.array(yl_cw)
            yl_ccw = np.array(yl_ccw)
            scatter_lines_with_err_band(ax, xl_cw, yl_cw, COLOR_CW, xrange, alpha_single=0.1, alpha_fill=0)
            scatter_lines_with_err_band(ax, xl_ccw, yl_ccw, COLOR_CCW, xrange, alpha_single=0.1, alpha_fill=0)
            sub_figs.append({"fig": "relation", "x": xl_cw, "y": yl_cw, "color": COLOR_CW, "alpha_single": 0.1, "alpha_fill": 0})
            sub_figs.append({"fig": "relation", "x": xl_ccw, "y": yl_ccw, "color": COLOR_CCW, "alpha_single": 0.1, "alpha_fill": 0})
        ax.set(xlabel=key1, ylabel=key2, xlim=xrange, ylim=yrange)
        return {"fig": "relation", "x": xs, "y": ys, "xlabel": key1, "ylabel": key2,
                "xlim": xrange, "ylim": yrange, "color": "k", "alpha_single": 0, "alpha_fill": 0.3,
                "sub_figs": sub_figs}

    def plot_data_relation_2d(self, ax, key1, key2, key3, range_para1, range_para2):
        # self.plot_data_relation_2d(ax, "vs", "vf", "norm_PVM", (-5, 5, 2), (-4, 12, 2))
        xl = [t[key1] for t in self.time_seq_l]
        xs = np.concatenate(xl)
        yl = [t[key2] for t in self.time_seq_l]
        ys = np.concatenate(yl)
        zl = [t[key3] for t in self.time_seq_l]
        zs = np.concatenate(zl)
        w, x_bins, y_bins = bin_2d(xs, ys, zs, range_para1, range_para2)
        extent = (x_bins.min(), x_bins.max() + (x_bins[1] - x_bins[0]), y_bins.min(), y_bins.max() + (y_bins[1] - y_bins[0]))
        im = ax.imshow(w, cmap="jet", origin="lower", extent=extent)
        ax.set_xlabel(key1)
        ax.set_ylabel(key2)
        ax.set_title(key3)
        ax.set_xticks(x_bins)
        ax.set_yticks(y_bins)
        PLOT_COLOR_BAR and plt.colorbar(im, ax=ax)
        return {"fig": "hot", "xlabel": key1, "ylabel": key2, "z": w, "zlabel": key3, "origin": "lower", "extent": extent}

    def plot_data_relation_2d_yz(self, ax, key1, df_name_yz, range_para1):
        # self.plot_data_relation_2d_yz(ax, "heading", "zscore", (-np.pi, np.pi, np.pi/10))
        xl = [t[key1] for t in self.time_seq_l]
        xs = np.concatenate(xl)
        yzl = [t[df_name_yz] for t in self.time_seq_l]
        yzs = np.concatenate(yzl)
        w, x_bins, y_bins = bin_2d_yz(xs, yzs, range_para1)
        extent = (x_bins.min(), x_bins.max() + (x_bins[1] - x_bins[0]), y_bins.min(), y_bins.max() + (y_bins[1] - y_bins[0]))
        im = ax.imshow(w, cmap="jet", origin="lower", extent=extent, aspect="auto")
        ax.set_xlabel(key1)
        ax.set_ylabel("ROIs")
        ax.set_title(df_name_yz)
        ax.set_xticks(x_bins)
        ax.set_yticks(y_bins)
        PLOT_COLOR_BAR and plt.colorbar(im, ax=ax)
        # set_pi_tick(ax)
        return {"fig": "hot", "xlabel": key1, "ylabel": "ROIs", "z": w, "zlabel": df_name_yz, "origin": "lower", "extent": extent}

    def plot_data_xcor(self, ax, key1, key2, xlim, xstep, ylim):
        # plot_data_xcor(ax, "PVM", "vf", (-10, 10), 1, (-0.2, 0.5))
        xl = [t[key1] for t in self.time_seq_l]
        xs = np.concatenate(xl)
        yl = [t[key2] for t in self.time_seq_l]
        ys = np.concatenate(yl)
        ts, cs = plot_cross_corr(ax, xs, ys, xlim[0], xlim[1], xstep)
        ax.set_ylim(ylim)
        xlabel = "frame (fps:%.2f)(+:%s prior)" % (self.fps, key1)
        ax.set_xlabel(xlabel)
        ylabel = "corr(%s, %s)" % (key1, key2)
        ax.set_ylabel(ylabel)
        return {"fig": "line", "x": ts, "y": cs, "xlim": xlim, "ylim": ylim, "xlabel": xlabel, "ylabel": ylabel}

    def plot_data_roi_cor_matrix(self, ax, df_name_yz):
        # plot_data_roi_cor_matrix(ax, "dff")
        yzl = [t[df_name_yz] for t in self.time_seq_l]
        zs_df = np.concatenate(yzl)
        n = zs_df.shape[1]
        m = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                m[i][j] = corr(zs_df[:, i], zs_df[:, j])
        im = ax.imshow(m, vmin=-1, vmax=1, cmap="jet", origin="lower")
        CDI = m[0][-1] - m[0][int(n / 2)]
        ax.set_title("CDI:%.2f" % CDI, fontsize=20)
        PLOT_COLOR_BAR and plt.colorbar(im, ax=ax)
        ax.set_xlabel("ROIs")
        ax.set_ylabel("ROIs")
        return {"fig": "hot", "z": m, "zlabel": df_name_yz, "origin": "lower"}

    def plot_summary(self, sum_names, title_l, img_path, cols=5):
        if len(self.time_seq_l) == 0:
            print(self.name, "is empty")
            return
        nfigs = len(sum_names)
        rows = int(nfigs / cols)
        if rows * cols < nfigs:
            rows += 1
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 4), dpi=300)
        axes = axs.flatten()
        for i, ax in enumerate(axes):
            if i >= len(sum_names):
                f = ""
            else:
                f = sum_names[i]
            if len(f):
                func = self.__getattribute__("_plot_sum_" + f)
                s = func(axes[i])
                if s:
                    title_l.append(s)
            else:
                ax.axis("off")
                ax.set_xticks([])
                ax.set_yticks([])
                for a in ["left", "right", "top", "bottom"]:
                    ax.spines[a].set_visible(False)

        plt.suptitle(" ".join(title_l))
        plt.tight_layout()
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        save_fig(img_path + "_" + self.name)

    def _plot_sum_trial(self, ax):  #1
        if self.n_cycles:
            self.plot_info_2d(ax, "mean_v", self.n_cycles, (0, 10))  # 12 for OL_dot

    def _plot_sum_fold(self, ax):  #2
        if self.config_d.get("stim_name", "").startswith("CL_"):
            ax.axis("off")
            return
        self.plot_data_mean(ax, self.r_dot_name, COLOR_CUE, alpha_single=0.1 if NEED_FOLD_CUE_SINGLE else None)
        self.plot_data_mean(ax, "av", COLOR_AV)
        ax2 = ax.twinx()
        self.plot_data_mean(ax2, "vf", COLOR_VF)
        ax.set_ylabel("Va, cue")
        ax.set_ylim(-3, 3)
        ax2.set_ylabel("Vf", color=COLOR_VF)
        ax2.set_ylim(-1, 8)
        ax.set_title("Fold", color=COLOR_AV)

    def _plot_sum_relation_cue_av(self, ax):  #3
        self.plot_data_relation(ax, self.r_dot_name, "av", (-np.pi, np.pi), (-5, 5), True)
        set_pi_tick(ax, vertical=True)
        ax.set_ylabel("Va", color=COLOR_AV)
        ax.set_xlabel("cue", color=COLOR_CUE)
        ax.set_title("Va", color=COLOR_AV)

    def _plot_sum_tune_roi_heading(self, ax):  #4
        self.plot_data_relation_2d_yz(ax, "heading", "zscore", (-np.pi, np.pi, np.pi/10))
        set_pi_tick(ax, vertical=True)
        ax.set_title("zscore", color=COLOR_PVA)
        ax.set_xlabel("heading", color=COLOR_HEADING)

    def _plot_sum_tune_roi_heading_offset(self, ax):  #4
        self.plot_data_relation_2d_yz(ax, "heading-offset", "zscore", (-np.pi, np.pi, np.pi/10))
        set_pi_tick(ax, vertical=True)

    def _plot_sum_tune_vs_vf_pvm(self, ax):  #5
        self.plot_data_relation_2d(ax, "vs", "vf", "norm_PVM", (-5, 5, 2), (-4, 12, 2))

    def _plot_sum_traj(self, ax):  #6
        if not self.is_cl:
            ax.axis("off")
            return

        xl = np.concatenate([t["y"] for t in self.time_seq_l])
        yl = np.concatenate([t["x"] for t in self.time_seq_l])
        ax.scatter(xl, yl, marker=",", s=1, color="k")
        ax.axis("square")
        if (np.max(xl)-np.min(xl)) < 20 and (np.max(yl)-np.min(yl)) < 20:
            ax.set_xlim(np.min(xl)-1, np.min(xl)+21)
            ax.set_ylim(np.min(yl)-1, np.min(yl)+21)
        x, y = xl[0], yl[0]
        ax.plot([x-10, x+10], [y, y], "g", lw=0.5)
        ax.plot([x, x], [y-10, y+10], "g", lw=0.5)
        ax.set_title("maxD: %.2f" % max_dist(xl, yl), fontsize=20)

    def _plot_sum_relation_cue_pva(self, ax):  #7
        self.plot_data_relation(ax, self.r_dot_name, "PVA", (-np.pi, np.pi), (-4, 4), True)
        set_pi_tick(ax, vertical=True)
        ax.set_title("PVA", color=COLOR_PVA)
        ax.set_xlabel("cue", color=COLOR_CUE)

    def _plot_sum_relation_v_pvm(self, ax):  #8
        self.plot_data_relation(ax, "v", "PVM", (0, 5), (0, 3), True)
        ax.set_title("PVM", color=COLOR_PVM)
        ax.set_xlabel("V", color=COLOR_V)

    def _plot_sum_xcorr_pvm_vf(self, ax):   #9
        t = 1  # s
        step = 0.1  # s
        self.plot_data_xcor(ax, "PVM", "vf", (int(-t*self.fps), int(t*self.fps)), int(step*self.fps+0.5), (-0.2, 0.5))

    def _plot_sum_roi_cor_matrix(self, ax):  #10
        self.plot_data_roi_cor_matrix(ax, "dff")

class ImagingTrial(BaseExp):

    def load_for_filter(self, is_fob=IS_FOB):
        try:
            ft_data = load_fob_dat(self.exp_folder)
        except:
            ft_data = None

        if ft_data:
            if ft_data["config"]["duration"] < FILTER_MIN_DURATION:  # abort
                self.invalid_type = "only_IM"
            else:
                v = ft_data["stim_df"]["v"]
                pos3 = int(len(v)/3)
                if v[pos3:pos3*2].mean() < FILTER_MIN_SPEED:  # slow
                    self.invalid_type = "slow"
                pos9 = int(len(v)/9)
                for i in range(9):
                    if v[pos9*i:pos9*(i+1)].mean() > FILTER_MAX_TSPEED:  # slow
                        self.invalid_type = "fast"
                        break
            if is_fob:
                if self.invalid_type == "only_IM":
                    self.invalid_type = "invalid"
                if self.invalid_type:
                    print(self.invalid_type, self.exp_folder)
                    self.invalid_path = os.path.join(ROOT, "_" + self.invalid_type, self.exp_name.split("_")[0], self.fly_name, self.exp_name)
                return
        else:
            self.invalid_type = "only_IM"

        valid_im, valid_ts = False, False
        im_exp_xml = glob(self.exp_folder + "/*/Experiment.xml")
        if len(im_exp_xml) > 1:
            print("> 1 img folder !!!", im_exp_xml)
        valid_im = len(im_exp_xml) > 0

        if valid_im and self.invalid_type != "only_IM":
            exp_info = load_exp_xml(im_exp_xml[0])
            ts_h5 = glob(self.exp_folder + "/*/Episode001.h5")
            valid_ts = len(ts_h5) > 0
            if valid_ts:
                frame_info = load_NI_h5(ts_h5[0], exp_info)
                if frame_info["FT_frame"][0] < frame_info["rate"] * 0.05:  # FT earlier
                    valid_ts = False

        if self.invalid_type == "only_IM" and not valid_im:
            self.invalid_type = "invalid"
        else:
            if self.invalid_type != "only_IM" and not valid_ts:
                if not valid_im:
                    self.invalid_type = "only_FT"
                else:
                    self.invalid_type = "not_sync"

        if self.invalid_type:
            print(self.invalid_type, self.exp_folder)
            self.invalid_path = os.path.join(ROOT, "_" + self.invalid_type, self.exp_name.split("_")[0], self.fly_name, self.exp_name)

    def load_raw_data(self):
        # NOTE: (config), stim_df[(cnt, ts), FT_frame, r_bar, r_dot, heading, v, vs, vf, av]
        ft_data = load_fob_dat(self.exp_folder)
        # NOTE: (2p_info, rate, FT_frame), dFF_frame, dFF_df, zscore_df
        ima_data = load_imaging_data(self.exp_folder)
        if ft_data:
            if ima_data:
                correct_ft_frame(ft_data["stim_df"], ima_data["FT_frame"], ima_data["rate"])
            else:
                ft_data["FT_frame"] = ft_data["stim_df"]["cnt"]/FT_RATE*SYNC_RATE
                ft_data["stim_df"]["FT_frame"] = ft_data["FT_frame"]
        self.align_ima_data(ft_data, ima_data)
        self.data = {"ft_data": ft_data, "ima_data": ima_data}

    def align_ima_data(self, ft_data, ima_data):
        ft_frame = ft_data["stim_df"]["FT_frame"] / SYNC_RATE
        ft_start = ft_frame.iloc[0]
        ft_end = ft_frame.iloc[-1]
        ft_duration = round(ft_end - ft_start)
        ima_rate = ima_data['2p_info']['frameRate']
        ima_cnt = round(ft_duration * ima_rate)
        dff_frame = ima_data["dFF_frame"] / SYNC_RATE
        difference_array = np.absolute(dff_frame - ft_start)
        start_index = difference_array.argmin()
        end_index = start_index + ima_cnt
        ima_data['delta_F'] = ima_data['F_df'][start_index:end_index]

    def proc_raw_data(self):
        self.ft_data = self.data["ft_data"]
        self.ima_data = self.data["ima_data"]
        self.config_d, self.stim_df = self.ft_data["config"], self.ft_data["stim_df"]
        self.stim_df["heading"] = lim_dir_l(self.stim_df["heading"] - self.stim_df["heading"].iloc[0])  # set the start heading to 0
        # self.stim_df["r_dot"] = lim_dir_l(np.deg2rad(self.stim_df["r_dot"]))
        # self.stim_df["r_bar"] = lim_dir_l(np.deg2rad(self.stim_df["r_bar"]))
        self.stim_name = self.config_d.get("stim_name", "stim?")
        self.exp_prefix = "%s    %s    %s" % (self.fly_name, self.exp_name, self.stim_name)
        self.sync_rate = SYNC_RATE
        self.heading_sign = 1
        self.only_ft = False
        if self.ft_data:
            self.ft_frame = self.stim_df["FT_frame"]
            self.ft_ts = self.ft_frame / self.sync_rate
            self.stim_df["TI"] = self.calc_TI()[0]
            if not self.ima_data:
                self.fig_names = FIG_TIME_NAMES_FT
                self.sum_names = FIG_SUM_NAMES_FT
                self.pr_names = FIG_PR_NAMES_FT
                self.only_ft = True
            ft_steph = (self.ft_frame.iloc[1] - self.ft_frame.iloc[0]) / 2
            self.end_t = (self.ft_frame.iloc[-1] + ft_steph) / self.sync_rate
            self.start_t = (self.ft_frame.iloc[0] - ft_steph) / self.sync_rate
        if self.ima_data:
            self.sync_rate = self.ima_data["rate"]
            self.dff_frame = self.ima_data["dFF_frame"]
            self.dff_ts = self.dff_frame / self.sync_rate
            roi_n = self.ima_data["dFF_df"].shape[1]
            wed_n = 8 if roi_n == 10 else 16
            self.ima_data["dFF_df"] = self.ima_data["dFF_df"][:, -wed_n:]
            self.ima_data["zscore_df"] = self.ima_data["zscore_df"][:, -wed_n:]
            # # NOTE: zscore_PVA, zscore_PVM
            # pva_dff, pvm_dff = calc_pva(self.ima_data["dFF_df"])
            # pva_zs, pvm_zs = calc_pva(self.ima_data["zscore_df"])
            # self.ima_data["dFF_PVA"], self.ima_data["dFF_PVM"], self.ima_data["norm_PVM"] = pva_dff, pvm_dff, norm_by_percentile(pvm_dff)
            # self.ima_data["zscore_PVA"], self.ima_data["zscore_PVM"], self.ima_data["norm_zsPVM"] = pva_zs, pvm_zs, norm_by_percentile(pvm_zs)
            if not self.data["exp_type"].endswith("EPG"):
                self.heading_sign = -1

            dFF_steph = (self.dff_frame[1] - self.dff_frame[0]) / 2
            self.end_t = (self.dff_frame[-1] + dFF_steph) / self.sync_rate
            self.start_t = (self.dff_frame[0] - dFF_steph) / self.sync_rate

            self.fig_names = FIG_TIME_NAMES
            self.sum_names = FIG_SUM_NAMES
            self.pr_names = FIG_PR_NAMES
        self.title_l = self.get_suptitle()
        self.is_cl = is_cl_stim(self.config_d)

    def calc_TI(self):
        av = np.array(self.stim_df["av"])
        r_dot = np.array(self.stim_df["r_dot"])
        fid = corr_slide_win(av, r_dot, 180)  # 3.6 s
        vig = (np.sum(av[r_dot > 0]) - np.sum(av[r_dot < 0]))/np.count_nonzero(r_dot != 0)
        TI = fid * np.abs(vig)
        return TI, fid, vig

    def plot_all(self):
        self.split_time_seq()
        os.makedirs(os.path.dirname(self.img_path), exist_ok=True)
        PLOT_TIME and self.plot_time()
        PLOT_SUM and self.plot_summary()

    def plot_pr(self):
        self.split_time_seq()
        os.makedirs(os.path.dirname(self.img_path), exist_ok=True)
        global VERTICAL
        VERTICAL = False

        title = self.title_l.copy()
        title.append("\n")

        time_n = np.count_nonzero([f.startswith("time") for f in self.pr_names])
        sum_n = len(self.pr_names) - time_n
        rows, cols = time_n + 1, sum_n
        fig = plt.figure(figsize=(18, rows*2), dpi=200)
        plt.subplots_adjust(0.05, 0.05, 0.96, 0.85, wspace=0.3, hspace=0.3)
        axes = []
        row_i, col_i = 0, 0
        for f in self.pr_names:
            if f.startswith("time"):
                ax = fig.add_subplot(rows, 1, row_i+1)
                row_i += 1
            elif f.startswith("sum"):
                ax = fig.add_subplot(rows, cols, row_i*cols+col_i+1)
                col_i += 1
            axes.append(ax)
        for i, f in enumerate(self.pr_names):
            if len(f) == 0:
                continue
            if f.startswith("time"):
                func = self.__getattribute__("_plot_" + f)
            else:
                if not self.timep_cycles:
                    continue
                func = self.timep_cycles.__getattribute__("_plot_" + f)
            s = func(axes[i])
            if s:
                title.append(s)
        plt.suptitle(" ".join(title), fontsize=16)

        plot_stim_schema(plt.axes([0.02, .86, .1, .1]), self.config_d)
        plot_legend(plt.axes([0.88, 0.86, 0.1, .1]), self.heading_sign, self.only_ft)
        # plt.tight_layout()
        # plt.show()
        save_fig(self.img_path + "_pr")

    def split_time_seq(self):
        if not self.ima_data:
            ima_fps = FT_RATE
        else:
            ima_fps = self.sync_rate*len(self.dff_frame)/(self.dff_frame[-1] - self.dff_frame[0])
        self.unify_fps = UNIFY_FPS or ima_fps
        print("ima_fps:", ima_fps)
        step = self.sync_rate/self.unify_fps

        ft_down_d = {}
        ft_k1 = ["r_bar", "r_dot", "heading"]
        ft_k2 = ["v", "vs", "vf", "av", "x", "y"]
        if self.ima_data:
            total_time = self.dff_frame[-1] - self.dff_frame[0]
            unify_frame = np.linspace(self.dff_frame[0], self.dff_frame[-1], int(total_time/step))
            for key in ft_k1:
                ft_down_d[key] = lim_dir_l(down_sample2(self.stim_df[key], self.ft_frame, unify_frame, circ=True))
            for key in ft_k2:
                ft_down_d[key] = down_sample2(self.stim_df[key], self.ft_frame, unify_frame, circ=False)
        else:
            # NOTE: ft_frame is from cnt
            #total_time = self.ft_frame.iloc[-1] - self.ft_frame.iloc[0]
            unify_frame = np.linspace(self.ft_frame.iloc[0], self.ft_frame.iloc[-1], len(self.ft_frame))#int(total_time/step)
            for key in ft_k1:
                ft_down_d[key] = lim_dir_l(self.stim_df[key])
            for key in ft_k2:
                ft_down_d[key] = self.stim_df[key]

        ft_down_df = pd.DataFrame(ft_down_d)
        if ima_fps > self.unify_fps:
            print("unify_fps(%.2f) < ima_fps(%.2f) !!!" % (self.unify_fps, ima_fps))

        if self.ima_data:
            sample_func = up_sample2 if ima_fps <= self.unify_fps else down_sample2
            sample_func_2d = up_sample2d if ima_fps <= self.unify_fps else down_sample2d

            dff_df_u = sample_func_2d(self.ima_data["dFF_df"], self.dff_frame, unify_frame)
            zscore_df_u = sample_func_2d(self.ima_data["zscore_df"], self.dff_frame, unify_frame)
            ima_pva = np.array(sample_func(self.ima_data["zscore_PVA"], self.dff_frame, unify_frame))
            ima_pvm = np.array(sample_func(self.ima_data["dFF_PVM"], self.dff_frame, unify_frame))  #zscore_PVM
            norm_pvm = np.array(sample_func(self.ima_data["norm_zsPVM"], self.dff_frame, unify_frame))  #norm_zsPVM

            offset = np.nanmean(diff_angle(ft_down_df["heading"], ima_pva))
            ft_down_df["heading-offset"] = ft_down_df["heading"] - offset

            unify_ts = unify_frame / self.sync_rate
            data = {"fps": self.unify_fps, "beh": ft_down_df, "dff": dff_df_u, "zscore": zscore_df_u,
                    "PVA": ima_pva, "PVM": ima_pvm, "norm_PVM": norm_pvm}
            self.times_ima = TimeSeq("IMA", data, unify_ts)

            def subset_u(idx1):
                return {"fps": self.unify_fps, "beh": ft_down_df.iloc[idx1], "dff": dff_df_u[idx1], "zscore": zscore_df_u[idx1],
                    "PVA": ima_pva[idx1], "PVM": ima_pvm[idx1], "norm_PVM": norm_pvm[idx1]}
        else:
            unify_ts = unify_frame / self.sync_rate
            self.times_ima = None

            def subset_u(idx1):
                return {"fps": self.unify_fps, "beh": ft_down_df.iloc[idx1]}

        idx_ft = (unify_ts > self.ft_ts.iloc[0]) & (unify_ts < self.ft_ts.iloc[-1])
        # idx1 ~= ~np.isnan(ft_down_df["heading"])
        self.times_ft = TimeSeq("FT", subset_u(idx_ft), unify_ts[idx_ft])

        # NOTE: for >1 trial
        not_wait_bout = calc_bout(self.stim_df["is_wait"].to_numpy() == 0)
        ts_stim_bout = [(self.ft_ts[b[0]], self.ft_ts[b[1]]) for b in not_wait_bout]
        if len(ts_stim_bout) == 0:
            return False
        ts_stim_on, ts_stim_off = ts_stim_bout[0][0], ts_stim_bout[-1][-1]

        # NOTE: for only one trial
        # ts_stim_on_s = self.ft_ts[self.stim_df["is_wait"] == 0]
        # ts_stim_on1, ts_stim_off1 = ts_stim_on_s.iloc[0], ts_stim_on_s.iloc[-1]

        idx_on = (unify_ts > ts_stim_on) & (unify_ts < ts_stim_off)
        self.times_stim_on = TimeSeq(STIM_ON, subset_u(idx_on), unify_ts[idx_on])

        idx_pre = unify_ts < ts_stim_on
        self.times_stim_pre = TimeSeq(STIM_PRE, subset_u(idx_pre), unify_ts[idx_pre])

        idx_post = unify_ts > ts_stim_off
        self.times_stim_post = TimeSeq(STIM_POST, subset_u(idx_post), unify_ts[idx_post])

        fold_cycles = 1
        dot_speed = self.config_d["dot_speed"]  # *2/3 #NOTE: for old data
        dot_start = self.config_d["dot_start"]
        dot_end = self.config_d["dot_end"]
        if dot_speed == 0:
            self.timep_cycles = None
            return False
        else:
            fold_duration = (dot_end-dot_start) / dot_speed * 2 * fold_cycles
        fold_len = int(fold_duration * self.unify_fps)

        # fold_idx_l = fold_by_fix_len(np.nonzero(idx_on)[0], fold_len)
        idx_on_idx_l = []
        for bout in ts_stim_bout:
            idx_on1 = (unify_ts > bout[0]) & (unify_ts < bout[1])
            idx_on_idx_l.append(np.nonzero(idx_on1)[0])
        fold_idx_l = fold_bouts_by_fix_len(idx_on_idx_l, fold_len)

        self.timep_cycles = TimeSeqPool(STIM_ON_FOLD, self.config_d, [
            TimeSeq("STIM_ON_FOLD#%d"%i, subset_u(fold_idx), unify_ts[fold_idx])
            for i, fold_idx in enumerate(fold_idx_l)
        ], n_cycles=len(fold_idx_l), fold_len=fold_len)
        return True

    def get_suptitle(self):
        twop_info = self.ima_data.get("2p_info", {})
        return [self.exp_prefix, "\n",
                "scr_width/dot_width/speed/y:%d/%d/%d/%d" % (self.config_d["scr_width_deg"], self.config_d["dot_width"], self.config_d["dot_speed"], self.config_d.get("dot_y", 0)), "    ",
                "dot/bg_color:[%s]/[%s]" % (" ".join(self.config_d["dot_color"].split()), " ".join(self.config_d["bg_color"].split())), "    ",
                "PMT/pockel:%.2f/%.2f" % (twop_info.get("gainA", 0), twop_info.get("pockels", 0))]

    def plot_summary(self):
        PLOT_IMA and self.times_ima.to_pool(self.config_d).plot_summary(self.sum_names, self.title_l, self.img_path)
        PLOT_FT and self.times_ft.to_pool(self.config_d).plot_summary(self.sum_names, self.title_l, self.img_path)
        PLOT_STIM_PRE and self.times_stim_pre.to_pool(self.config_d).plot_summary(self.sum_names, self.title_l, self.img_path)
        PLOT_STIM_ON and self.times_stim_on.to_pool(self.config_d).plot_summary(self.sum_names, self.title_l, self.img_path)
        PLOT_STIM_POST and self.times_stim_post.to_pool(self.config_d).plot_summary(self.sum_names, self.title_l, self.img_path)
        PLOT_STIM_ON_FOLD and self.timep_cycles.plot_summary(self.sum_names, self.title_l, self.img_path)

    def export_video(self, need_v=True, need_ima=True):
        bar_deg, dot_deg = 0, 0
        if self.config_d["bg_type"] > 0:
            bar_deg = self.config_d["dot_width"]
        if self.config_d["dot_type"] > 0:
            dot_deg = self.config_d["dot_width"]

        stim_avi = "temp/%s_stim.avi"%self.exp_name
        ft_avi = "temp/%s_ft.avi"%self.exp_name
        traj_avi = "temp/%s_traj.avi"%self.exp_name
        ima_avi = "temp/%s_ima.avi"%self.exp_name
        stim_ft_avi = "img/%s_comb.mp4" % self.exp_name #self.exp_folder + "/stim+ft.avi"

        write_stim_video(stim_avi, self.stim_df["r_bar"], self.stim_df["r_dot"], bar_deg, dot_deg,
                         is_wait=self.stim_df["is_wait"], ts=self.ft_ts) #delay_frames=self.stim_df["cnt"].iloc[0]
        plt.figure(figsize=(4, 8))
        plt.scatter(-self.stim_df["r_bar"], self.ft_ts, s=1, color="r")
        plt.ylim(self.ft_ts.max(), self.ft_ts.min())
        ft_video = self.exp_folder + "/fictrac-debug.avi"
        ## write_ft_vel_video(ft_avi, ft_video, 0, 320, 0, 480, df=self.stim_df, need_v=need_v) #crop_video(ft_avi, ft_video, 0, 320, 0, 480)
        write_ft_vel_video(ft_avi, ft_video, 2, 318, 20, 270, df=self.stim_df, need_v=need_v, traj_video=traj_avi)

        if need_ima:
            tif_file = glob(os.path.join(self.exp_folder, "*", "*.tif_avg.tif"))[0]
            roi_file = glob(os.path.join(self.exp_folder, "*", "roi.npy"))[0]

            pva = smooth_angle(self.ima_data["zscore_PVA"], 5)
            _, up_idx = up_sample2(pva, self.dff_ts, self.ft_ts, return_idx=True)
            pva = pva[up_idx]
            roi_n = self.ima_data["dFF_df"].shape[1]
            write_ima_video(ima_avi, tif_file, roi_file, pva, up_idx, roi_n, ts=self.ft_ts, scale=300/128, r_bar=self.stim_df["r_bar"])
            plt.scatter(pva, self.ft_ts, s=1, color="b")

        stack_video(stim_ft_avi, stack_video(None, stim_avi, ft_avi, True), stack_video(None, ima_avi, traj_avi, True))
        # plt.tight_layout()
        # plt.show()
        self.plot_time_hot()

    def plot_time_hot(self, name="zscore"):
        # plt.figure(figsize=(10, 2), dpi=300)
        plt.figure(figsize=(10, 2))
        ax = plt.gca()
        dFF_im = self.ima_data[name + "_df"]

        roi_n = dFF_im.shape[1]
        im = ax.imshow(dFF_im.T[::-1, :], cmap="Blues", aspect="auto", extent=[self.start_t, self.end_t, 0, roi_n])
        heading = self.heading_sign * lim_dir_l(self.stim_df["heading"])
        plot_angle(ax, scale_x(heading, 0, roi_n), COLOR_HEADING, xs=self.ft_ts, vertical=False, alpha=0.5)
        # plot_angle(ax, scale_x(self.stim_df["r_bar"], 0, roi_n), COLOR_HEADING, xs=self.ft_ts, vertical=False, alpha=0.2)

        ax.set_title(name)
        ax.set_xlim(self.start_t, self.end_t)
        yt = np.arange(self.start_t, np.floor(self.end_t), 10).astype(int)
        # ax.set_xticks(yt)
        # ax.set_xticklabels(yt - int(start_t))
        # ax.set_yticks(scale_x([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], 0, 16))
        # ax.set_yticklabels(["$-\pi$", "$-\pi/2$", 0, "$\pi/2$", "$\pi$"])
        plt.tight_layout()
        plt.savefig("img/%s_time_hot.png" % self.exp_name)#pdf
        # plt.show()

    def plot_time(self):
        nfigs = len(self.fig_names)
        title = self.title_l.copy()
        title.append("\n")
        fig, axes = plt.subplots(1, nfigs, figsize=(2*nfigs, 14), sharey=True, dpi=200)
        for i, f in enumerate(self.fig_names):
            func = self.__getattribute__("_plot_time_" + f)
            s = func(axes[i])
            if s:
                title.append(s)
        plt.suptitle(" ".join(title), fontsize=16)
        plt.tight_layout()
        plot_stim_schema(plt.axes([0.02, .8, .1, .1]), self.config_d)
        # plt.show()
        save_fig(self.img_path + "_time")

    def plot_midline_and_cue(self, ax, rmin, rmax, ts, lw=0.8):
        # plot_angle(ax, np.deg2rad(np.array(self.stim_df["r_dot"]))/5, "k", lw=0.4)
        if self.config_d.get("dot_type", 1) > 0:
            # ax_plot(ax, ts, scale_x(self.stim_df["r_dot"], rmin, rmax), color=COLOR_CUE, lw=lw, vertical=VERTICAL)
            r_bar = lim_dir_l(np.array(self.stim_df["r_dot"]))
            plot_angle(ax, r_bar, COLOR_CUE, xs=ts, lw=lw, vertical=VERTICAL)
        if self.is_cl or self.config_d.get("bg_type", 0) > 0:
            r_bar = lim_dir_l(np.array(self.stim_df["r_bar"]))
            plot_angle(ax, r_bar, COLOR_CUE, xs=ts, lw=lw, vertical=VERTICAL)

    def plot_0_line(self, ax):
        ax.axvline(0, color="k", lw=0.1) if VERTICAL else ax.axhline(0, color="k", lw=0.2)

    def plot_stim_on_off(self, ax, lw=0.8):
        ts_stim_on = self.ft_ts[self.stim_df["is_wait"] == 0]
        for t in [ts_stim_on.min(), ts_stim_on.max()]:
            ax.axhline(t, color="k", linestyle="--", lw=lw) if VERTICAL else ax.axvline(t, color="k", linestyle="--", lw=lw)

    def inset_hist(self, ax, xs, h=1, bins=32, color="r", alpha=0.5, vertical=True, remove_0=True):
        xs = xs[~np.isnan(xs)]
        if remove_0:
            xs = xs[xs != 0]  # NOTE: r_bar=0 when wait
        ys, xs = np.histogram(xs, bins=bins, range=(-np.pi, np.pi))
        norm_max = np.sum(ys) #np.max(ys)
        if vertical:
            ax.fill_between((xs[:-1] + np.pi) / 2 / np.pi + 0.5/bins, np.zeros(len(ys)), ys / norm_max * h,
                            color=color, alpha=alpha, transform=ax.transAxes)
        else:
            ax.fill_betweenx((xs[:-1] + np.pi) / 2 / np.pi + 0.5/bins, np.ones(len(ys)), 1 - (ys / norm_max * h),
                            color=color, alpha=alpha, transform=ax.transAxes)

    def inset_line(self, ax, xs, m, sd, h=0.05, color="r", alpha=0.6, xrange=None):
        y_sc = h
        xs = scale_x(xs, 0, 1)
        ax.plot(xs, m*y_sc+h, color=color, alpha=alpha, lw=0.6, transform=ax.transAxes)
        if sd is not None:
            ax.fill_between(xs, (m-sd)*y_sc, (m+sd)*y_sc, color=color, alpha=alpha/2, transform=ax.transAxes)
        ax.plot(xs, np.ones(len(xs))*y_sc, "k", lw=0.2, transform=ax.transAxes)
        ax.plot(xs, np.ones(len(xs))*y_sc*2, "k", lw=0.2, transform=ax.transAxes)
        if xrange is not None:
            ax.text(0, y_sc*2, str(xrange[0]), horizontalalignment='left',
                    verticalalignment='top', color="b", fontsize=12, transform=ax.transAxes)
            ax.text(1, y_sc*2, str(xrange[1]), horizontalalignment='right',
                    verticalalignment='top', color="b", fontsize=12, transform=ax.transAxes)

    def _plot_time_dFF(self, ax):
        return self._plot_time_zscore(ax, name="dFF")

    def _plot_time_zscore(self, ax, name="zscore"):
        dFF_im = self.ima_data[name + "_df"]

        roi_n = dFF_im.shape[1]
        im = ax_imshow(ax, dFF_im, cmap="Blues", aspect="auto", extent=[self.start_t, self.end_t, 0, roi_n], vertical=VERTICAL)
        if name == "dFF":
            ax_scatter(ax, self.dff_ts, (self.ima_data["dFF_PVA"]+np.pi)*roi_n/2/np.pi, s=5, color="b", vertical=VERTICAL)
        if name == "zscore":
            # plt.figure()
            # plt.imshow(self.ima_data["zscore_df"][-30:], cmap="Blues")
            # plt.scatter((self.ima_data["zscore_PVA"][-30:] + np.pi) * roi_n / 2 / np.pi, range(30), color="r", s=4)
            # plt.show()

            # ax.scatter((self.ima_data["zscore_PVA"]+np.pi)*roi_n/2/np.pi, self.dff_ts, s=5, color="b")
            heading = self.heading_sign * lim_dir_l(self.stim_df["heading"])
            plot_angle(ax, scale_x(heading, 0, roi_n), COLOR_HEADING, xs=self.ft_ts, alpha=0.8, vertical=VERTICAL)
            ax_set_ylabel2(ax, "heading", color=COLOR_HEADING, fontsize=20, vertical=VERTICAL)

            # phi = np.arctan2(rolling_mean(self.stim_df["vs"], 25), rolling_mean(self.stim_df["vf"], 25))
            # travel_dir = self.heading_sign * lim_dir_l(self.stim_df["heading"] + phi)
            # travel_dir[rolling_mean(self.stim_df["v"], 25) < 2] = np.nan
            # plot_angle(ax, scale_x(travel_dir, 0, roi_n), "y", xs=self.ft_ts, vertical=True, alpha=0.5)

        ax_set_ylabel1(ax, name, color=COLOR_PVA, fontsize=20, vertical=VERTICAL)
        ax_set_xlim(ax, self.start_t, self.end_t, vertical=VERTICAL)
        yt = np.arange(np.ceil(self.start_t), np.floor(self.end_t)).astype(int)
        ax_set_xticklabels(ax, yt, vertical=VERTICAL)
        # plt.colorbar(im, ax=ax)
        return name + "(%.2f,%.2f)" % (np.min(dFF_im), np.max(dFF_im))

    def _plot_time_PVA(self, ax):
        ax_set_ylabel1(ax, "PVA", color=COLOR_PVA, fontsize=20, vertical=VERTICAL)
        pva = self.ima_data["zscore_PVA"]
        if NEED_SCATTER:
            ax_scatter(ax, self.dff_ts, pva, s=5, color=COLOR_PVA, vertical=VERTICAL)
        plot_angle(ax, smooth_angle(pva, 5), COLOR_PVA, xs=self.dff_ts, vertical=VERTICAL, lw=0.8)

        heading = self.heading_sign * lim_dir_l(self.stim_df["heading"])
        plot_angle(ax, heading, COLOR_HEADING, xs=self.ft_ts, vertical=VERTICAL, lw=0.8)
        self.inset_hist(ax, heading, color=COLOR_HEADING, vertical=VERTICAL)
        self.inset_hist(ax, pva, color=COLOR_PVA, vertical=VERTICAL)
        self.plot_0_line(ax)
        self.plot_stim_on_off(ax)
        set_pi_tick(ax, vertical=VERTICAL)
        ax_set_xlim(ax, self.start_t, self.end_t, vertical=VERTICAL)
        xl = ("" if self.heading_sign > 0 else "-") + "heading"
        ax_set_ylabel2(ax, xl, color=COLOR_HEADING, fontsize=20, vertical=VERTICAL)

    def _plot_time_v(self, ax):
        ax.set_title("v", color=COLOR_V)
        v = np.array(self.stim_df["v"])
        ax.scatter(v, self.ft_ts, c=COLOR_V, marker=",", s=1, alpha=0.5, linewidths=0)
        ax.plot(rolling_mean(v, 25), self.ft_ts, c=COLOR_V, lw=0.4)

        pvm = rolling_mean(self.ima_data["dFF_PVM"], 5)*10
        ax.plot(pvm, self.dff_ts, c=COLOR_PVM, lw=0.4, alpha=0.5)
        # pvm = rolling_mean(self.ima_data["zscore_PVM"], 5)*5
        # ax.plot(pvm, self.dff_ts, c=COLOR_PVM, lw=0.4, alpha=0.8)

        zscore_m = self.ima_data["zscore_df"].mean(axis=1) #[:, 0]#
        # dff_bg = zscore_m / np.max(np.abs(zscore_m)) * 10 + 10
        # ax.scatter(dff_bg, self.dff_ts, s=1, c=COLOR_PVM, alpha=0.5, lw=0)
        # ax.plot(rolling_mean(dff_bg, 5), self.dff_ts, c=COLOR_PVM, lw=0.4)

        # ft_bg = self.stim_df["FT_MZS"] * 10 + 10
        # ax.plot(ft_bg, self.ft_ts, c=COLOR_PVM, lw=0.4)
        ax.set_xlabel("PVM, mean_zs", color=COLOR_PVM)
        down_v = down_sample2(v, self.ft_frame, self.dff_frame, circ=False)

        ax.text(0.3, 0.03, "dFF_PVM", color="k", alpha=0.5, fontsize=12, transform=ax.transAxes)
        ax.text(0.3, 0.06, "zscore_PVM", color="k", alpha=0.8, fontsize=12, transform=ax.transAxes)
        return "v~PVM(%.2f,%.2f),v~MZS(%.2f,%.2f)" % (*nancorr2(down_v, zscore_m), *nancorr2(down_v, self.ima_data["dFF_PVM"]))

    def _plot_time_av(self, ax):
        if self.is_cl:
            ax.axis("off")
            return
        ax_set_ylabel1(ax, "Va", color=COLOR_AV, fontsize=20, vertical=VERTICAL)
        av = np.array(self.stim_df["av"])
        if NEED_SCATTER:
            ax_scatter(ax, self.ft_ts[av > 0], av[av > 0], c="r", marker=",", s=2, alpha=0.5, linewidths=0, vertical=VERTICAL)
            ax_scatter(ax, self.ft_ts[av < 0], av[av < 0], c="b", marker=",", s=2, alpha=0.5, linewidths=0, vertical=VERTICAL)
        ax_plot(ax, self.ft_ts, rolling_mean(av, 25), c=COLOR_AV, lw=0.8, vertical=VERTICAL)

        self.plot_midline_and_cue(ax, -np.pi, np.pi, self.ft_ts)
        self.plot_stim_on_off(ax)
        if not self.ima_data:
            ax_set_xlim(ax, self.ft_ts.iloc[-1], self.ft_ts.iloc[0], vertical=VERTICAL)
            # heading = self.heading_sign * lim_dir_l(self.stim_df["heading"])
            # plot_angle(ax, heading, COLOR_HEADING, xs=self.ft_ts, vertical=VERTICAL, alpha=0.2)

        self.inset_hist(ax, av, color=COLOR_AV, vertical=VERTICAL)
        r_bar = np.array(self.stim_df["r_bar"])
        self.inset_hist(ax, r_bar, color=COLOR_CUE, vertical=VERTICAL)
        set_pi_tick(ax, vertical=VERTICAL)
        ax_set_xlim(ax, self.start_t, self.end_t, vertical=VERTICAL)
        ax_set_ylabel2(ax, "cue", color=COLOR_CUE, fontsize=20, vertical=VERTICAL)

    def _plot_time_vf(self, ax):
        ax_set_ylabel1(ax, "Vf", color=COLOR_VF, fontsize=20, vertical=VERTICAL)
        vf = np.array(self.stim_df["vf"])
        vs = np.array(self.stim_df["vs"])
        if NEED_SCATTER:
            ax_scatter(ax, self.ft_ts, vf, c=COLOR_VF, marker=",", s=1, alpha=0.5, linewidths=0, vertical=VERTICAL)
        rmin, rmax = -5, 15
        ax_plot(ax, self.ft_ts, rolling_mean(vf, 25), c=COLOR_VF, lw=0.8, vertical=VERTICAL)
        if VERTICAL:
            ax_plot(ax, self.ft_ts, rolling_mean(vs, 25), c=COLOR_VS, lw=0.8, vertical=VERTICAL)
            ax_set_ylabel1(ax, "Vs", color=COLOR_VS, vertical=VERTICAL)
        else:
            ax2 = ax.twinx()
            ax_plot(ax2, self.ft_ts, rolling_mean(vs, 25), c=COLOR_VS, lw=0.8, vertical=VERTICAL)
            ax2.set_ylim(-5, 5)
            ax2.set_ylabel("Vs", color=COLOR_VS, fontsize=20)
            ax2.axhline(0, color=COLOR_VS, lw=0.1)
        self.inset_hist(ax, vf, color=COLOR_VF, vertical=VERTICAL)
        self.inset_hist(ax, vs, color=COLOR_VS, vertical=VERTICAL)
        # ax.plot(np.array(self.stim_df["dot_width"]), ts, "gray", lw=0.4)
        # self.plot_midline_and_cue(ax, rmin, rmax, self.ft_ts)
        ax_set_ylim(ax, rmin, rmax, vertical=VERTICAL)
        self.plot_0_line(ax)
        self.plot_stim_on_off(ax)
        ax_set_xlim(ax, self.start_t, self.end_t, vertical=VERTICAL)
        return "V/Vf/Vs/Va:[%.2f %.2f %.2f %.2f]" % (self.stim_df["v"].mean(), self.stim_df["vf"].mean(), self.stim_df["vs"].mean(), self.stim_df["av"].mean())

    def _plot_time_TI(self, ax):
        ax.set_title("TI, (U)nwrap", color=COLOR_TI)
        ax.plot(self.stim_df["TI"], self.ft_ts, COLOR_TI, lw=0.4)
        rmin, rmax = -1, 1
        self.plot_midline_and_cue(ax, rmin, rmax, self.ft_ts, lw=0.1)
        # ax.set_xlim(rmin, rmax)
        if self.ima_data:
            zs_upva = unwrap_dir(smooth_angle(self.ima_data["zscore_PVA"], 5))
            ax.plot(zs_upva, self.dff_ts, c=COLOR_PVA, lw=0.4)

            heading = smooth_angle(self.heading_sign * self.stim_df["heading"], 25)
            uheading = unwrap_dir(heading)
            down_uheading = unwrap_dir(down_sample2(heading, self.ft_frame, self.dff_frame, circ=True))

            # ax.plot(down_uheading, self.dff_ts, c=COLOR_HEADING, lw=1)

            ax.plot(uheading, self.ft_ts, c=COLOR_HEADING, lw=0.4)
            idx = ~np.isnan(down_uheading)
            down_uheading = np.array(down_uheading)[idx]
            zs_upva = np.array(zs_upva)[idx]
            cor_pva_heading = corr2(zs_upva, down_uheading)
            # cor_pva_heading = corr2(self.stim_df["FT_UPVA"], uheading)

            dff_fps = len(self.dff_ts)/(self.dff_ts[-1]-self.dff_ts[0])
            fs = np.arange(-int(dff_fps*3), int(dff_fps*3), 1)
            for i, win_t in enumerate([2, 8, 16, 32]):
                # xcor_win = cross_corr_win(ft_upva, uheading, fs, win_t*FT_RATE)
                xcor_win = cross_corr_win(zs_upva, down_uheading, fs, int(win_t*dff_fps))
                # xcor_win_m = np.array(cross_corr(ft_upva, uheading, fs))
                # xcor_win_m = np.array(cross_corr(zs_upva, down_uheading, fs))
                xcor_win_m = np.nanmean(xcor_win, axis=1)
                xcor_win_sd = None#np.nanstd(xcor_win, axis=1)
                self.inset_line(ax, fs/FT_RATE, xcor_win_m, xcor_win_sd, color="b", alpha=1-0.3*i, xrange=[-3, 3])
            ax.set_xlabel("cor(UPVA,Uheading)", color=COLOR_PVA)
            ax.text(0.1, 0.02, "win=[2,8,16,32]", color="b", fontsize=12, transform=ax.transAxes)
        else:
            cor_pva_heading = 0, 0
            ax.set_xlabel(("" if self.heading_sign > 0 else "-") + "UHeading", color=COLOR_HEADING)
        return "TI:%.2f heading~PVA(%.2f,%.2f)" % (np.nanmean(self.stim_df["TI"]), *cor_pva_heading)

    def _plot_time_offset(self, ax):
        ax.set_title("offset (%sheading-PVA)" % ("" if self.heading_sign > 0 else "-"), color="k")
        ret = ""
        if self.ima_data is not None:
            pva = self.ima_data["zscore_PVA"]
            heading = lim_dir_l(self.heading_sign * self.stim_df["heading"])
            down_heading = lim_dir_l(down_sample2(heading, self.ft_frame, self.dff_frame, circ=True))
            offset = diff_angle(down_heading, pva)
            plot_angle(ax, offset, "k", xs=self.dff_ts, vertical=True)

            # phi = np.arctan2(rolling_mean(self.stim_df["vs"], 25), rolling_mean(self.stim_df["vf"], 25))
            # travel_dir = self.heading_sign * lim_dir_l(down_sample2(lim_dir_l(self.stim_df["heading"] + phi), self.ft_frame, self.dff_frame, circ=True))
            # offset2 = diff_angle(travel_dir, pva)
            # down_v = down_sample2(self.stim_df["v"], self.ft_frame, self.dff_frame, circ=False)
            # offset2[down_v < 2] = np.nan
            # plot_angle(ax, offset2, "y", xs=self.dff_ts, vertical=True)

            # plot_angle(ax, pva, COLOR_PVA, xs=self.dff_ts, vertical=True)
            # plot_angle(ax, down_heading, COLOR_HEADING, xs=self.dff_ts, vertical=True)
            ret = "std(offset):%.2f" % circstd(offset[~np.isnan(offset)]+np.pi)
            self.inset_hist(ax, offset, color="k")
        set_pi_tick(ax)
        return ret




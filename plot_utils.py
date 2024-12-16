
import os
import numpy as np
import pandas as pd
from scipy import signal
from misc_utils import load_dict, save_dict
import matplotlib.pyplot as plt
from scipy.stats import circmean, pearsonr

SHOW_FIG = False
FIG_EXT = ".png"

COLOR_V = "g"
COLOR_VF = "y"
COLOR_VS = "m"
COLOR_AV = "darkcyan"
COLOR_ACC = "lightgreen"
COLOR_TI = "k"
COLOR_PVA = "b"
COLOR_PVM = "k"
COLOR_HEADING = "r"
COLOR_CUE = "hotpink"
COLOR_CW = "yellow"#[.75, .75, .75]
COLOR_CCW = "magenta"#[.5, .5, .5]

CUE_TYPE = {0: "", 1: "dot", 2: "bar", 3: "pattern", 4: "movie"}
# old
BG_TYPE = {0: "", 1: "bar", 3: "dot", 6: "star", 11: "pattern"}
DOT_TYPE = {0: "none", 1: "dot", 2: "bar", 11: "pattern"}

def plot_angle(ax, al, c, pi=np.pi, xs=None, lw=1, vertical=False, alpha=1):
    # al = unwrap_dir(al)  #[lim_dir(a-2) for a in al]
    if xs is None:
        xs = range(len(al))
    last = 0
    start = 0
    for i, a in enumerate(al):
        if abs(a - last) > pi*1.5:
            ax_plot(ax, xs[start:i], al[start:i], vertical=vertical, c=c, lw=lw, alpha=alpha)
            ax_scatter(ax, xs[i], al[i], c=c, s=0.1)
            start = i
        last = a
    ax_plot(ax, xs[start:len(al)], al[start:], vertical=vertical, c=c, lw=lw, alpha=alpha)

def ax_plot(ax, x, y, vertical=False, **kwargs):
    if vertical:
        ax.plot(y, x, **kwargs)
    else:
        ax.plot(x, y, **kwargs)

def ax_scatter(ax, x, y, vertical=False, **kwargs):
    if vertical:
        ax.scatter(y, x, **kwargs)
    else:
        ax.scatter(x, y, **kwargs)

def ax_set_xlim(ax, start, end, vertical=False):
    if vertical:
        ax.set_ylim(end, start)
    else:
        ax.set_xlim(start, end)

def ax_set_ylim(ax, start, end, vertical=False):
    if vertical:
        ax.set_xlim(start, end)
    else:
        ax.set_ylim(start, end)

def ax_set_xticklabels(ax, xt, vertical=False):
    if vertical:
        ax.set_yticks(xt)
        ax.set_yticklabels(xt)
    else:
        ax.set_xticks(xt)
        ax.set_xticklabels(xt)

def ax_set_xlabel(ax, xl, vertical=False, **kwargs):
    if vertical:
        ax.set_ylabel(xl, **kwargs)
    else:
        ax.set_xlabel(xl, **kwargs)

def ax_imshow(ax, im, vertical=False, extent=None, **kwargs):
    if vertical:  # im is vertical
        ax.imshow(im, extent=np.array(extent)[[2, 3, 1, 0]] if extent is not None else extent, **kwargs)
    else:
        ax.imshow(np.flipud(im.T), extent=extent, **kwargs)

def ax_set_ylabel1(ax, t, vertical=False, **kwargs):
    if vertical:
        ax.set_title(t, **kwargs)
    else:
        ax.set_ylabel(t, **kwargs)

def ax_set_ylabel2(ax, t, vertical=False, **kwargs):
    if vertical:
        ax.set_xlabel(t, **kwargs)
    else:
        ax.annotate(t, rotation=90, xy=(0, 1), xycoords='axes fraction',
                    horizontalalignment='left', verticalalignment='top', **kwargs)

def fold_by_fix_len(seq, length):
    a = np.array(seq)
    rows = len(a) // length * length
    return np.reshape(a[:rows], (-1, length))

def fold_bouts_by_fix_len(seq_l, length):
    if len(seq_l) == 1:
        return fold_by_fix_len(seq_l[0], length)
    return np.vstack([fold_by_fix_len(seq, length) for seq in seq_l])

def unify_sample(ts, fps): # no interp
    idx = []
    inter = 1.0 / fps
    i = 0
    ts = np.array(ts)
    t = ts[0] - inter
    while True:
        t += inter  # first frame t=0
        if ts[i+1] < t:
            while i < len(ts) - 1 and ts[i+1] < t:
                i += 1
        if i+1 >= len(ts):
            break
        if t < ts[i]:
            idx.append(i)
            continue
        if t - ts[i] > ts[i+1] - t:
            idx.append(i+1)
        else:
            idx.append(i)
    return idx

def resample(a, to_fps, from_fps):
    return signal.resample_poly(a, to_fps, from_fps)

def bin_seq(x, xrange, bins=10):
    bin_sep = np.linspace(xrange[0], xrange[1], bins+1)
    bin_start = bin_sep[:-1]
    bin_end = bin_sep[1:]
    bin_center = (bin_start + bin_end) / 2
    x1 = np.full(x.shape, np.nan)
    for s, c, e in zip(bin_start, bin_center, bin_end):
        x1[(x >= s) & (x < e)] = c
    return x1

def bin_2d(x, y, z, xrange_para, yrange_para):
    bin_min_x, bin_max_x, bin_step_x = xrange_para
    x_bins = np.arange(bin_min_x, bin_max_x, bin_step_x)
    idx_x_bin = (x - bin_min_x) / bin_step_x
    idx_x_bin[idx_x_bin < 0] = -1  # not in range
    idx_x_bin = idx_x_bin.astype(int)
    x_n = len(x_bins)

    bin_min_y, bin_max_y, bin_step_y = yrange_para
    y_bins = np.arange(bin_min_y, bin_max_y, bin_step_y)
    idx_y_bin = (y - bin_min_y) / bin_step_y
    idx_y_bin[idx_y_bin < 0] = -1
    idx_y_bin = idx_y_bin.astype(int)
    y_n = len(y_bins)

    w = np.zeros((y_n, x_n))
    for i in range(x_n):
        for j in range(y_n):
            frame_flag = (idx_x_bin == i) & (idx_y_bin == j)
            w[j][i] = np.mean(z[frame_flag])
    return w, x_bins, y_bins

def bin_2d_yz(x, yz, xrange_para):
    bin_min_x, bin_max_x, bin_step_x = xrange_para
    x_bins = np.arange(bin_min_x, bin_max_x, bin_step_x)  # start of the bin
    idx_x_bin = (x - bin_min_x) / bin_step_x
    idx_x_bin[idx_x_bin < 0] = -1  # not in range
    idx_x_bin = idx_x_bin.astype(int)
    x_n = len(x_bins)
    w = []
    for i in range(x_n):
        zs_on_x = yz[idx_x_bin == i]
        x_trig_zs = zs_on_x.mean(axis=0)
        w.append(x_trig_zs)
    w = np.array(w).T
    return w, x_bins, np.arange(yz.shape[1])

def lines_to_err_band(py_l, USE_SEM=True):
    py = np.nanmean(py_l, axis=0)
    err_l = np.nanstd(py_l, axis=0)
    if USE_SEM:
        err_l = err_l / np.sqrt(np.count_nonzero(~np.isnan(py_l), axis=0))
    return py, err_l

def scatter_lines_with_err_band(ax, xs, ys, color, xrange, bins=10, alpha=0.5, alpha_single=0, alpha_fill=0.3):
    if alpha_single:
        ax.scatter(xs, ys, color=color, marker=".", s=1, alpha=alpha_single)
    bs = bin_seq(xs, xrange, bins)
    gr = pd.DataFrame({"bs": bs, "ys": ys}).groupby("bs")
    py = gr.mean()
    px = list(py.index)
    pe = gr.std()/np.sqrt(len(px))
    ax.plot(px, list(py["ys"]), color=color, lw=alpha)
    if alpha_fill:
        ax.fill_between(px, list((py - pe)["ys"]), list((py + pe)["ys"]), facecolor=color, alpha=alpha_fill)

def plot_lines_with_err_band(ax, py_l, color, xs=None, xs_l=None, alpha_single=0):
    lens = [len(a) for a in py_l]
    nx = np.min(lens)
    if xs is None:
        if xs_l is None:
            xs = range(nx)
        else:
            xs = xs_l[np.argmin(lens)]
    else:
        xs = xs[:nx]
    py, pe = lines_to_err_band(np.array([a[:nx] for a in py_l]))
    ax.plot(xs, py, color=color, lw=0.5)
    ax.fill_between(xs, py - pe, py + pe, facecolor=color, alpha=.3)
    if alpha_single:
        for py in py_l:
            ax.plot(xs, py[:nx], color=color, alpha=alpha_single, lw=0.1)
    return xs, py, pe

def rolling_mean(x, window=200):
    s = pd.Series(x)
    ret = s.rolling(window, center=True).mean()
    ret[:window-1] = ret[window-1]
    ret[-window:] = ret[len(ret)-window]
    return ret

def concat_list(ss):
    ret = []
    for s in ss:
        ret.extend(s)
    return ret

def save_fig(name):
    if SHOW_FIG:
        plt.show()
    else:
        if not name.endswith(FIG_EXT):
            name = name + FIG_EXT
        print(name)
        plt.savefig(name)
        plt.close("all")

def corr_slide_win(x, y, win):
    return pd.Series(x).rolling(win, center=True).corr(pd.Series(y))

def corr(x, y):
    return np.corrcoef(x, y)[0][1]

def nancorr2(x, y):
    x = np.array(x)
    y = np.array(y)
    idx = (~np.isnan(x)) & (~np.isnan(y))
    if len(x[idx]) < 2:
        return 0, 1
    return corr2(x[idx], y[idx])

def corr2(x, y):
    # r1 = np.corrcoef(x, y)[0][1]
    r, p = pearsonr(x, y)
    # print("corr", r1, "pearsonr", r, p)
    return r, p

def cir_cor(xs, ys): # circstat circ_corrcc
    mx, my = circmean(xs), circmean(ys)
    num = np.sin(xs - mx).dot(np.sin(ys - my))
    den = np.sqrt(np.sum(np.sin(xs - mx) ** 2) * np.sum(np.sin(ys - my) ** 2))
    return num / den

def cross_corr(x, y, ts):
    # t > 0 => x prior
    ret = []
    for t in ts:
        x1 = np.roll(x, t)
        ret.append(nancorr2(x1, y))
    return ret

def cross_corr_win(x, y, ts, win):
    # t > 0 => x prior
    ret = []
    for t in ts:
        x1 = np.roll(x, t)
        ret.append(corr_slide_win(x1, y, win))
    return np.array(ret)

def plot_cross_corr(ax, x, y, t_start, t_end, t_step):
    ts = np.arange(t_start, t_end+1, t_step)
    cs = cross_corr(x, y, ts)
    ax.plot(ts, cs)
    ax.axvline(0, color="k", lw=0.5)
    ax.set(xlim=(t_start, t_end), xlabel="frame", ylabel="Cross correlation")
    return ts, cs

def norm_by_percentile(x, percentile=5):
    f0 = np.mean(x[x <= np.percentile(x, percentile)])
    fm = np.mean(x[x >= np.percentile(x, 100-percentile)])
    return (x - f0) / (fm - f0)

def parse_color_str(s):
    return [float(r) for r in s.split()]

def set_pi_tick(ax, vertical=False):
    if vertical:
        ax.set_xlim(-np.pi, np.pi)
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_xticklabels(["$-\pi$", "$-\pi/2$", 0, "$\pi/2$", "$\pi$"])
    else:
        ax.set_ylim(-np.pi, np.pi)
        ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_yticklabels(["$-\pi$", "$-\pi/2$", 0, "$\pi/2$", "$\pi$"])

def is_cl_stim(config_d):
    stim_name = config_d.get("stim_name", "")
    if stim_name.startswith("CL_"):
        return True
    if stim_name.startswith("OL_"):
        return False
    if config_d.get("bg_type") == 1:  # for stim version <= 221114
        return True
    return False

def create_stim_schema(stim_config_json):
    if isinstance(stim_config_json, dict):
        stim_config = stim_config_json
    else:
        stim_config = load_dict(stim_config_json)
    is_sovr = stim_config.get("stim_name", "SoVR").startswith("SoVR")
    bg_color = parse_color_str(stim_config["bg_color"])
    dot_color = parse_color_str(stim_config["dot_color"])
    dotr_color = [1-c for c in dot_color]
    dot_width = stim_config["dot_radius"]*14 if is_sovr else stim_config["dot_width"]
    scr_width, scr_height = 256, 64
    plt.figure(figsize=(4, 2))
    ax = plt.gca()
    ax.axis("equal")
    ax.axis("off")
    ax.set(xlim=(0, scr_width), ylim=(0, scr_height))

    bg_type = int(stim_config["bg_type"])
    ax.add_patch(plt.Rectangle((0, 0), scr_width, scr_height, color=bg_color, fill=True))
    ax.annotate(BG_TYPE.get(bg_type, bg_type), (0, scr_height), ha='left', va='top', fontsize=50, color="y")

    if is_sovr:
        ax.add_patch(plt.Circle((scr_width/2, scr_height/2), dot_width/2,
                                facecolor=dot_color, edgecolor=dotr_color, lw=1, fill=True))
    else:
        dot_type = int(stim_config["dot_type"])
        if dot_type == 1:  # dot
            ax.add_patch(plt.Circle((scr_width/2, scr_height/2), dot_width/2,
                                    facecolor=dot_color, edgecolor=dotr_color, lw=1, fill=True))
        elif dot_type == 2:  # bar
            ax.add_patch(plt.Rectangle((scr_width/2-dot_width/2, 0), dot_width, scr_height,
                                       facecolor=dot_color, lw=1, edgecolor=dotr_color, fill=True))
        elif dot_type == 11:  # grating
            ax.annotate("grating", (scr_width/2, scr_height/2), ha='center', va='center', fontsize=30, color="b")
        ax.annotate("%d°, %d°/s\n%d° <--> %d°" % (stim_config["dot_width"], stim_config["dot_speed"],
                                                  stim_config["dot_start"], stim_config["dot_end"]),
                    (scr_width/2, scr_height),
                    ha='center', va='bottom', fontsize=16, color="k")
    plt.subplots_adjust(0, 0, 1)
    if not isinstance(stim_config_json, dict):
        save_fig(stim_config_json.replace(".json", ".png"))
    # os.startfile(os.path.abspath("img/temp.png"))

def plot_stim_schema_old(ax, stim_config):
    ax.axis("equal")
    ax.axis("off")
    scr_width, scr_height = 256, 64  # 2560*1280
    ax.set(xlim=(0, scr_width), ylim=(0, scr_height))
    bg_color = parse_color_str(stim_config["bg_color"])
    dot_color = parse_color_str(stim_config["dot_color"])
    dotr_color = [1 - c for c in dot_color]
    dot_width = stim_config["dot_width"]
    dot_type = int(stim_config["dot_type"])
    bg_type = int(stim_config["bg_type"])
    ax.add_patch(plt.Rectangle((0, 0), scr_width, scr_height, color=bg_color, fill=True))
    ax.annotate(BG_TYPE.get(bg_type, bg_type), (0, scr_height), ha='left', va='top', fontsize=20, color=dot_color)
    if dot_type == 1:  # dot
        ax.add_patch(plt.Circle((scr_width / 2, scr_height / 2), dot_width / 2,
                                facecolor=dot_color, edgecolor=dotr_color, lw=1, fill=True))
    elif dot_type == 2:  # bar
        ax.add_patch(plt.Rectangle((scr_width / 2 - dot_width / 2, 0), dot_width, scr_height,
                                   facecolor=dot_color, lw=1, edgecolor=dotr_color, fill=True))
    elif dot_type == 11:  # grating
        ax.annotate("pattern", (scr_width / 2, scr_height / 2), ha='center', va='center', fontsize=20, color=dot_color)
    #config_d["scr_width_deg"]
    ax.annotate("%d°, %d°/s, %d° <--> %d°, %d°" % (stim_config["dot_width"], stim_config["dot_speed"],
                                              stim_config["dot_start"], stim_config["dot_end"], stim_config.get("dot_y", 0)),
                (0, scr_height),
                ha='left', va='bottom', fontsize=16, color="k")

def plot_stim_schema(ax, stim_config):
    cl_type = stim_config.get("cl_type")
    if cl_type is None:
        return plot_stim_schema_old(ax, stim_config)
    cl_type = int(cl_type)
    ol_type = int(stim_config["ol_type"])
    bg_type = int(stim_config["bg_type"])
    bg_color = parse_color_str(stim_config["bg_color"])
    cue_color = parse_color_str(stim_config["dot_color"])
    cue_width = stim_config["dot_width"]

    ax.axis("equal")
    ax.axis("off")
    scr_width, scr_height = 256, 64  # 2560*1280
    cue_color_r = [1 - c for c in cue_color]
    center = (scr_width / 2, scr_height / 2)
    left = (5, scr_height / 2)
    ax.add_patch(plt.Rectangle((0, 0), scr_width, scr_height, color=bg_color, fill=True))
    if ol_type == 1:  # dot
        ax.add_patch(plt.Circle(center, cue_width / 2, facecolor=cue_color, edgecolor=cue_color_r, lw=1, fill=True))
    elif ol_type == 2:  # bar
        ax.add_patch(plt.Rectangle((scr_width / 2 - cue_width / 2, 0), cue_width, scr_height, facecolor=cue_color, lw=1, edgecolor=cue_color_r, fill=True))
    elif ol_type == 3:  # pattern
        ax.annotate("pat", np.array(center)+3, ha='center', va='center', fontsize=20, color=cue_color_r)
        ax.annotate("pat", center, ha='center', va='center', fontsize=20, color=cue_color)

    if cl_type == 1:  # dot
        ax.add_patch(plt.Circle(left, cue_width / 2, facecolor=cue_color, edgecolor=cue_color_r, lw=1, fill=True))
    elif cl_type == 2:  # bar
        ax.add_patch(plt.Rectangle((5, 0), cue_width, scr_height, facecolor=cue_color, lw=1, edgecolor=cue_color_r, fill=True))
    elif cl_type == 3:  # pattern
        ax.annotate("pat", left, ha='left', va='center', fontsize=20, color=cue_color)

    if bg_type == 4:  # movie
        ax.annotate("mov", left, ha='left', va='center', fontsize=20, color=cue_color)

def plot_legend(ax, heading_sign, only_ft=False):
    # rect = [0.1, 0.09, 0.7, 0.03] if vertical else [0.92, 0.8, 0.99, 0.1]
    ax.axis("off")
    rect = [0, 0, 1, 1]
    step_x = (rect[2]-rect[0])/3
    step_y = (rect[3]-rect[1])/3
    if only_ft:
        ax.text(rect[0], rect[1], "CW", color=COLOR_CW, fontsize=20, transform=ax.transAxes)
        ax.text(rect[0]+step_x, rect[1], "CCW", color=COLOR_CCW, fontsize=20, transform=ax.transAxes)
    else:
        ax.text(rect[0], rect[1], "PVA", color=COLOR_PVA, fontsize=20, transform=ax.transAxes)
        ax.text(rect[0]+step_x, rect[1], "CW", color=COLOR_CW, fontsize=20, transform=ax.transAxes)
        ax.text(rect[0]+step_x*2, rect[1], "CCW", color=COLOR_CCW, fontsize=20, transform=ax.transAxes)

    ax.text(rect[0], rect[1]+step_y, ("" if heading_sign > 0 else "-") + "heading", color=COLOR_HEADING,
            fontsize=20, transform=ax.transAxes)
    ax.text(rect[0]+step_x*2, rect[1]+step_y, "cue", color=COLOR_CUE, fontsize=20, transform=ax.transAxes)

    ax.text(rect[0], rect[1]+step_y*2, "Vs", color=COLOR_VS, fontsize=20, transform=ax.transAxes)
    ax.text(rect[0]+step_x, rect[1]+step_y*2, "Vf", color=COLOR_VF, fontsize=20, transform=ax.transAxes)
    ax.text(rect[0], rect[1]+step_y*3, "Va", color=COLOR_AV, fontsize=20, transform=ax.transAxes)
    ax.text(rect[0]+step_x, rect[1]+step_y*3, "V", color=COLOR_V, fontsize=20, transform=ax.transAxes)
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from glob import glob
from _0_constants import BALL_RADIUS


def load_ft_config(config_txt):
    ft_config = open(config_txt, "r")
    ft_config_l = []
    while True:
        line = ft_config.readline()
        if not line:
            break
        if line.startswith("#"):
            continue
        if line.find(":") > 0:
            ft_config_l.append(line.split(":"))
    ft_config_d = {line[0].strip(): line[1][:-1].strip() for line in ft_config_l}
    return ft_config_d

def load_stim_txt(stim_txt):
    stim_config = open(stim_txt, "r")
    stim_config_d = {}
    header = []
    line_no = 0
    while True:
        line = stim_config.readline()
        if not line.startswith("#"):
            break
        line_no += 1
        if line.find(":") > 0:
            tt = line.split(":")
            k, v = tt[0][1:].strip(), tt[1][:-1].strip()
            if not k.endswith("color") and not k.endswith("name"):
                v = float(v)
            stim_config_d[k] = v
        else:
            header = line[1:-1].split(", ")
    if len(header):
        rec = []
        for line in stim_config.readlines():
            if not len(line):
                break
            t = line[:-1].split()
            rec.append([tt for tt in t])
        # header.insert(2, "period")  # TODO
        stim_df = pd.DataFrame(rec, columns=header).apply(pd.to_numeric)
        stim_df = stim_df.set_index("cnt")
    else:
        stim_config.close()
        stim_df = pd.read_csv(stim_txt, skiprows=line_no, index_col=0)
        # stim_df = pd.read_csv(stim_txt, skiprows=line_no, index_col=0, delimiter=" ")
        # stim_df.columns = "cur_t,r_bar,r_dot,dot_width,pos_mx,pos_my,pos_fx,pos_fy,is_wait,pathi,path_frame".split(",")
    return stim_config_d, stim_df

"""
// frame_count
ss << _cnt << ", ";
// rel_vec_cam[3] | error
ss << _dr_cam[0] << ", " << _dr_cam[1] << ", " << _dr_cam[2] << ", " << _err << ", ";
// rel_vec_world[3]
ss << _dr_lab[0] << ", " << _dr_lab[1] << ", " << _dr_lab[2] << ", ";
// abs_vec_cam[3]
ss << _r_cam[0] << ", " << _r_cam[1] << ", " << _r_cam[2] << ", ";
// abs_vec_world[3]
ss << _r_lab[0] << ", " << _r_lab[1] << ", " << _r_lab[2] << ", ";
// integrated xpos | integrated ypos | integrated heading
ss << _posx << ", " << _posy << ", " << _heading << ", ";
// direction (radians) | speed (radians/frame)
ss << _step_dir << ", " << _step_mag << ", ";
// integrated x movement | integrated y movement (mouse output equivalent)
ss << _intx << ", " << _inty << ", ";
// timestamp | sequence number
ss << _ts << ", " << _seq << std::endl;
"""
def load_fob_dat(fob_dir):
    dat_l = glob(fob_dir + "/*.dat")
    if len(dat_l) != 1:
        print("dat not found!")
        return {}
    dat = dat_l[0]
    config_d, stim_df = load_stim_txt(fob_dir + "/stim.txt")
    ft_config_d = load_ft_config(fob_dir + "/config.txt")
    FICTRAC_RATE = int(float(ft_config_d["src_fps"]))
    if FICTRAC_RATE < 0:
        FICTRAC_RATE = 50

    FT_df = pd.read_csv(dat, header=None, index_col=0)
    # if len(FT_df) < 20*50:
    #     return None

    FT_vs = -FT_df[5] * FICTRAC_RATE * BALL_RADIUS
    FT_vf = FT_df[6] * FICTRAC_RATE * BALL_RADIUS
    FT_av = -FT_df[7] * FICTRAC_RATE
    FT_v = FT_df[18] * FICTRAC_RATE * BALL_RADIUS
    FT_x = FT_df[14]
    FT_y = FT_df[15]
    FT_heading = FT_df[16]
    FT_ts = FT_df[21]

    print(fob_dir, "v:", FT_v.mean(), "av:", np.abs(FT_av).mean())
    # mean_v = FT_v.mean()
    # if mean_v < 2 or mean_v > 50 or FT_v[-200:].mean() > 50:
    #     return None, None
    stim_df = stim_df.join(pd.DataFrame({"vs": FT_vs, "vf": FT_vf, "va": FT_av, "v": FT_v, "x": FT_x, "y": FT_y, "heading": FT_heading, "ts": FT_ts}))
    config_d["src_fps"] = FICTRAC_RATE
    if config_d.get("bg_type"):
        config_d["stim_type"] = int(config_d["bg_type"])
    config_d["duration"] = FT_ts.iloc[-1] - FT_ts.iloc[0]
    stim_df.reset_index(inplace=True)

    # if config_d.get("dot_type") == 3:
    #     stim_df["r_dot"] = lim_dir_l(stim_df["r_dot"] + stim_df["r_bar"])
    if "is_wait" not in stim_df:
        stim_df["is_wait"] = 0
        stim_df["trial"] = 0
    return {"config": config_d, "stim_df": stim_df}

def remove_bad_trials():
    import shutil
    for f in glob(r"\\192.168.1.63\nj\FoB_data\220211\*"):
        try:
            ret = load_fob_dat(f)
        except:
            ret = None
        if not ret:
            print(f)
            shutil.move(f, r"\\192.168.1.63\nj\FoB_data\remove")

def load_all_fob_dat(fob_glob, cache_name=None, replace_cache=False):
    if cache_name:
        cache = "img/" + cache_name + ".pickle"
        if not replace_cache and os.path.exists(cache):
            df = pd.read_pickle(cache)
            print("load cache", len(df))
            return df
    df = pd.DataFrame()
    for f in glob(fob_glob):
        conf, stim_df = load_fob_dat(f)
        t = os.path.basename(f)
        stim_df["fly_trial"] = t
        t = t.split("-")
        stim_df["fly"] = t[0]
        stim_df["trial"] = int(t[1])
        need_bar = int(float(conf.get("need_bar", 1)))
        if not need_bar:
            stim_df["stim_type"] = 0
        else:
            stim_df["stim_type"] = int(float(conf["stim_type"])*10+float(conf.get("landmark_type", 0)))

        df = df.append(stim_df, ignore_index=True)
    print("all fob dat:", len(df))
    if cache_name:
        df.to_pickle(cache)
    return df

# def main(fob_dir, folder=""):
#     df = load_all_fob_dat(r"\\192.168.1.63\nj\FoB_data\220215\*", replace_cache=False)
#     df = load_all_fob_dat(r"\\192.168.1.63\nj\FoB_data\220215\f*", cache_name="220215")
#     sns.displot(data=df, x="heading", y="fly_trial", col="stim_type", aspect=.7, height=8)
#
#     sns.set_theme(style="darkgrid")
#     sns.jointplot(df["vs"], df["av"], kind="reg", color="m")
#
#     g = sns.PairGrid(df[["vs", "vf", "av"]], diag_sharey=False)
#     g.map_upper(sns.scatterplot, color="m")
#     g.map_lower(sns.kdeplot)
#     g.map_diag(sns.kdeplot, lw=2)
#     sns.despine()
#
#     sns.set_theme(style="ticks")
#     g = sns.displot(data=df, x="heading", col="fly_trial", hue="stim_type", col_wrap=8)
#
#     g = sns.FacetGrid(df, col="fly_trial", hue="stim_type", col_wrap=8)
#     g.map(plt.hist, "heading", bins=50)
#     g.set(ylim=(0, 1000))
#     g.add_legend()
#     plt.savefig("img/hist_220215.png")
#     plt.show()
#
#
# if __name__ == '__main__':
#     # remove_bad_trials()
#     main(r"\\192.168.1.85\data\220525\220525_233122", folder="220525")

# def social_vr_video(pair_dir, trial_no):
#     ft_cap = cv2.VideoCapture(os.path.join(pair_dir, "fictrac-debug.avi"))
#     ft_fps = ft_cap.get(cv2.CAP_PROP_FPS)
#     # w, h = int(ft_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(ft_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#     ft_w, ft_h = 320, 320
#     stim_w = 320
#     stim_h = int(stim_w * 64 / 256)
#     w, h = ft_w + 100, ft_h + stim_h
#     trial_txt = os.path.join(pair_dir, "trial_%d.txt" % trial_no)
#     output_video = cv2.VideoWriter(trial_txt + ".avi", cv2.VideoWriter_fourcc(*"h264"), ft_fps, (w, h))
#
#     stim_config = open(os.path.join(pair_dir, "stim_config.txt"), "r")
#     stim_config_l = [line.split(": ") for line in stim_config.readlines()]
#     stim_config_d = {line[0]: line[1][:-1] for line in stim_config_l}
#     bg_color = [float(c)*255*5 for c in stim_config_d["bg_color"].split()][::-1]
#     dot_color = [float(c)*255*5 for c in stim_config_d["dot_color"].split()][::-1]
#     landmark_color = [float(c)*255*5 for c in stim_config_d["landmark_color"].split()][::-1]
#     landmark = np.array([float(c) for c in stim_config_d["landmark"].split()]).reshape((2, -1)).T
#
#     trial_f = open(trial_txt, "r")
#     text_l = []
#     last_frame = 0
#     for line in trial_f.readlines():
#         if line[0] == "#":
#             print(line)
#             text_l.append(line[:-1])
#             if len(text_l) > 5:
#                 text_l = text_l[1:]
#             continue
#         row = line[:-1].split() #cnt, cur_t, r_bar, r_dot
#         frame = int(row[0])
#         print(frame)
#         if last_frame + 1 != frame:
#             ft_cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
#         last_frame = frame
#         ret, img = ft_cap.read()
#         if not ret:
#             break
#         img_ft = img[:ft_h, :ft_w]
#
#         # TODO: virtual trajectory
#
#         r_bar, r_dot = float(row[2]), float(row[3])
#         img_stim = np.zeros((stim_h, stim_w, 3), dtype=np.uint8)
#         cv2.rectangle(img_stim, (0, 0), (stim_w, stim_h), bg_color, -1)
#         for lm_a, lm_wid in landmark:
#             if r_bar > 180:
#                 r_bar -= 360
#             x = int(stim_w * (r_bar + 135 + lm_a) / 270)
#             lm_hw = lm_wid/2
#             cv2.rectangle(img_stim, (int(x-lm_hw), 0), (int(x+lm_hw), stim_h), landmark_color, -1)
#         cv2.circle(img_stim, (int(stim_w*(r_dot+135)/270), int(stim_h/2)), 5, dot_color, -1)
#
#         img_out = np.zeros((h, w, 3), dtype=np.uint8)
#         img_out[0:stim_h, 0:stim_w] = img_stim
#         img_out[stim_h:stim_h+ft_h, 0:ft_w] = img_ft
#
#         cv2.putText(img_out, format_time(frame / ft_fps), (ft_w+6, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)
#         for i in range(len(text_l)):
#             cv2.putText(img_out, text_l[i], (ft_w+6, 20 * i + 80), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
#         output_video.write(img_out)
#     output_video.release()

# social_vr_video(r"\\192.168.1.63\nj\FoB_data\211230\F1-9", 1)

# a = pd.read_csv(r"\\192.168.1.38\nj\Imaging_data\fictrac-20221117_135614.dat", header=None, index_col=0)[21]
# # a = []
# # for l in open(r"\\192.168.1.38\nj\Imaging_data\221103\fictrac-20221103_225057.log", "r").readlines():
# #     t = l.split()[0]
# #     if not t.startswith("No"):
# #         a.append(float(t))
# a = np.array(a)
# a = a - a[0]
# d = np.diff(a)[1:]
# idx_delay = np.nonzero(d>0.5)
# import matplotlib.pyplot as plt
# plt.scatter(a[idx_delay], d[idx_delay])
# plt.show()

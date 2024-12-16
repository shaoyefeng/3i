# -*- coding: utf-8 -*-

import os
import sys
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd

from _0_constants import *
from _0_function_analysis import unify_sample, load_roi, load_tif, norm_img, format_time, smooth_angle, up_sample2, plot_angle


def write_fictrac_stim_dFF_video(pickle_name):
    parent = os.path.dirname(pickle_name)
    NI_d = pickle.load(open(pickle_name, "rb"))
    print(NI_d.keys())

    FT_BP = NI_d["FT_BP"]
    FT_DP = NI_d["FT_DP"]
    dFF_frame = NI_d["dFF_frame"]
    rate = NI_d["rate"]
    dFF_t = dFF_frame / rate
    FT_t = NI_d["FT_frame"] / rate
    # pva = NI_d["dFF_PVA"] + np.pi
    # pva = np.array(smooth_angle(NI_d["dFF_PVA"], 5)) + np.pi
    pva, dff_unify_idx = unify_sample(NI_d["dFF_PVA"], dFF_frame/rate, FT_RATE, int(dFF_frame[-1]/rate)*FT_RATE)
    pva = np.array(smooth_angle(pva, 20)) + np.pi

    # plt.figure()
    # plot_angle(plt.gca(), pva-np.pi, "r", xs=np.arange(len(pva))/FT_RATE)
    # plot_angle(plt.gca(), NI_d["dFF_PVA"], "b", xs=dFF_frame/rate)
    # plt.show()

    m = load_tif(glob(os.path.join(parent, "*.tif_avg.tif"))[0])
    m_color = m[dff_unify_idx]

    names, points, contours = load_roi(parent + "/roi.npy", m[0].shape)
    contours = contours[1:]  # remove bg
    roi_n = len(contours)  # 16 or 18

    if roi_n == 10:
        roi_n -= 2
        contours = contours[2:]  # remove NO
    pv_contour_idx = (pva*roi_n/(2*np.pi)).astype(int)  # (0~16)->(0-15)

    write_zscore_video(m_color, parent + "/v_color.avi", pv_contour_idx, contours)
    w, h = write_stim_video(parent + "/v_stim.avi", FT_BP, FT_DP)
    tt = os.path.basename(parent).split("_")
    ft_folder = os.path.join(os.path.dirname(parent), "%s-%d" % (tt[0], int(tt[-1].rstrip("+"))))
    write_fictrac_vel_video(ft_folder + "/fictrac-debug.avi", parent + "/v_vel.avi", NI_d)

    embed_video(parent + "/v_vel.avi", parent + "/v_stim.avi", parent + "/v_color.avi", 0, FT_t[0], (0, 320), (h, 320))

def plot_line_by_cv2(img, xs, ys, xlim, ylim, x_pix_range, y_pix_range, color, lw=1):
    dxs = x_pix_range[0] + (xs-xlim[0]) * ((x_pix_range[1] - x_pix_range[0]) / (xlim[1] - xlim[0]))
    dys = y_pix_range[0] + (ys-ylim[0]) * ((y_pix_range[1] - y_pix_range[0]) / (ylim[1] - ylim[0]))
    dxs = np.round(dxs).astype(int)
    dys = np.round(dys).astype(int)
    x0, y0 = dxs[0], dys[0]
    for x, y in zip(dxs[1:], dys[1:]):
        cv2.line(img, (x0, y0), (x, y), color, lw)
        x0, y0 = x, y

def update_FT_line(img, FT, seq, xlim, nframe, x_pix_range, y_pix_range, color):
    if seq > nframe:
        xs = FT[seq-nframe:seq]
    else:
        xs = FT[:seq]
    xs = xs[::-1]
    ys = np.arange(len(xs))

    plot_line_by_cv2(img, np.array([0, 0]), np.array([0, nframe]), xlim, (0, nframe), x_pix_range, y_pix_range, color)
    plot_line_by_cv2(img, xs, ys, xlim, (0, nframe), x_pix_range, y_pix_range, color)

def update_FT_pos(img, FT_x, FT_y, seq, nframe, x_pix_range, y_pix_range, color, lw=1):
    if seq > nframe:
        xs = FT_x[seq-nframe:seq]
        ys = FT_y[seq-nframe:seq]
    else:
        xs = FT_x[:seq]
        ys = FT_y[:seq]
    if len(xs) < 2:
        return
    xlim = [np.min(xs), np.max(xs)]
    ylim = [np.min(ys), np.max(ys)]
    d = xlim[1] - xlim[0] - (ylim[1] - ylim[0])*(x_pix_range[1]-x_pix_range[0])/(y_pix_range[1]-y_pix_range[0])
    if d > 0:
        ylim[1] += d*(y_pix_range[1]-y_pix_range[0])/(x_pix_range[1]-x_pix_range[0])
    else:
        xlim[1] -= d
    plot_line_by_cv2(img, xs, ys, xlim, ylim, x_pix_range, y_pix_range, color, lw=lw)

def write_fictrac_vel_video(src_video, dest_video, NI_d=None, FT_dat_name=None):
    # NOTE: dat latent 4-5 frames (0.1 s)
    if NI_d is not None:
        FT_PVM = np.array(up_sample2(NI_d["dFF_PVM"], NI_d["dFF_frame"], NI_d["FT_frame"]))
        FT_df = NI_d["FT_df"]
        FT_vs = NI_d["FT_vs"]
        FT_vf = NI_d["FT_vf"]
        FT_av = NI_d["FT_av"]
    else:
        FT_PVM = None
        FT_df = pd.read_csv(FT_dat_name, header=None).to_numpy()
        FT_vs = FT_df[:, 5] * FT_RATE * BALL_RADIUS
        FT_vf = FT_df[:, 6] * FT_RATE * BALL_RADIUS
        FT_av = FT_df[:, 7] * FT_RATE * 180 / np.pi
    FT_x = FT_df[:, 14]
    FT_y = FT_df[:, 15]

    cap = cv2.VideoCapture(src_video)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("total frame: %d, dat frame: %d" % (total_frame, len(FT_vs)))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video = cv2.VideoWriter(dest_video, cv2.VideoWriter_fourcc(*"DIVX"), FT_RATE/4, (w, h))
    nframe = 3*FT_RATE
    y_pix_range = (50, 280)
    for seq in range(len(FT_vs) - total_frame, total_frame):  # NOTE: video record lost begin frames !!!
        ret, img = cap.read()
        if not ret:
            break
        if seq > 0:
            update_FT_line(img, -FT_vs, seq, (-10, 10), nframe, (0, 100), y_pix_range, (255, 0, 255))
            update_FT_line(img, -FT_av, seq, (-400, 400), nframe, (100, 200), y_pix_range, (255, 255, 0))
            update_FT_line(img, FT_vf, seq, (-4, 20), nframe, (200, 300), y_pix_range, (0, 255, 255))
            if FT_PVM is not None:
                update_FT_line(img, FT_PVM, seq, (-0.2, 1), nframe, (200, 300), y_pix_range, (0, 0, 128))
            update_FT_pos(img, FT_y, -FT_x, seq, FT_RATE*20, (50, 200), (330, 480), (128, 128, 128))
            output_video.write(img)
    output_video.release()

def write_stim_video(name, FT_BP, FT_DP, bar_deg=0, dot_deg=28, fps=FT_RATE, delay_frames=0, is_wait=None, ts=None):
    print("write_stim_video ...")
    m_stim = []
    w, h = 300, 150
    barh, dotr = int(bar_deg * 320 / 360 / 2), int(dot_deg * 320 / 360 / 2)
    print("delay", delay_frames)
    for i in range(int(delay_frames)):
        img = np.zeros((h, w), dtype=np.uint8)
        m_stim.append(img)

    bp, dp = np.array(FT_BP), np.array(FT_DP)
    if is_wait is None:
        inv = np.zeros(len(bp))
    else:
        inv = np.array(is_wait)
    for i in range(len(bp)):
        d, d2 = bp[i], dp[i]
        img = np.zeros((h, w), dtype=np.uint8)
        if np.isnan(d) or np.isnan(d2) or inv[i]:
            m_stim.append(img)
            continue
        d_deg, d2_deg = np.rad2deg(d), np.rad2deg(d2)
        mid = int((d_deg+180) * w / 360)
        if bar_deg:
            cv2.rectangle(img, (mid - barh, 0), (mid + barh, h), 128, -1)
            cv2.putText(img, "%.2f" % np.rad2deg(d), (6, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 128), 1)
        if dot_deg:
            cv2.circle(img, (int((d2_deg+180) * w / 360), int(h / 2)), dotr, 255, -1)
            cv2.putText(img, "%.2f" % np.rad2deg(d2), (6, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 128), 1)
        m_stim.append(img)
    if ts is None:
        ts = np.arange(len(bp))/fps
    write_video(name, m_stim, fps=fps, ts=ts)
    return w, h

def write_video(path, m, cvt=cv2.COLOR_GRAY2BGR, fps=FT_RATE, ts=None):
    h, w = m[0].shape[:2]
    output_video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"DIVX"), fps, (w, h))
    for i, mi in enumerate(m):
        img_bgr = cv2.cvtColor(mi, cvt)
        if ts is not None:
            cv2.putText(img_bgr, format_time(ts[i]), (6, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)
        output_video.write(img_bgr)
    output_video.release()

def embed_video(v1, v2, v3, t2, t3, offset2=(0, 320), offset3=(100, 320), rate=FT_RATE):
    cap1 = cv2.VideoCapture(v1)
    cap2 = cv2.VideoCapture(v2)
    cap3 = cv2.VideoCapture(v3)
    n1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT) or cap3.get(cv2.CAP_PROP_FRAME_COUNT))
    cap2.set(cv2.CAP_PROP_POS_FRAMES, t2 * rate)
    cap3.set(cv2.CAP_PROP_POS_FRAMES, t3 * rate)
    w, h = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video = cv2.VideoWriter(v2 + "_combine.avi", cv2.VideoWriter_fourcc(*"DIVX"), rate, (w, h))
    for seq in range(0, n1):
        ret, img1 = cap1.read()
        if not ret:
            break
        ret, img2 = cap2.read()
        if not ret:
            break
        ret, img3 = cap3.read()
        if not ret:
            break
        h2, w2, c = img2.shape
        img1[offset2[0]:offset2[0]+h2, offset2[1]:offset2[1]+w2, :] = img2

        h3, w3, c = img3.shape
        # img1[offset3[0]:offset3[0]+h3, offset3[1]:offset3[1]+w3, :] = img3
        w3s = w - offset3[1]
        h3s = int(w3s/w3*h3)
        img3s = cv2.resize(img3, (w3s, h3s))
        img1[offset3[0]:offset3[0]+h3s, offset3[1]:offset3[1]+w3s, :] = img3s
        h3b = h3s + offset3[0]
        if h3b < h:
            img1[h3b:, offset3[1]:offset3[1]+w3s, :] = 0

        output_video.write(img1)
    output_video.release()

def write_zscore_video(m, name, pv_idx=None, pva=None, contours=None, fps=FT_RATE, ts=None, r_bar=None, scale=1):
    # 221223_183354 video
    # 221223_161453 video
    m = np.array([cv2.GaussianBlur(mi, (9, 9), 0) for mi in m])
    m_mean = np.mean(m, axis=0)
    m_std = np.std(m, axis=0)
    h, w = m_mean.shape

    # NOTE: F
    # m1 = [norm_img(mi) for mi in m]

    # NOTE: F-mean
    m1 = m - m_mean
    m1[m1 < 0] = 0
    m1 = [norm_img(mi) for mi in m1]

    # NOTE: zs
    # mz = np.array([(mi - m_mean) / m_std for mi in m])
    # mz[mz < 0] = 0
    # m1 = [norm_img(mi) for mi in mz]

    # NOTE: zs norm
    # mz = np.array([(mi - m_mean) / m_std for mi in m])
    # mz[mz < 0] = 0
    # m_min = np.percentile(mz, 5, axis=0)
    # m_max = np.percentile(mz, 95, axis=0)
    # mz_norm = []
    # for mi in mz:
    #     zs_norm = 255 * (mi - m_min) / (m_max - m_min)
    #     np.clip(zs_norm, 0, 255, zs_norm)
    #     mz_norm.append(zs_norm)
    # m1 = mz_norm

    cmap = plt.cm.get_cmap("plasma")

    output_video = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*"DIVX"), fps, (int(w * scale), int(h * scale)))
    for i, mi in enumerate(m1):
        if scale > 1:
            mi = cv2.resize(mi, (int(w * scale+.5), int(h * scale+.5)))
        img = mi.astype(np.uint8)#cv2.GaussianBlur(mi.astype(np.uint8), (3, 3), 0)
        img_c = np.dstack([img, img, img])
        # img_c = (cmap(mi)[:, :, :3] * 255).astype(np.uint8)
        if pv_idx is not None and i < len(pv_idx) and pv_idx[i] >= 0:
            cv2.putText(img_c, "%.2f" % pv_idx[i], (6, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 128), 1)
            cv2.putText(img_c, format_time(ts[i]), (6, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)
            # cv2.drawContours(img_c, [(contours[int(pv_idx[i]+.5)]*scale).astype(int)], 0, color=(255, 255, 255), thickness=int(scale))
        if r_bar is not None:
            rx = h * scale / 3
            cv2.circle(img_c, (int(rx*np.cos(-r_bar[i])+w * scale/2+.5), int(rx*np.sin(-r_bar[i])+h * scale/2+.5)), 5, (0, 0, 255), -1)
        output_video.write(cv2.cvtColor(img_c, cv2.COLOR_RGB2BGR))
    output_video.release()

    # write_video(name, (cmap(m2)[:,:,:,:3] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR, fps=fps, ts=ts)

def write_ft_vel_video(output_avi, input_avi, x1, x2, y1, y2, df, need_v=True, traj_video=False):
    print("write_ft_vel_video ...")
    cap = cv2.VideoCapture(input_avi)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc_s = "DIVX" if output_avi.endswith("avi") else "mp4v"
    fourcc = cv2.VideoWriter_fourcc(*fourcc_s)
    output_video = cv2.VideoWriter(output_avi, fourcc, fps, (x2 - x1, y2 - y1))
    if traj_video:
        traj_h, traj_w = 224, x2-x1
        traj_video = cv2.VideoWriter(traj_video, fourcc, fps, (traj_w, traj_h))
    y_pix_range = (50-y1, 270)
    nframe = 3*FT_RATE
    for seq in range(len(df) - total_frame, total_frame):  # NOTE: video record skip 100 frames (wait 2s) !!!
        ret, img = cap.read()
        if not ret:
            break
        if seq > 0:
            img = img[y1:y2, x1:x2]
            if need_v:
                update_FT_line(img, -np.array(df["vs"]), seq, (-10, 10), nframe, (0, 100), y_pix_range, (255, 0, 255))
                update_FT_line(img, -np.rad2deg(np.array(df["av"])), seq, (-400, 400), nframe, (100, 200), y_pix_range, (255, 255, 0))
                update_FT_line(img, np.array(df["vf"]), seq, (-4, 20), nframe, (200, 300), y_pix_range, (0, 255, 255))
                # if FT_PVM is not None:
                #     update_FT_line(img, FT_PVM, seq, (-0.2, 1), nframe, (200, 300), y_pix_range, (0, 0, 128))
                if traj_video:
                    img2 = np.zeros((traj_h, traj_w, 3), dtype=np.uint8)
                    update_FT_pos(img2, np.array(df["y"]), -np.array(df["x"]), seq, FT_RATE*20, (10, traj_w-10), (10, traj_h-10), (255, 255, 255), lw=1)
                    traj_video.write(img2)
                else:
                    update_FT_pos(img, np.array(df["y"]), -np.array(df["x"]), seq, FT_RATE*20, (50, 200), (330, 480), (128, 128, 128))
            output_video.write(img)
    cap.release()
    output_video.release()
    traj_video and traj_video.release()

def write_ima_video(output_avi, tif_file, roi_file, pva, up_idx, roi_n, ts=None, scale=1, r_bar=None):
    print("write_ima_video ...")
    m = load_tif(tif_file)
    m_color = m[up_idx]

    names, points, contours = load_roi(roi_file, m[0].shape)
    contours = contours[-roi_n:]
    pv_contour_idx = ((pva+np.pi) * roi_n / (2 * np.pi)).astype(int)  # (-pi~pi)->(0-15)
    write_zscore_video(m_color, output_avi, pv_contour_idx, pva, contours, ts=ts, scale=scale, r_bar=r_bar)

def crop_video(output_avi, input_avi, x1, x2, y1, y2):
    cap = cv2.VideoCapture(input_avi)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    output_video = cv2.VideoWriter(output_avi, fourcc, fps, (x2 - x1, y2 - y1))
    for seq in range(0, total_frame):
        ret, img = cap.read()
        if not ret:
            break
        output_video.write(img[y1:y2, x1:x2])
    cap.release()
    output_video.release()

g_tmp_no = 0
def stack_video(output_avi, avi1, avi2, hori=False):
    if output_avi is None:
        global g_tmp_no
        output_avi = "temp/tmp_%06d.avi" % np.random.randint(1000000)
        g_tmp_no += 1
    print("stack_video", avi1, avi2)
    cap1 = cv2.VideoCapture(avi1)
    cap2 = cv2.VideoCapture(avi2)
    fps = cap1.get(cv2.CAP_PROP_FPS)
    total_frame1 = cap1.get(cv2.CAP_PROP_FRAME_COUNT)
    total_frame2 = cap2.get(cv2.CAP_PROP_FRAME_COUNT)
    print("avi1:avi2 %d:%d frames" % (total_frame1, total_frame2))
    total_frame = int(min(total_frame1, total_frame2))
    width1, height1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2, height2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if hori:
        width = width1 + width2
        height = max(height1, height2)
    else:
        height = height1 + height2
        width = max(width1, width2)
    fourcc_s = "DIVX" if output_avi.endswith("avi") else "mp4v"
    fourcc = cv2.VideoWriter_fourcc(*fourcc_s)
    output_video = cv2.VideoWriter(output_avi, fourcc, fps, (width, height))
    for seq in range(0, total_frame):
        ret1, img1 = cap1.read()
        ret2, img2 = cap2.read()
        if not ret1 or not ret2:
            break
        scr = np.zeros((height, width, 3), dtype=img1.dtype)
        if hori:
            top = int((height-height1)/2)
            scr[top:top+height1, :width1] = img1
            top = int((height-height2)/2)
            scr[top:top+height2, width1:] = img2
        else:
            left = int((width-width1)/2)
            scr[:height1, left:left+width1] = img1
            left = int((width-width2)/2)
            scr[height1:, left:left+width2] = img2
        output_video.write(scr)
    cap1.release()
    cap2.release()
    output_video.release()
    return output_avi

# def write_fig_to_video(output_video, fig, w, h, desc=None, save_img_path=None):
#     fig.canvas.draw()
#     img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
#     img.shape = (h*2, w*2, 3)
#     img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     desc and cv2.putText(img_bgr, desc, (6, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 128), 1)
#     output_video.write(img_bgr)
#     save_img_path and cv2.imwrite(save_img_path, img_bgr)
#     return img_bgr

if __name__ == '__main__':
    write_fictrac_stim_dFF_video(sys.argv[1])
    # ft_folder = os.path.dirname(sys.argv[1])#"\\192.168.1.15\lqt\LQTdata\NJ\211228\F1-13"
    # write_fictrac_vel_video(ft_folder + "/fictrac-debug.avi", ft_folder + "/v_vel.avi", FT_dat_name=glob(ft_folder + "/*.dat")[0])
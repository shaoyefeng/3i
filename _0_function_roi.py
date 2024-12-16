# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


SCALE_SPEED = 0.01
ROTATE_SPEED = 1

COLORS = ["k", "r", "g", "b", "y", "c", "m", "gray", "pink", "springgreen", "deepskyblue", "yellow",]

def contours_bbox(contours):
    if isinstance(contours, list):
        c = np.concatenate(contours)
    else:
        c = contours
    return c[:, 0, 0].min(), c[:, 0, 1].min(), c[:, 0, 0].max(), c[:, 0, 1].max()

def contour_center(c):
    return (c[:, 0, 0].min() + c[:, 0, 0].max())/2, (c[:, 0, 1].min() + c[:, 0, 1].max())/2

def roi_contours_to_points(contours_xy, shape):
    xy = []
    for c in contours_xy:
        temp = np.zeros(shape)
        cv2.drawContours(temp, [c.astype(int)], 0, color=1, thickness=-1)
        xy.append(temp.nonzero())
        # plt.imshow(temp, cmap="Greys_r")
        # plt.show()
    return xy

def plot_contours(ax, contours, flags, texts=None):
    for i, c in enumerate(contours):
        xs, ys = c[:, 0, 0], c[:, 0, 1]
        ax.plot(np.concatenate([xs, xs[:1]]), np.concatenate([ys, ys[:1]]), linestyle="--" if flags[i] else "-")#, alpha=0.6)
        if texts is not None and texts[i]:
            x, y = contour_center(c)
            ax.text(x, y, texts[i])

def plot_rois(rois, m):
    plt.figure(figsize=(3, 3), dpi=300)
    plt.axis("off")
    ax = plt.gca()
    ax.set_position([0, 0, 1, 1], which="both")
    ax.imshow(m, cmap=plt.cm.gray)#norm_img(np.std(m, axis=0))
    for i, xys in enumerate(rois):
        ax.add_patch(plt.Polygon(xys, alpha=0.6, fill=False, linewidth=1, color=COLORS[i%12]))
        # x, y = xys.min(axis=0)/2 + xys.max(axis=0)/2
        pts = roi_contours_to_points([xys], m.shape)
        y, x = np.mean(pts[0], axis=1)
        ax.text(x, y, str(i), color="r")

class ROITemplateUI(object):
    def __init__(self, file, threshold=245):
        self.file = file
        if file.endswith(".png"):
            gray = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            ret, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            _, self.contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.RETR_CCOMP)  #cv2.RETR_EXTERNAL
        else:
            self.contours = np.load(file, allow_pickle=True)
        self.flags = np.ones((1000, ), np.bool).tolist()

        fig, self.ax = plt.subplots(figsize=(8, 6), num="ROI Template")
        plt.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.15)
        fig.canvas.mpl_connect('key_press_event', self.onkey)
        fig.canvas.mpl_connect('button_press_event', self.onclick)
        fig.canvas.mpl_connect("close_event", self.onclose)
        self.refresh()

    def refresh(self):
        self.ax.cla()
        texts = []
        i = 0
        for f in self.flags:
            if f:
                texts.append(str(i))
                i += 1
            else:
                texts.append(None)
        plot_contours(self.ax, self.contours, self.flags, texts)
        self.ax.invert_yaxis()
        print(self.flags)
        plt.draw()

    def save_result(self):
        if self.file.endswith(".png"):
            sel = []
            for i, c in enumerate(self.contours):
                if self.flags[i]:
                    sel.append(c)
            np.save(self.file.replace(".png", ".npy"), sel)

    def onclick(self, event):
        if event.xdata and event.ydata:
            if event.button == 3:
                for i, c in enumerate(self.contours):
                    if self.flags[i] and cv2.pointPolygonTest(c, (event.xdata, event.ydata), False) == 1:
                        self.contours.pop(i)
                        self.contours.insert(0, c)
                        self.flags.pop(i)
                        self.flags.insert(0, True)
                        break
            else:
                for i, c in enumerate(self.contours):
                    if cv2.pointPolygonTest(c, (event.xdata, event.ydata), False) == 1:
                        self.flags[i] = not self.flags[i]
            self.refresh()

    def onkey(self, event):
        if event.key == "enter":
            self.save_result()
        if event.key == "z":
            for i, c in enumerate(self.contours):
                self.flags[i] = False
            self.refresh()

    def show(self):
        plt.show()

    def onclose(self, event):
        self.save_result()


class ROIModifyUI(object):
    def __init__(self, template, bg_file):
        self.file = bg_file
        self.bg_img = cv2.imread(bg_file)
        # if template and os.path.exists(template):
        #     use_template = template.find("template") > 0
        #     self.contours = np.load(template, allow_pickle=True)
        #     self.contours = [c.astype(float) for c in self.contours]
        if template and os.path.exists(template):
            use_template = True
            self.contours = np.load(template, allow_pickle=True)

        else:
            use_template = False
            roi_file = os.path.join(os.path.dirname(bg_file), "roi.npy")
            if os.path.exists(roi_file):
                result = np.load(roi_file, allow_pickle=True)
            else:
                result = []
            self.contours = result
        self.ins_points = []
        self.flags = np.ones((1000, ), dtype=bool)
        self.is_press_left = False
        self.is_press_right = False
        self.is_ctrl = False
        self.is_shift = False
        self.last_move = None
        self.undo_contours = None
        # self.fig, self.ax = plt.subplots(figsize=(8, 8), num="Modify ROI " + bg_file)
        self.fig, self.ax = plt.subplots(figsize=(20, 10), num="Modify ROI " + bg_file)
        plt.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.05)
        self.reset_event()
        # if use_template:
        #     self.scale_contours_to_center()
        # if len(self.contours) == 10:
        #     self.contours.append(np.array([[[92, 9]], [[93, 22]], [[105, 22]], [[106, 6]]]))
        self.refresh()

    def reset_event(self):
        self.fig.canvas.mpl_connect('key_press_event', self.onkeypress)
        self.fig.canvas.mpl_connect('key_release_event', self.onkeyrelease)
        self.fig.canvas.mpl_connect('button_press_event', self.onpress)
        self.fig.canvas.mpl_connect('motion_notify_event', self.onmove)
        self.fig.canvas.mpl_connect('button_release_event', self.onrelease)
        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)
        self.fig.canvas.mpl_connect("close_event", self.onclose)

    def scale_contours_to_center(self):
        if not len(self.contours):
            return
        l, t, r, b = contours_bbox(self.contours)
        cx, cy = (l + r) / 2, (t + b) / 2
        ix, iy = self.bg_img.shape[1] / 2, self.bg_img.shape[0] / 2
        sx, sy = ix - cx, iy - cy
        sc = self.bg_img.shape[1] / (r - l) * 0.8
        for c in self.contours:
            c[:, 0, 0] = (c[:, 0, 0] - cx) * sc + ix
            c[:, 0, 1] = (c[:, 0, 1] - cy) * sc + iy

    def scale_contours(self, contours, stepx, stepy):
        if not len(contours):
            return
        l, t, r, b = contours_bbox(contours)
        cx, cy = (l + r) / 2, (t + b) / 2
        for c in contours:
            c[:, 0, 0] = (c[:, 0, 0] - cx) * (1 + stepx * SCALE_SPEED) + cx
            c[:, 0, 1] = (c[:, 0, 1] - cy) * (1 + stepy * SCALE_SPEED) + cy

    def move_contours(self, contours, dx, dy):
        for c in contours:
            c[:, 0, 0] += dx
            c[:, 0, 1] += dy

    def rotate_contours(self, contours, angle):
        if not len(contours):
            return
        angle = np.deg2rad(angle) * ROTATE_SPEED
        l, t, r, b = contours_bbox(contours)
        cx, cy = (l + r) / 2, (t + b) / 2
        for c in contours:
            x1, y1 = c[:, 0, 0] - cx, c[:, 0, 1] - cy
            c[:, 0, 0] = x1 * np.cos(angle) - y1 * (np.sin(angle)) + cx
            c[:, 0, 1] = x1 * np.sin(angle) + y1 * (np.cos(angle)) + cy

    def get_sel_contours(self):
        ret = []
        for i, c in enumerate(self.contours):
            if self.flags[i]:
                ret.append(c)
        return ret

    def refresh(self):
        self.ax.cla()
        self.ax.imshow(self.bg_img)
        self.ax.set_xlim(0, self.bg_img.shape[1])
        self.ax.set_ylim(self.bg_img.shape[0], 0)
        plot_contours(self.ax, self.contours, self.flags, [str(i) for i in range(len(self.contours))])
        if len(self.ins_points):
            self.ax.plot([x for x, y in self.ins_points], [y for x, y in self.ins_points], ".-", c="r", lw=1)
        plt.draw()

    def save_result(self):
        np.save(os.path.join(os.path.dirname(self.file), "roi.npy"), self.contours)
        self.save_csv_result()

    def save_csv_result(self):
        import pandas as pd
        res = []
        for i, c in enumerate(self.contours):
            res.extend([[i, p[0][0], p[0][1]] for p in c])
        pd.DataFrame(res, columns=["id", "x", "y"]).to_csv(os.path.join(os.path.dirname(self.file), "roi.csv"), index=False)

    def onpress(self, event):
        if event.xdata and event.ydata:
            self.last_move = None
            if event.button == 1:
                self.is_press_left = True
            elif event.button == 3:
                self.is_press_right = True

    def onmove(self, event):
        if event.xdata and event.ydata:
            if self.is_press_left:  # NOTE: move
                if self.last_move is not None:
                    self.move_contours(self.get_sel_contours(), event.xdata - self.last_move[0], event.ydata - self.last_move[1])
                self.last_move = event.xdata, event.ydata
                self.refresh()
            elif self.is_press_right:  # NOTE: rotate by x
                if self.last_move is not None:
                    self.rotate_contours(self.get_sel_contours(), event.x - self.last_move[0])
                self.last_move = event.x, event.y
                self.refresh()

    def onrelease(self, event):
        if self.last_move is None:
            if self.is_press_left:
                for i, c in enumerate(self.contours):
                    if cv2.pointPolygonTest(c.astype(int), (event.xdata, event.ydata), False) == 1:
                        if self.is_shift:
                            self.flags[i] = not self.flags[i]
                        else:
                            for j in range(len(self.contours)):
                                self.flags[j] = False
                            self.flags[i] = True
                        self.refresh()
                        break
            elif self.is_press_right:  # NOTE: insert point
                self.ins_points.append([event.xdata, event.ydata])
                self.refresh()

        if event.button == 1:
            self.is_press_left = False
        elif event.button == 3:
            self.is_press_right = False

    def onscroll(self, event):
        print(event.step)
        if self.is_ctrl:
            self.scale_contours(self.get_sel_contours(), 0, event.step)
        elif self.is_shift:
            self.scale_contours(self.get_sel_contours(), event.step, 0)
        else:
            self.scale_contours(self.get_sel_contours(), event.step, event.step)
        self.refresh()

    def confirm_insert_contour(self):
        if not len(self.ins_points):
            return
        contour = np.array([[p] for p in self.ins_points])
        self.contours.insert(0, contour)
        self.ins_points = []
        self.refresh()

    def onkeypress(self, event):
        print(event.key)
        global g_contours_copy
        if event.key == "enter":
            self.confirm_insert_contour()
            self.save_result()
        elif event.key == "control":
            self.is_ctrl = True
        elif event.key == "shift":
            self.is_shift = True
        elif event.key == "ctrl+z":
            if len(self.ins_points):
                self.ins_points.pop()
                self.refresh()
        elif event.key == "ctrl+c":
            g_contours_copy = self.contours.copy()
        elif event.key == "ctrl+v":
            self.undo_contours = self.contours.copy()
            if g_contours_copy is not None:
                self.contours = g_contours_copy
            self.refresh()
        elif event.key == "ctrl+z":
            if self.undo_contours is not None:
                self.contours = self.undo_contours
        elif event.key == "ctrl+a":
            for i in range(len(self.contours)):
                self.flags[i] = True
            self.refresh()
        elif event.key == "A":
            for i in range(len(self.contours)):
                self.flags[i] = False
            self.refresh()

    def onkeyrelease(self, event):
        if event.key == "control":
            self.is_ctrl = False
        elif event.key == "shift":
            self.is_shift = False

    def set_window_pos(self, x, y):
        mngr = plt.get_current_fig_manager()
        mngr.window.wm_geometry("+%d+%d" % (x, y))

    def show(self):
        # self.set_window_pos(500, 600)
        plt.show()
        # draw_roi_png(os.path.dirname(self.file))

    def onclose(self, event):
        self.save_result()

def create_template(ref_fig, threshold):
    # *.png, 100
    ROITemplateUI(ref_fig, threshold).show()  # PB 100 EB 245

def define_roi_no_show(template, bg_file):
    return ROIModifyUI(template, bg_file)

def define_roi(template, bg_file):
    # FB.npy, i_std.png
    ROIModifyUI(template, bg_file).show()

def draw_roi_png(trial_folder):
    roi_file = trial_folder + "/roi.npy"
    bg_file = trial_folder + "/i_std.png"
    png_file = trial_folder + "/roi.png"
    contours = np.load(roi_file, allow_pickle=True)
    plot_rois([r[:, 0, :] for r in contours], cv2.imread(bg_file, cv2.IMREAD_GRAYSCALE))
    plt.savefig(png_file)


import os
import sys
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

from _0_function_analysis import cmd_to_time_glob

PDF_SHOW_TITLE = False
FONT = cv2.FONT_HERSHEY_COMPLEX_SMALL
COLOR = (0, 0, 0)
FONT_SIZE = 1
FONT_BOLD = 1
def img_grid(img_l, title_l=None, cols=1, gapw=0, gaph=0, output_path=None, fix_h=None):
    if output_path and output_path.endswith(".pdf"):
        return img_grid_pdf(img_l, title_l, cols, gap, output_path)
    n = len(img_l)
    if n == 0:
        return None
    rows = n//cols
    if n%cols > 0:
        rows += 1
    im = cv2.imread(img_l[0]) if isinstance(img_l[0], str) else img_l[0]
    if title_l is None:
        title_l = [None] * n
    h, w, c = im.shape
    if fix_h:
        h = fix_h
    width = (w + gapw) * cols
    height = (h + gaph) * rows
    m = np.ones((height, width, c), np.uint8)*255
    for i, (img, title) in enumerate(zip(img_l, title_l)):
        col = i % cols
        row = i // cols
        x_start = (w + gapw) * col + gapw
        y_start = (h + gaph) * row + gaph
        if isinstance(img, str):
            print(img)
        im = cv2.imread(img) if isinstance(img, str) else img
        if im is not None:
            h1, w1, c1 = im.shape
            m[y_start:y_start+h1, x_start:x_start+w1] = im
        if title:
            cv2.putText(m, title, (x_start, y_start-10), FONT, FONT_SIZE, COLOR, FONT_BOLD)
    if output_path:
        cv2.imwrite(output_path, m)
        print(output_path)
    return m

def img_grid_pdf(img_l, title_l, cols, gap, output_path):
    n = len(img_l)
    rows = n//cols
    if n%cols > 0:
        rows += 1
    figsize = cols*2.5, rows*2.2
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = axs.flatten()
    for i, (img, title) in enumerate(zip(img_l, title_l)):
        ax = axs[i]
        ax.axis("off")
        if PDF_SHOW_TITLE:
            ax.set_title(title, fontsize=30)
        im = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB) if isinstance(img, str) else img
        ax.imshow(im)

    for ax in axs[i:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)

def img_grid_glob(g, out_name, cols=4, base_cut=7):
    fl = g if isinstance(g, list) else glob(g)
    if isinstance(base_cut, str):
        title_l = [os.path.basename(f).split(base_cut)[0] for f in fl]
    else:
        title_l = [os.path.basename(f)[:-base_cut] for f in fl]
    img_grid(fl, title_l, cols, 60, out_name)

def img_grid_files_col(fs, out_name, title_l=None):
    imgs = [img_grid(f) for f in fs]
    h = np.max([im.shape[0] for im in imgs if im is not None])
    os.makedirs(os.path.dirname(out_name), exist_ok=True)
    img_grid(imgs, cols=len(imgs), output_path=out_name, title_l=title_l, gaph=30, fix_h=h)


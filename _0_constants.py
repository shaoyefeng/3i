# -*- coding: utf-8 -*-
import numpy as np

fps = 7
stim_t_l = np.array([19, 79, 139, 199])

BALL_RADIUS = 3  #mm

# video
stim_start = -135
stim_end = 135
scr_deg = 180
scr_w = 2160  # 128*2 pixel
scr_h = 1080  # 64 pixel
scr_deg_per_pix = scr_deg / scr_w  # deg/pixel
video_w = scr_w
video_h = scr_h


# looming
r = 8  # cm
v = 160  # cm/s
duration = 2  # s
persist = 2  # s
t = np.arange(-duration, 0, 1/fps)
ang = 2*np.rad2deg(np.arctan(r/(-v*t)))
mi, ma = np.min(ang), np.max(ang)
wid = np.concatenate([(ang-mi)/(ma-mi) * video_h, np.full((int(persist*fps),), video_h)])
looming_size = wid * scr_deg_per_pix


DEFAULT_DATE = "230522"
DEFALUT_STIM = "CL_bar"
ROOT = r"\\192.168.1.38\nj\Imaging_data"
# ROOT = r"\\192.168.1.38\nj\FoB_data"
# ROOT = r"C:\fictrac\data"
IMG_ROOT = "img/"
# IMG_ROOT = "img_fob/"
DATA_ROOT = "data/"
# DATA_ROOT = "data_fob/"
IS_FOB = (IMG_ROOT == "img_fob/")

SUB_ROOT = ""
#SUB_ROOT = "/_fast/"
ROOT = ROOT + SUB_ROOT
IMG_ROOT = IMG_ROOT + SUB_ROOT

EXP_TYPE = "2P"  # LFM, 2PGG
# EXP_TYPE = "Shim"

LOAD_CACHE = False
SKIP_PLOT = False

SYNC_THRESHOLD = 0.05
SYNC_RATE = 30000  #30000
FT_RATE = 50
UNIFY_FPS = 0  # 0: ima_fps(8.79), 50: same as TwopExp
BALL_RADIUS = 4  #mm

FILTER_MIN_DURATION = 25
FILTER_MIN_SPEED = 2
FILTER_MAX_TSPEED = 25

GENO_MAP = {
    "CX1001": "EPG",

    "CX1004": "PFNpm",

    "CX019": "PFNd",
    "CX1013": "PFNd",
    "CX1015": "PFNd",

    "CX1016": "PFNv",
    "CX1017": "PFNv",
    "CX1018": "PFNv",

    "CX026": "PFR",
    "CX1038": "PFR",
    "CX1089": "PFR",

    "CX027": "PFLx",
    "CX1039": "PFL",
    "CX1088": "PFL",

    "CX1035": "PFG",
    "CX1087": "PFG",
}
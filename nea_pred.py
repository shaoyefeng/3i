import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from glob import glob

from _0_function_analysis import up_sample2
from plot_utils import rolling_mean

def LR_test(X, y, label, label_pred):
    lr = LinearRegression().fit(X, y)
    y1 = lr.predict(X)
    plt.figure(figsize=(12, 2))
    plt.plot(y, label=label)
    plt.plot(y1, label=label_pred)
    plt.legend()
    print("MSE: %.2f, R2: %.2f" % (mean_squared_error(y, y1), R2(y, y1)))

    plt.figure()
    plt.plot(lr.coef_)
    print("coef:", lr.coef_)
    plt.show()

def R2(y_real, y_pred):
    ssr = np.sum((y_pred - np.mean(y_real))**2)
    sst = np.sum((y_real - np.mean(y_real))**2)
    return ssr/sst

def load_dataset(path):
    data = pickle.load(open(path, 'rb'))
    stim_df = data["stim_df"]
    dff_df = data["dFF_df"]
    stim_df["FT_PVA"] = up_sample2(data["dFF_PVA"], data["dFF_frame"], stim_df["FT_frame"])
    stim_df["FT_PVM"] = up_sample2(data["dFF_PVM"], data["dFF_frame"], stim_df["FT_frame"])
    stim_df["FT_MZS"] = up_sample2(data["zscore_mean"], data["dFF_frame"], stim_df["FT_frame"])

    dff_up = []
    for roi in range(dff_df.shape[1]):
        dff_up.append(up_sample2(dff_df[:, roi], data["dFF_frame"], stim_df["FT_frame"]))
    FT_dFF_df = np.vstack(dff_up).T
    return stim_df, FT_dFF_df

g = r"D:\exp_FoB\foba\data\2P_PFN\*\221103_172909.pickle"
f = glob(g)[0]
stim_df, FT_dFF_df = load_dataset(f)
# LR_test(stim_df["FT_MZS"].to_numpy().reshape(-1, 1), rolling_mean(stim_df["vf"], 25))
LR_test(FT_dFF_df, rolling_mean(stim_df["vf"], 25), "vf", "f(dF/F)")
LR_test(FT_dFF_df, rolling_mean(stim_df["av"], 25), "av", "f(dF/F)")

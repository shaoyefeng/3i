import os
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fps=7
trial_path = r"F:\Project\Physiology\3i\Data\MBON11XVC91\231207\MBON11XVC91_fly1\2p_300_4sti"
df_path = os.path.join(trial_path, 'F.csv')
df = pd.read_csv(df_path)
names = df.columns
ratio_df = pd.DataFrame([])
ratio_df['frame'] = df.index
ratio_df['time'] = ratio_df['frame'] / fps
for g in names[1:]:
    f = df[g] - df['0']
    f0 = np.average(f.iloc[-50:-1])
    ratio_df['ROI_' + str(g)] = (f - f0) / f0
ratio_df.to_csv(df_path.replace('F.csv', 'ratio.csv'), index=False)
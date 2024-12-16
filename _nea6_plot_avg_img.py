import os, sys
from glob import glob

from img_grid import img_grid_files_col
from _0_constants import DEFAULT_DATE
from _0_function_analysis import cmd_to_time_glob


def proc_date(d):
    time_glob = cmd_to_time_glob(d)
    fs = []
    col_name_l = []
    title_l = []
    last_fly = None
    for folder in glob(time_glob):
        avg = glob(os.path.join(folder, "*/i_avg.png"))
        if len(avg) > 0:
            parent = os.path.dirname(avg[0])
            fly_dir, time_name = os.path.split(folder)
            tt = os.path.basename(fly_dir).split("-")
            fly = tt[1]
            geno = tt[2][3:]
            if fly != last_fly:
                title_l.append(geno)
                title_l.append([])
                if last_fly:
                    fs.append([])
            else:
                title_l.append([])
            last_fly = fly
            avg_imgs = glob(os.path.join(parent, "*avg*.png"))
            fs.append(avg_imgs)
            col_name_l.append("%s-%s-%s" % (fly, geno, time_name))
    row_name_l = [os.path.basename(f).split("_")[-1][:-4] for f in fs[0]]
    print(row_name_l, title_l)
    img_grid_files_col(fs, "img/_Date/%s.png" % d, title_l)

    info = open("img/_Date/%s.txt" % d, "w")
    info.writelines(" ".join(row_name_l))
    info.write("\n")
    info.writelines("\n".join(col_name_l))

def proc_all():
    geno_d = {}
    for img_folder in glob(r"\\192.168.1.38\nj\Imaging_data\2*"):
        time_glob = cmd_to_time_glob(os.path.basename(img_folder))
        last_fly = None
        print(time_glob)
        for folder in glob(time_glob):
            avg = glob(os.path.join(folder, "*/i_avg.png"))
            if len(avg) > 0:
                parent = os.path.dirname(avg[0])
                fly_dir, time_name = os.path.split(folder)
                tt = os.path.basename(fly_dir).split("-")
                fly = tt[1]
                geno = tt[2][3:]

                if fly != last_fly:
                    geno_d.setdefault(geno, [])
                    geno_d[geno].append(glob(os.path.join(parent, "*avg*.png")))
                last_fly = fly
    for geno, fs in geno_d.items():
        img_grid_files_col(fs, "img/_Date/%s.png" % geno)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        cmd = DEFAULT_DATE
    else:
        cmd = sys.argv[1]
    proc_date(cmd)
    # proc_all()
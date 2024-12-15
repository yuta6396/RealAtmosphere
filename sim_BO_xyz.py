import os
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import subprocess

import cartopy.crs as ccrs
import gradsio2 as gio
import matplotlib.ticker as mticker
import imageio  # GIF作成用
from datetime import datetime

# 時刻を計測するライブラリ
import time
import pytz
from datetime import datetime
from zoneinfo import ZoneInfo

import warnings
# BO用
from skopt import gp_minimize
from skopt.space import Integer
from skopt.learning import GaussianProcessRegressor
from skopt.acquisition import gaussian_ei
from skopt.space import Space
from skopt.utils import use_named_args
from joblib import Parallel, delayed
from skopt import Optimizer

import random
matplotlib.use('Agg')

# 警告を抑制
warnings.filterwarnings('ignore', category=DeprecationWarning)

jst = pytz.timezone('Asia/Tokyo')# 日本時間のタイムゾーンを設定
current_time = datetime.now(jst).strftime("%m-%d-%H-%M")

nofpe = 4
fny = 2
fnx = 2
X_size = 90
Y_size = 90

TIME_INTERVAL = 3600 #TIME_INTERVAL[sec]ごとに降水強度を出力できる
varname = 'PREC'

init_file = "../init/init_d01_20070714-180000.000.pe######.nc"  #使っていない
# org_file = "../init/init_d01_20070714-180000.000.pe######.org.nc"
org_file = "restart_t=10.pe######.nc"
history_file = "history_d01.pe######.nc"

orgfile = 'no-control_24h.pe######.nc'
file_path = '/home/yuta/scale-5.5.3/scale-rm/test/tutorial/real/experiment_init_BO/run'
gpyoptfile=f"BO_init_4var.pe######.nc"

# 関数評価回数
update_batch_times = 2
batch_size=5 #WSでは<=6
opt_num=batch_size*update_batch_times  

trial_num = 1#  試行回数

# 降水強度を最小化したい領域

Objective_X_low = 65
Objective_X_high = 75
Objective_Y_low = 60
Objective_Y_high = 70
Area_size=(Objective_X_high-Objective_X_low)*(Objective_Y_high-Objective_Y_low)
# 降水強度を最小化したい時刻
Objective_T = 5

# 制御する変数
Control_Var_name = "MOMZ"

# 制御対象範囲
Control_X_low = 60
Control_X_high =75#AX =90
Control_Y_low = 40
Control_Y_high = 55 #AX =90
Control_Z_low = 0 
Control_Z_high = 25#MAX =35?

# 介入領域の大きさ
Control_X_size = 5 #あんま変えられない　いくつか同時に変更する地点ありrandom_samples1 とか
Control_Y_size = 5
Control_Z_size = 5
#　介入の大きさ（負の数もあり得ます）
Control_Var_size = 10




#BOの獲得関数
Base_estimator="GP"
Acq_func="EI"

base_dir = f"../test_result_xyz/{Control_Var_name}_t={Objective_T}_{Control_X_size}*{Control_Y_size}*{Control_Z_size}grids_FET={opt_num}_trials={trial_num}_{current_time}"


def predict(inputs):
    """
    与えられた制御変数の値の配列(=input)を用いて、並列的に目的関数の値を評価し、配列の形で目的関数の値を返す。
    """
    #inputs= (batch_size,input_space_size)の構造
    
    global sub_history_file,sub_init_file

    for i in range(batch_size):
        print(f"predict_input{i}:{inputs[i]}\n")
        for pe in range(nofpe):
            sub_init_file = f"000{i}/init_d01_20070714-180000.000.pe######.nc"
            if i>=10:
                sub_init_file = f"00{i}/init_d01_20070714-180000.000.pe######.nc"
            sub_init = sub_init_file.replace('######', str(pe).zfill(6))
            init = init_file.replace('######', str(pe).zfill(6))
            subprocess.run(["cp", init, sub_init])

    
    for i in range(batch_size):
        for pe in range(nofpe):
            sub_history_file = f"000{i}/history_d01.pe######.nc"
            sub_history = sub_history_file.replace('######', str(pe).zfill(6))
            history_path = file_path+'/'+sub_history
            if (os.path.isfile(history_path)):
                subprocess.run(["rm", sub_history])
        init_val_intervation(i,inputs[i][0],inputs[i][1],inputs[i][2]) # ここで初期値を書き換える

    # 書き換えた初期値から runを回す（6時間分）
    result_mpi = subprocess.run(
        ["mpirun", "-n", str(nofpe*batch_size), "./scale-rm", "run.launch.conf"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(result_mpi.stdout.decode())
    print(result_mpi.stderr.decode())
    
    # 目的関数の値の計算
    result = [0]*batch_size # ステップごとの結果を初期化
    for i in range(batch_size):
        for pe in range(nofpe):
            sub_history_file = f"000{i}/history_d01.pe######.nc"
            fiy, fix = np.unravel_index(pe, (fny, fnx))
            nc = netCDF4.Dataset(sub_history_file.replace('######', str(pe).zfill(6)))
            nt = nc.dimensions['time'].size
            nx = nc.dimensions['x'].size
            ny = nc.dimensions['y'].size
            nz = nc.dimensions['z'].size
            gx1 = nx * fix
            gx2 = nx * (fix + 1)
            gy1 = ny * fiy
            gy2 = ny * (fiy + 1)
            if(pe==0):
                dat = np.zeros((nt, nz, fny*ny, fnx*nx))
            dat[:, 0, gy1:gy2, gx1:gx2] = nc[varname][:]
        for j in range(Objective_X_low,Objective_X_high):
            for k in range(Objective_Y_low,Objective_Y_high):
                result[i] += dat[Objective_T, 0, k, j]*TIME_INTERVAL
        
    print(f"t={Objective_T}:Ave of PREC in X[{Objective_X_low}, {Objective_X_high-1}], Y[{Objective_Y_low}, {Objective_Y_high-1}] = {result} [mm/h]\n")
    f.write(f"t={Objective_T}:Ave of PREC in X[{Objective_X_low}, {Objective_X_high-1}], Y[{Objective_Y_low}, {Objective_Y_high-1}] = {result} [mm/h]\n")
    return result

def init_val_intervation(num,Center_Control_X,Center_Control_Y,Control_Z,):
    """
    与えられた制御変数の値（Center_Control_X~4）を用いて初期値を変更する。
    """
    
    global org_file
    f.write(f"control Center_Control_X={Center_Control_X},Center_Control_Y={Center_Control_Y},Control_Z={Control_Z}\n")
    print(f"control Center_Control_X={Center_Control_X},Center_Control_Y={Center_Control_Y},Control_Z={Control_Z}\n")
    for pe in range(nofpe):
        output_file = f"000{num}/out-{Control_Var_name}.pe######.nc"
        sub_init_file = f"000{num}/init_d01_20070714-180000.000.pe######.nc"
        if num>=10:
            output_file = f"00{num}/out-{Control_Var_name}.pe######.nc"
            sub_init_file = f"00{num}/init_d01_20070714-180000.000.pe######.nc"
        sub_init = sub_init_file.replace('######', str(pe).zfill(6))
        output = output_file.replace('######', str(pe).zfill(6))

        with netCDF4.Dataset(sub_init) as src, netCDF4.Dataset(output, "w") as dst:
            dst.setncatts(src.__dict__)
            for name, dimension in src.dimensions.items():
                dst.createDimension(
                    name, (len(dimension) if not dimension.isunlimited() else None))
            for name, variable in src.variables.items():
                x = dst.createVariable(
                    name, variable.datatype, variable.dimensions)
                dst[name].setncatts(src[name].__dict__)
                if name == Control_Var_name:
                    var = src[name][:]
                    #print(var.shape) # 出力　(47, 47, 36)
                    if pe == 0:
                        if Center_Control_X<X_size/fnx and Center_Control_Y<X_size/fnx:
                            var[int(Center_Control_Y)-2:int(Center_Control_Y)+3,int(Center_Control_X)-2:int(Center_Control_X)+3 , int(Control_Z)-2 :  int(Control_Z)+3] += Control_Var_size # (y, x, z) 3:5なら3, 4のみ
                    elif pe == 1:
                        if Center_Control_X>=X_size/fnx and Center_Control_Y<X_size/fnx:
                            var[int(Center_Control_Y)-2:int(Center_Control_Y)+3, int(Center_Control_X-X_size/fnx)-2:int(Center_Control_X-X_size/fnx)+3, int(Control_Z)-2 :  int(Control_Z)+3] += Control_Var_size
                    elif pe==2:
                        if Center_Control_X<X_size/fnx and Center_Control_Y>=X_size/fnx:
                            var[int(Center_Control_Y-X_size/fnx)-2:int(Center_Control_Y-X_size/fnx)+3, int(Center_Control_X)-2:int(Center_Control_X)+3, int(Control_Z)-2 :  int(Control_Z)+3] += Control_Var_size
                    elif pe==3:
                        if Center_Control_X>=X_size/fnx and Center_Control_Y>=X_size/fnx:
                            var[int(Center_Control_Y-X_size/fnx)-2:int(Center_Control_Y-X_size/fnx)+3, int(Center_Control_X-X_size/fnx)-2:int(Center_Control_X-X_size/fnx)+3, int(Control_Z)-2 :  int(Control_Z)+3] += Control_Var_size
                    dst[name][:] = var
                else:
                    dst[name][:] = src[name][:]
        subprocess.run(["cp", output, sub_init ])
    return

def sim(Center_Control_X,Center_Control_Y,Control_Z):
    """
    得られた最適解を用いて目的関数の値を再度計算する。
    制御しない場合と制御した場合における、ある時刻のある領域の降水強度の値を返す。
    """

    # 目的の降水強度
    TEST_prec_matrix=np.zeros((Objective_X_high-Objective_X_low,Objective_Y_high-Objective_Y_low)) #ベイズ最適化の累積降水量  
    CTRL_prec_matrix=np.zeros((Objective_X_high-Objective_X_low,Objective_Y_high-Objective_Y_low)) # 制御前の累積降水量
    TEST_CTRL_prec_matrix = np.zeros((Objective_X_high-Objective_X_low,Objective_Y_high-Objective_Y_low)) #制御あり -制御なし　の各地点のある時刻の降水強度　負の値ほど、良い制御
    TEST_prec_sum=0
    CTRL_prec_sum=0

    for pe in range(nofpe):
        output_file = f"out-{Control_Var_name}.pe######.nc"
        # input file
        init = init_file.replace('######', str(pe).zfill(6))
        org = org_file.replace('######', str(pe).zfill(6))
        history = history_file.replace('######', str(pe).zfill(6))
        output = output_file.replace('######', str(pe).zfill(6))
        history_path = file_path+'/'+history
        if(os.path.isfile(history_path)):
            subprocess.run(["rm",history])
        subprocess.run(["cp", org, init]) #初期化

        with netCDF4.Dataset(init) as src, netCDF4.Dataset(output, "w") as dst:
            # copy global attributes all at once via dictionary
            dst.setncatts(src.__dict__)
            # copy dimensions
            for name, dimension in src.dimensions.items():
                dst.createDimension(
                    name, (len(dimension) if not dimension.isunlimited() else None))
            # copy all file data except for the excluded
            for name, variable in src.variables.items():
                x = dst.createVariable(
                    name, variable.datatype, variable.dimensions)
                # copy variable attributes all at once via dictionary
                dst[name].setncatts(src[name].__dict__)

                # min_Var = float('inf')
                # max_Var = -min_Var
                # if name == "QR":
                #     for i in range(47):
                #         for j in range(47):
                #             for k in range(36):
                #                 if src[name][i, j, k] > max_Var:
                #                     max_Var= src[name][i, j, k]
                #                 elif src[name][i, j, k] < min_Var:
                #                     min_Var= src[name][i, j, k]
                #     print(f"{name}:{pe=}_{min_Var=}\n")
                #     print(f"{name}:{pe=}_{max_Var=}\n")


                if name == Control_Var_name:
                    var = src[name][:]
                    if pe == 0:
                        if Center_Control_X<X_size/fnx and Center_Control_Y<X_size/fnx:
                            var[int(Center_Control_Y)-2:int(Center_Control_Y)+3,int(Center_Control_X)-2:int(Center_Control_X)+3 , int(Control_Z)-2 :  int(Control_Z)+3] += Control_Var_size
                    elif pe == 1:
                        if Center_Control_X>=X_size/fnx and Center_Control_Y<X_size/fnx:
                            var[int(Center_Control_Y)-2:int(Center_Control_Y)+3, int(Center_Control_X-X_size/fnx)-2:int(Center_Control_X-X_size/fnx)+3, int(Control_Z)-2 :  int(Control_Z)+3] += Control_Var_size
                    elif pe==2:
                        if Center_Control_X<X_size/fnx and Center_Control_Y>=X_size/fnx:
                            var[int(Center_Control_Y-X_size/fnx)-2:int(Center_Control_Y-X_size/fnx)+3, int(Center_Control_X)-2:int(Center_Control_X)+3, int(Control_Z)-2 :  int(Control_Z)+3] += Control_Var_size
                    elif pe==3:
                        if Center_Control_X>=X_size/fnx and Center_Control_Y>=X_size/fnx:
                            var[int(Center_Control_Y-X_size/fnx)-2:int(Center_Control_Y-X_size/fnx)+3, int(Center_Control_X-X_size/fnx)-2:int(Center_Control_X-X_size/fnx)+3, int(Control_Z)-2 :  int(Control_Z)+3] += Control_Var_size
                  
                    dst[name][:] = var
                else:
                    # dst[name][:] = hfi[name][:] #(y,x,z)=(time,z,y,x)
                    dst[name][:] = src[name][:]
        subprocess.run(["cp", output, init])

    subprocess.run(["mpirun", "-n", str(nofpe), "./scale-rm","run.d01.conf"])

    # run　の結果から、目的関数の計算のために必要なデータを取ってくる。
    # time.sleep(0.3)
    for pe in range(nofpe):
        gpyopt = gpyoptfile.replace('######', str(pe).zfill(6))
        history = history_file.replace('######', str(pe).zfill(6))
        subprocess.run(["cp", history,gpyopt])
    for pe in range(nofpe):  # history処理
        fiy, fix = np.unravel_index(pe, (fny, fnx))
        nc = netCDF4.Dataset(history_file.replace('######', str(pe).zfill(6)))
        onc = netCDF4.Dataset(orgfile.replace('######', str(pe).zfill(6)))
        nt = nc.dimensions['time'].size
        nx = nc.dimensions['x'].size
        ny = nc.dimensions['y'].size
        nz = nc.dimensions['z'].size
        gx1 = nx * fix
        gx2 = nx * (fix + 1)
        gy1 = ny * fiy
        gy2 = ny * (fiy + 1)
        # print(f"onc[varname][:] = {onc[varname][:]}")
        if pe == 0:
            dat = np.zeros((nt, nz, fny*ny, fnx*nx))
            odat = np.zeros((nt, nz, fny*ny, fnx*nx))
        dat[:, 0, gy1:gy2, gx1:gx2] = nc[varname][:]
        odat[:, 0, gy1:gy2, gx1:gx2] = onc[varname][:]

    # データから目的関数の値を計算する
    # 目的関数に該当する領域以外もPRECは計算しない
    for j in range(Objective_X_low,Objective_X_high):
        for k in range(Objective_Y_low,Objective_Y_high):
            TEST_prec_matrix[j-Objective_X_low,k-Objective_Y_low] += dat[Objective_T,0,k,j]*TIME_INTERVAL #(y,x,z)=(time,z,y,x)　j-Objective_X_low,k-Objective_Y_lowは[0,0]->[5,5]とか
            CTRL_prec_matrix[j-Objective_X_low,k-Objective_Y_low] += odat[Objective_T,0,k,j]*TIME_INTERVAL
            TEST_CTRL_prec_matrix[j-Objective_X_low,k-Objective_Y_low] = TEST_prec_matrix[j-Objective_X_low,k-Objective_Y_low] - CTRL_prec_matrix[j-Objective_X_low,k-Objective_Y_low]
            TEST_prec_sum+=TEST_prec_matrix[j-Objective_X_low,k-Objective_Y_low]
            CTRL_prec_sum+=CTRL_prec_matrix[j-Objective_X_low,k-Objective_Y_low]
    return TEST_prec_sum, CTRL_prec_sum, TEST_prec_matrix, CTRL_prec_matrix ,TEST_CTRL_prec_matrix

def plot_PREC(trial_i, TEST_CTRL_prec_matrix):
    """
    得られた最適解（制御入力）でシミュレーションをした結果を、"可視化する"関数。
    目的時刻t=Objective_T の降水強度を可視化する
    """
    outtype = 'fcst'                    # outtype in convert_letkfout.py
    ncsuf = 'pe______.nc'

    # params for plot
    cmap = plt.cm.jet
    cmap.set_under('lightgray')
    m_levels = [0,10,20,30,40]
    s_levels = [50,100,150,200] 
    plt_extent = [126,144,26,44]
    history_name = 'history_d01'
    ncfilebasetmp = '{:s}.{:}'.format(history_name, ncsuf)

    # get lon lat
    slon = gio.readvar2d('lon',ncfilebasetmp,outtype,fnx,fny)
    slat = gio.readvar2d('lat',ncfilebasetmp,outtype,fnx,fny)
    try:
        var = gio.readvar2d('PREC',ncfilebasetmp,outtype,fnx,fny)
    except ZeroDivisionError:
        print("ゼロ除算が発生しました。デフォルト値を使用します。")
        var = 0  # または適切なデフォルト値
    except ValueError as e:
        print(f"値エラーが発生しました: {e}")
        var = 0  # または適切なデフォルト値
    except Exception as e:
        print(f"予期しないエラーが発生しました: {e}")
        var = 0  # または適切なデフォルト値

    # unit conversion (kg/m/s --> mm/d)
    var *=  60 * 60
    print(f"{len(var)=}")
    for t in range(len(var)):
        # plot
        fig = plt.figure()

        # scale
        ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
        ax.set_extent(plt_extent)
        ax.coastlines(resolution='50m',linewidth=0.5)

        # grid line
        gl = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=0.5,linestyle='--',color='gray')
        gl.xlocator = mticker.FixedLocator(range(126,146))
        gl.ylocator = mticker.FixedLocator(range(26,44))
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size':6}
        gl.ylabel_style = {'size':6}

        im = ax.contourf(slon,slat,var[t],cmap=cmap,levels=m_levels, extend='both',transform=ccrs.PlateCarree())

        pos = ax.get_position()
        fig.subplots_adjust(right=0.85)
        cax = fig.add_axes([pos.x1,pos.y0,0.02,pos.y1-pos.y0])
        clb = fig.colorbar(im, orientation='vertical', cax=cax)

        filename = os.path.join(base_dir, "PREC-heatmap", f"LonLat_t={t}_seed={trial_i}.png")
        plt.savefig(filename ,dpi=300)
        plt.close()

    # ヒートマップの作成 緯度経度でなくグリッド
    ## 制御後の降水強度
    vmin = 5  # 最小値
    vmax = 35  # 最大値
    plt.figure(figsize=(10, 8))
    plt.grid(True)
    plt.imshow(TEST_prec_matrix.T, cmap='viridis', aspect='auto',origin='lower',vmin = vmin, vmax=vmax) #       カラーマップは好みに応じて変更可能
    plt.colorbar(label="precipitation (mm/h)")  # カラーバーのラベル
    plt.title(f"Time={Objective_T}h_X={Objective_X_low}-{Objective_X_high-1}_Y={Objective_Y_low}-{Objective_Y_high-1} ")
    plt.xlabel("X")
    plt.ylabel("Y")
    filename = os.path.join(base_dir, "PREC-heatmap", f"Grid_seed={trial_i}.png")
    plt.savefig(filename ,dpi=300)

    ## 制御後- 制御前の降水強度
    plt.figure(figsize=(10, 8))
    plt.grid(True)
    plt.imshow(TEST_CTRL_prec_matrix.T, cmap='viridis', aspect='auto',origin='lower')  # カラーマップは好みに応じて変更可能
    plt.colorbar(label="precipitation (mm/h)")  # カラーバーのラベル
    plt.title(f"Time={Objective_T}h_X={Objective_X_low}-{Objective_X_high-1}_Y={Objective_Y_low}-{Objective_Y_high-1} ")
    plt.xlabel("X")
    plt.ylabel("Y")
    filename = os.path.join(base_dir, "PREC-heatmap", f"Grid_diff_C-noC_seed={trial_i}.png")
    plt.savefig(filename ,dpi=300)
    return

def make_directory(base_dir):
    os.makedirs(base_dir, exist_ok=False)
    # 階層構造を作成
    sub_dirs = ["PREC-heatmap", "summary"]
    for sub_dir in sub_dirs:
        path = os.path.join(base_dir, sub_dir)
        os.makedirs(path, exist_ok=True) 
    return


### 実行
make_directory(base_dir)
filename = f"config.txt"
config_file_path = os.path.join(base_dir, filename)  # 修正ポイント

# ファイルに書き込む
with open(config_file_path,  "w") as f:
    # 関数評価回数
    f.write(f"update_batch_times = {update_batch_times}\n")
    f.write(f"batch_size = {batch_size}\n")
    f.write(f"opt_num = {opt_num}  # WSでは<=24\n")
    
    # 試行回数
    f.write(f"\ntrial_num = {trial_num}  # 試行回数\n")
    
    # 降水強度を最小化したい領域
    f.write("\n# 降水強度を最小化したい領域\n")
    f.write(f"Objective_X_low = {Objective_X_low}\n")
    f.write(f"Objective_X_high = {Objective_X_high}\n")
    f.write(f"Objective_Y_low = {Objective_Y_low}\n")
    f.write(f"Objective_Y_high = {Objective_Y_high}\n")
    f.write(f"Area_size = {Area_size}\n")
    
    # 降水強度を最小化したい時刻
    f.write("\n# 降水強度を最小化したい時刻\n")
    f.write(f"Objective_T = {Objective_T}\n")
    
    # 制御する変数
    f.write("\n# 制御する変数\n")
    f.write(f"Control_Var_name= \"{Control_Var_name}\"\n")
    
    # 制御対象範囲
    f.write("\n# 制御対象範囲\n")
    f.write(f"Control_X_low = {Control_X_low}\n")
    f.write(f"Control_X_high = {Control_X_high}\n")
    f.write(f"Control_Y_low = {Control_Y_low}\n")
    f.write(f"Control_Y_high = {Control_Y_high}  # MAX =90\n")
    f.write(f"Control_Z_low = {Control_Z_low}\n")
    f.write(f"Control_Z_high = {Control_Z_high}  # MAX =35?\n")
    
    # 介入領域の大きさ
    f.write("\n# 介入領域の大きさ\n")
    f.write(f"Control_X_size = {Control_X_size}\n")
    f.write(f"Control_Y_size = {Control_Y_size}\n")
    f.write(f"Control_Z_size = {Control_Z_size}\n")
    # 介入幅
    f.write("\n# 介入の大きさ\n")
    f.write(f"Control_Var_size = {Control_Var_size}  \n")
    
    # BOの獲得関数
    f.write("\n# BOの獲得関数\n")
    f.write(f"Base_estimator = \"{Base_estimator}\"\n")
    f.write(f"Acq_func = \"{Acq_func}\"\n")

log_file = os.path.join(base_dir, "summary", f"BO_log.txt")
summary_file = os.path.join(base_dir, "summary", f"BO_summary.txt")

# 入力次元と最小値・最大値の定義
bounds = [ Integer(Control_X_low+2, Control_X_high-3),Integer(Control_Y_low+2, Control_Y_high-3) ,Integer(Control_Z_low+2   , Control_Z_high-3  )] # Int(2, 4) なら2, 3, 4からランダム

with open(log_file, 'w') as f, open(summary_file, 'w') as f_s:
    for trial_i in range(trial_num):
        f.write(f"{trial_i=}\n")
        random.seed(trial_i)

        start = time.time()  # 現在時刻（処理開始前）を取得
        opt = Optimizer(bounds, base_estimator=Base_estimator, acq_func=Acq_func, random_state=trial_i)

        for pe in range(nofpe):
            org = org_file.replace('######', str(pe).zfill(6))
            init = init_file.replace('######', str(pe).zfill(6))
            subprocess.run(["cp", org, init])

        best_values = []
        current_best = float('inf')

        # 各範囲でランダム値を生成
        random_samples1 = [[random.randint(Control_X_low+2, Control_X_high-3),] for _ in range(batch_size)]
        random_samples2 = [[random.randint(Control_Y_low+2, Control_Y_high-3) ] for _ in range(batch_size)]
        random_samples3 = [[random.randint(Control_Z_low+2, Control_Z_high-3) ] for _ in range(batch_size)]

        # リストを結合してサンプルを作成
        combined_samples = [sample1 + sample2 + sample3  for sample1, sample2, sample3 in zip(random_samples1, random_samples2, random_samples3)]

# シミュレーション1回だけやりたいときはここから
        Y = predict(combined_samples)
        opt.tell(combined_samples,Y)

        for j in range(update_batch_times-1):
            # アクイジション関数を用いて次の探索点を決定
            next_points = opt.ask(n_points=batch_size)

            # 並列で評価関数を計算
            values = predict(next_points)
            print(f"values{values}")

            # 評価結果をモデルに反映
            opt.tell(next_points, values)
            print(f"Batch {j+1}: Best value so far: {min(opt.yi)}\n")
            f.write(f"Batch {j+1}: Best value so far: {min(opt.yi)}\n\n")

        # 結果の取得
        best_value = min(opt.yi)
        # 最小値のインデックスを取得
        min_index = opt.yi.index(min(opt.yi))

        # 対応するベストポイントを取得
        best_point = opt.Xi[min_index]
        # best_point = opt.Xi[np.argmin(opt.yi)]
        print(f"Best value: {best_value} at point {best_point}")

        optimal_inputs = best_point
# ここまでコメントアウトして下のoptimalで探索地点手入力
        # optimal_inputs=[70, 50, 20, 0]

        for pe in range(nofpe):
            org = org_file.replace('######', str(pe).zfill(6))
            init = init_file.replace('######', str(pe).zfill(6))
            subprocess.run(["cp", org, init])

        TEST_prec_sum, CTRL_prec_sum, TEST_prec_matrix, CTRL_prec_matrix , TEST_CTRL_prec_matrix=sim(optimal_inputs[0],optimal_inputs[1],optimal_inputs[2])#  change

        end=time.time()
        time_diff = end - start
        print(f'\n\n実行時間:{time_diff}\n')


        # 結果のクイック描画
        subprocess.run(["mpirun", "-n", str(nofpe), "./sno", "sno.vgridope.d01.conf"])
        subprocess.run(["mpirun", "-n", "1","./sno", "sno.hgridope.d01.conf"])
        subprocess.run(["grads", "-blc", "checkfig_real.gs"])

        # シミュレーション結果
        print(f"CTRL_prec_matrix={CTRL_prec_matrix}")
        print(f"TEST_prec_matrix={TEST_prec_matrix}")

        print(f"{TEST_prec_sum=}")
        print(f"{CTRL_prec_sum=}")
        print(f"%={TEST_prec_sum/CTRL_prec_sum*100}%")
        f_s.write(f'{trial_i=}')

        f_s.write(f'実行時間:{time_diff}\n')
        f_s.write(f'{opt_num}回の評価で得られた最適解：{optimal_inputs}\n')
        f_s.write(f"CTRL_prec_matrix={CTRL_prec_matrix}\n")
        f_s.write(f"TEST_prec_matrix={TEST_prec_matrix}\n")

        f_s.write(f"{TEST_prec_sum=}")
        f_s.write(f"{CTRL_prec_sum=}")
        f_s.write(f"%={TEST_prec_sum/CTRL_prec_sum*100}%\n\n")
        plot_PREC(trial_i, TEST_CTRL_prec_matrix)






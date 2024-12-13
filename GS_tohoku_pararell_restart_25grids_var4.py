import os
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import subprocess

import cartopy.crs as ccrs
import gradio as gio
import matplotlib.ticker as mticker
import imageio  # GIF作成用
from datetime import datetime

# 時刻を計測するライブラリ
import time
from skopt import gp_minimize
from skopt.space import Real,Integer
import warnings

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


nofpe = 4
fny = 2
fnx = 2


varname = 'PREC'

init_file = "../init/init_d01_20070714-180000.000.pe######.nc"  #使っていない
# org_file = "../init/init_d01_20070714-180000.000.pe######.org.nc"
org_file = "restart_t=10.pe######.nc"
history_file = "history_d01.pe######.nc"

orgfile = 'no-control_24h.pe######.nc'
file_path = '/home/yuta/scale-5.5.3/scale-rm/test/tutorial/real/experiment_init_BO/run'
gpyoptfile=f"GS_init_4var.pe######.nc"

sum_gpy=np.zeros((90,90)) #ベイズ最適化の累積降水量
sum_no=np.zeros((90,90)) #制御前の累積降水量
sum_prec=np.zeros((90,90))

cnt=0
opt_num=10

n=5


def predict(inputs):
    #inputs= (5,4)の構造
    
    print(f"predict_input1{inputs[0]},input2{inputs[1]},input3{inputs[2]},input4{inputs[3]},input5{inputs[4]}") 
    
    global sub_history_file,sub_init_file
    

    for i in range(n):
        for pe in range(nofpe):
            sub_init_file = f"000{i}/init_d01_20070714-180000.000.pe######.nc"
            if i>=10:
                sub_init_file = f"00{i}/init_d01_20070714-180000.000.pe######.nc"
            sub_init = sub_init_file.replace('######', str(pe).zfill(6))
            init = init_file.replace('######', str(pe).zfill(6))
            subprocess.run(["cp", init, sub_init])

    
    for i in range(n):
        for pe in range(nofpe):
            sub_history_file = f"000{i}/history_d01.pe######.nc"
            if i>=10:
                sub_init_file = f"00{i}/init_d01_20070714-180000.000.pe######.nc"
            sub_history = sub_history_file.replace('######', str(pe).zfill(6))
            history_path = file_path+'/'+sub_history
            if (os.path.isfile(history_path)):
                subprocess.run(["rm", sub_history])
                # subprocess.run(["rm", history_path])
        control(i,inputs[i][0],inputs[i][1],inputs[i][2],inputs[i][3])

    result = subprocess.run(
        ["mpirun", "-n", "20", "./scale-rm", "run.launch.conf"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # time.sleep(0.1)
    print(result.stdout.decode())
    print(result.stderr.decode())
    
    result = [0]*n # ステップごとの結果を初期化
    for i in range(n):
        for pe in range(nofpe):
            sub_history_file = f"000{i}/history_d01.pe######.nc"
            if i>=10:
                sub_history_file = f"00{i}/history_d01.pe######.nc"
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
        for j in range(70,80):
            for k in range(60,70):
                result[i] += dat[5, 0, k, j]*3600
        
    print(f"Result {result}")
    # print(f"predict,loop={loop}")
    
    return result

def control(num,input1,input2,input3,input4):
    
    global org_file
    print(f"control input1={input1},input2={input2},input3={input3},input4={input4}\n")
    for pe in range(nofpe):
        output_file = f"000{num}/out-MOMY.pe######.nc"
        sub_init_file = f"000{num}/init_d01_20070714-180000.000.pe######.nc"
        if num>=10:
            output_file = f"00{num}/out-MOMY.pe######.nc"
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
                if name == 'MOMY':
                    var = src[name][:]
                    if pe == 0:
                        if input1<45 and input2<45:
                            var[int(input2), int(input1), int(input3)] += input4
                    elif pe == 1:
                        if input1>=45 and input2<45:
                            var[int(input2), int(input1-45), int(input3)] += input4
                    elif pe==2:
                        if input1<45 and input2>=45:
                            var[int(input2-45), int(input1), int(input3)] += input4
                    elif pe==3:
                        if input1>=45 and input2>=45:
                            var[int(input2-45), int(input1-45), int(input3)] += input4
                            
                    dst[name][:] = var
                else:
                    dst[name][:] = src[name][:]
        subprocess.run(["cp", output, sub_init ])
    return


def update_control(input1,input2,input3,input4):
    
    global org_file
    print(f"control input1={input1},input2={input2},input3={input3}")
    for pe in range(nofpe):
        output_file = f"out-MOMY.pe######.nc"
        init = init_file.replace('######', str(pe).zfill(6))
        output = output_file.replace('######', str(pe).zfill(6))
        with netCDF4.Dataset(init) as src, netCDF4.Dataset(output, "w") as dst:
            dst.setncatts(src.__dict__)
            for name, dimension in src.dimensions.items():
                dst.createDimension(
                    name, (len(dimension) if not dimension.isunlimited() else None))
            for name, variable in src.variables.items():
                x = dst.createVariable(
                    name, variable.datatype, variable.dimensions)
                dst[name].setncatts(src[name].__dict__)
                if name == 'MOMY':
                    var = src[name][:]
                    if pe == 0:
                        if input1<45 and input2<45:
                            var[int(input2)-2:int(input2)+3,int(input1)-2:int(input1)+3 , int(input3)] += input4
                    elif pe == 1:
                        if input1>=45 and input2<45:
                            var[int(input2)-2:int(input2)+3, int(input1-45)-2:int(input1-45)+3, int(input3)] += input4
                    elif pe==2:
                        if input1<45 and input2>=45:
                            var[int(input2-45)-2:int(input2-45)+3, int(input1)-2:int(input1)+3, int(input3)] += input4
                    elif pe==3:
                        if input1>=45 and input2>=45:
                            var[int(input2-45)-2:int(input2-45)+3, int(input1-45)-2:int(input1-45)+3, int(input3)] += input4
                            
                    dst[name][:] = var
                else:
                    dst[name][:] = src[name][:]
        subprocess.run(["cp", output, init])
    return

def f(inputs):
    print(f"f_inputs{inputs}\n")
    
    # control(inputs)
    cost_sum = predict(inputs)
    print(f"Cost at input {inputs}: Cost_sum {cost_sum}\n")
    
    return cost_sum #n個の配列　jobごとのコスト関数が入ってる


def sim(input1,input2,input3,input4):
    sum=0
    no=0
    global sum_gpy,sum_no
    for pe in range(nofpe):
        output_file = f"out-MOMY.pe######.nc"
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
                if name == 'MOMY':
                    var = src[name][:]
                    if pe == 0:
                        if input1<45 and input2<45:
                            var[int(input2)-2:int(input2)+3,int(input1)-2:int(input1)+3 , int(input3)] += input4
                    elif pe == 1:
                        if input1>=45 and input2<45:
                            var[int(input2)-2:int(input2)+3, int(input1-45)-2:int(input1-45)+3, int(input3)] += input4
                    elif pe==2:
                        if input1<45 and input2>=45:
                            var[int(input2-45)-2:int(input2-45)+3, int(input1)-2:int(input1)+3, int(input3)] += input4
                    elif pe==3:
                        if input1>=45 and input2>=45:
                            var[int(input2-45)-2:int(input2-45)+3, int(input1-45)-2:int(input1-45)+3, int(input3)] += input4
                  
                    dst[name][:] = var
                else:
                    # dst[name][:] = hfi[name][:] #(y,x,z)=(time,z,y,x)
                    dst[name][:] = src[name][:]
        subprocess.run(["cp", output, init])

    subprocess.run(["mpirun", "-n", "4", "./scale-rm","run.d01.conf"])

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
    # for i in range(nt):
    for j in range(70,80):
        for k in range(60,70):
            sum_gpy[j,k]+=dat[5,0,k,j]*3600 #(y,x,z)=(time,z,y,x)
            sum_no[j,k]+=odat[5,0,k,j]*3600
            sum+=sum_gpy[j,k]
            no+=sum_no[j,k]
    return sum,no


start = time.time()  # 現在時刻（処理開始前）を取得

# 入力次元と最小値・最大値の定義
X_low = 70
X_high = 80
Y_low = 25
Y_high = 35
bounds = [ Integer(X_low , X_high ),Integer(Y_low, Y_high)]

for pe in range(nofpe):
    org = org_file.replace('######', str(pe).zfill(6))
    init = init_file.replace('######', str(pe).zfill(6))
    subprocess.run(["cp", org, init])

batch_size = n

best_values = []
current_best = float('inf')



# リストを結合してサンプルを作成
combined_samples = [sample1 + sample2 + sample3 + sample4 for sample1, sample2, sample3, sample4 in zip(random_samples1, random_samples2, random_samples3, random_samples4)]

X = combined_samples

Y = f(X)

opt.tell(X,Y)

for Y_i in range(Y_low, Y_high):
    for X_i in range(X_low, X_high):
    # アクイジション関数を用いて次の探索点を決定
    next_points = opt.ask(n_points=batch_size)

    # 並列で評価関数を計算
    values = f(
    print(f"values{values}")

    # 評価結果をモデルに反映
    opt.tell(next_points, values)
    print(f"Batch {j+1}: Best value so far: {min(opt.yi)}\n")

# 結果の取得
best_value = min(opt.yi)
# 最小値のインデックスを取得
min_index = opt.yi.index(min(opt.yi))

# 対応するベストポイントを取得
best_point = opt.Xi[min_index]
# best_point = opt.Xi[np.argmin(opt.yi)]
print(f"Best value: {best_value} at point {best_point}")

optimal_inputs=best_point

# update_control(optimal_inputs[0],optimal_inputs[1],optimal_inputs[2],optimal_inputs[3])
for pe in range(nofpe):
    org = org_file.replace('######', str(pe).zfill(6))
    init = init_file.replace('######', str(pe).zfill(6))
    subprocess.run(["cp", org, init])


no=0
gpy=0
gpy,no=sim(optimal_inputs[0],optimal_inputs[1],optimal_inputs[2],optimal_inputs[3])

end=time.time()
time_diff = end - start
print(f'実行時間{time_diff}')

print(f"sum_gpy={sum_gpy}")





end = time.time()  # 現在時刻（処理完了後）を取得
time_diff = end - start
print(f'実行時間{time_diff}')

# print(f'入力:x={result.x[0]},y={result.x[1]},z={result.x[2]},input={result.x[3]}')
print(f"sum_no={sum_no}")
print(f"sum_gpy={sum_gpy}")


print(f"BO={gpy}")
print(f"no-controk={no}")
print(f"change%={(no-gpy)/no*100}%")
print(f"%={(gpy)/no*100}%")


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
var = gio.readvar2d('PREC',ncfilebasetmp,outtype,fnx,fny)

# unit conversion (kg/m/s --> mm/d)
var *=  60 * 60


# plot
# base
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

im = ax.contourf(slon,slat,var[5],cmap=cmap,levels=m_levels, extend='both',transform=ccrs.PlateCarree())

pos = ax.get_position()
fig.subplots_adjust(right=0.85)
cax = fig.add_axes([pos.x1,pos.y0,0.02,pos.y1-pos.y0])
clb = fig.colorbar(im, orientation='vertical', cax=cax)

plt.savefig(f'result/GS_tohoku_restart_25grids_PREC_t=5.png',dpi=300)
plt.close()

# ヒートマップの作成 緯度経度でなくグリッド
plt.figure(figsize=(10, 8))
plt.grid(True)
plt.imshow(sum_gpy.T, cmap='viridis', aspect='auto',origin='lower')  # カラーマップは好みに応じて変更可能
plt.colorbar(label="precipitation (mm/h)")  # カラーバーのラベル
plt.title("precipitation")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig(f"result/PREC-visualize.png")


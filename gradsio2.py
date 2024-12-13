import os
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import datetime

def grread_all(ctlfile,grdfile,varname,tstr,tend) :

    nt = tend - tstr 
    #nt = tend - tstr + 1

    for t,tstep in enumerate(range(tstr,tend)) :
    #for t,tstep in enumerate(range(tstr,tend+1)) :
        if t == 0 :
            tmpdat = grread(ctlfile,grdfile,varname)
            if len(np.shape(tmpdat)) == 3 :
                nz,ny,nx = np.shape(tmpdat)
                dat = np.empty((nt,nz,ny,nx))
            elif len(np.shape(tmpdat)) == 2 :
                ny,nx = np.shape(tmpdat)
                dat = np.empty((nt,ny,nx))
            dat[t] = tmpdat
        else :
            dat[t] = grread(ctlfile,grdfile,varname,tstep=t)

    return dat


# read GrADS file
def grread(ctlfile,grdfile,varname,tstep=0) :

    # check
    if not os.path.exists(grdfile) or not os.path:
        print('XX data file not found :', grdfile)
        exit()
    if not os.path.exists(ctlfile) or not os.path:
        print('XX ctl file not found :', ctlfile)
        exit()

    # read ctl
    with open(ctlfile) as f :

        ctl = f.read().split()

        for i,word in enumerate(ctl) :

            # zdef
            if word.lower() == 'zdef' :
                nz = int(ctl[i+1])
                if ctl[i+2].lower() == 'linear' :
                    slev = float(ctl[i+3])
                    dlev = float(ctl[i+4])
                    elev = slev + dlev * nz
                    lev = np.arange(slev,elev,dlev)
                elif ctl[i+2].lower() == 'levels' :
                    lev = np.empty(nz,dtype=float)
                    for ii in np.arange(i+3,i+3+nz) :
                        lev[ii-(i+3)] = float(ctl[ii])
        
            if word.lower() == 'pdef' :
                nx = int(ctl[i+1])
                ny = int(ctl[i+2])

            # vars
            if word.lower() == 'vars' :
                nvars = int(ctl[i+1])


    # read ctl for vars
    varline = 999999
    count = 0
    var2d_mylist = []
    var3d_mylist = []
    dstart = {}
    with open(ctlfile) as f :

        lines = f.read().splitlines()

        for i,line in enumerate(lines) :
            if line == '' :
                continue
            description = ''
            words = line.split()
            if words[0].lower() == 'vars' :
                varline = i
                continue
            if i > varline and i <= varline + nvars :
                varnametmp = words[0].lower()
                if int(words[1]) > 1 :
                    var3d_mylist.append(varnametmp)
                    dstart[varnametmp] = count
                    count += 4 * nx * ny * nz
                else :
                    var2d_mylist.append(varnametmp)
                    dstart[varnametmp] = count
                    count += 4 * nx * ny 

    if not varname in var3d_mylist and not varname in var2d_mylist :
        print('XX varname '+varname+' not found in ctl file :',ctlfile)
        exit()

    # read data
    with open(grdfile,'rb') as f :
        if tstep == 0 :
            f.seek( dstart[varname] ) 
            if varname in var3d_mylist :
                dat = np.fromfile( f, '<f', nx*ny*nz ).reshape( nz,ny,nx )
            elif varname in var2d_mylist :
                dat = np.fromfile( f, '<f', ny*nx ).reshape( ny,nx)
        else :
            f.seek( dstart[varname] + count * tstep )
            if varname in var3d_mylist :
                dat = np.fromfile( f, '<f', nx*ny*nz ).reshape( nz,ny,nx )
            elif varname in var2d_mylist :
                dat = np.fromfile( f, '<f', ny*nx ).reshape( ny,nx)
    return dat
    

# read netCDF file
def readvar3d(varname) :

    if outtype == 'anal' or outtype == 'gues' :
        bufsize = 2
    else :
        bufsize = 0

    for pe in np.arange(nofpe) :
        # buffer
        if pe % fnx == 0 :
            wbuf, ebuf = bufsize, 0
        elif pe % fnx == fnx - 1 :
            wbuf, ebuf = 0, bufsize
        else :
            wbuf, ebuf = 0, 0
        if pe // fnx == 0 :
            sbuf, nbuf = bufsize, 0
        elif pe // fnx == fny - 1 :
            sbuf, nbuf = 0, bufsize
        else :
            sbuf, nbuf = 0, 0
        # read
        fiy,fix = np.unravel_index(pe,(fny,fnx))
        nc = netCDF4.Dataset(filebaseX.replace('<PE>',str(pe).zfill(6)))
        nx = nc.dimensions['x'].size
        ny = nc.dimensions['y'].size
        nz = nc.dimensions['z'].size
        lnx = nx - wbuf - ebuf
        lny = ny - sbuf - nbuf
        if not 'dat' in locals() :
            dat = np.empty((fny*lny,fnx*lnx,nz))
        # global index of x & y
        gx1 = lnx * fix
        gx2 = lnx * (fix + 1)
        gy1 = lny * fiy
        gy2 = lny * (fiy + 1)
        # read
        dat[gy1:gy2,gx1:gx2,:] = nc[varname][sbuf:ny-nbuf,wbuf:nx-ebuf,:]

    return dat

def readvar2d(varname,ncfile,outtype,fnx,fny) :

    if outtype == 'anal' or outtype == 'gues' :
        bufsize = 2
    else :
        bufsize = 0

    nofpe = fnx * fny

    # try
    nc = netCDF4.Dataset(ncfile.replace('______','0'.zfill(6)))
    nx = nc.dimensions['x'].size
    ny = nc.dimensions['y'].size
    nz = nc.dimensions['z'].size
    nt = nc.dimensions['time'].size
    ndim = nc.variables[varname].ndim
    dims = nc.variables[varname].dimensions

    # working axis = []
    # working for dd,dim in enumerate(dims) :
    # working     if dim == 'time' :
    # working         axis[dd] = 
    # working     elif dim == 'z' :
    # working     elif dim == 'y' :
    # working     elif dim == 'x' :

    # read
    #--- assum z
    if ndim == 1 :
        dat= nc[varname][:]
    #--- assume (y,x), (t,y,x), or (t,z,y,x)
    else :
        for pe in np.arange(nofpe) :
            # buffer
            if pe % fnx == 0 :
                wbuf, ebuf = bufsize, 0
            elif pe % fnx == fnx - 1 :
                wbuf, ebuf = 0, bufsize
            else :
                wbuf, ebuf = 0, 0
            if pe // fnx == 0 :
                sbuf, nbuf = bufsize, 0
            elif pe // fnx == fny - 1 :
                sbuf, nbuf = 0, bufsize
            else :
                sbuf, nbuf = 0, 0
            # read
            fiy,fix = np.unravel_index(pe,(fny,fnx))
            nc = netCDF4.Dataset(ncfile.replace('______',str(pe).zfill(6)))
            lnx = nx - wbuf - ebuf
            lny = ny - sbuf - nbuf
            # global index of x & y
            gx1 = lnx * fix
            gx2 = lnx * (fix + 1)
            gy1 = lny * fiy
            gy2 = lny * (fiy + 1)
            # read
            if nc.variables[varname].ndim == 2 :
                if not 'dat' in locals() :
                    dat = np.empty((fny*lny,fnx*lnx))
                dat[gy1:gy2,gx1:gx2] = nc[varname][sbuf:ny-nbuf,wbuf:nx-ebuf]
            elif nc.variables[varname].ndim == 3 :
                if not 'dat' in locals() :
                    dat = np.empty((nt,fny*lny,fnx*lnx))
                dat[:,gy1:gy2,gx1:gx2] = nc[varname][:,sbuf:ny-nbuf,wbuf:nx-ebuf]
            elif nc.variables[varname].ndim == 4 :
                if not 'dat' in locals() :
                    dat = np.empty((nt,nz,fny*lny,fnx*lnx))
                dat[:,:,gy1:gy2,gx1:gx2] = nc[varname][:,:,sbuf:ny-nbuf,wbuf:nx-ebuf]

    return dat

def readvar_sno(varname,ncfile,outtype,ix=-1,iz=-1) :

    # try
    nc = netCDF4.Dataset(ncfile.replace('______','0'.zfill(6)))
    nx = nc.dimensions['x'].size
    ny = nc.dimensions['y'].size
    nz = nc.dimensions['z'].size
    nt = nc.dimensions['time'].size
    ndim = nc.variables[varname].ndim
    dims = nc.variables[varname].dimensions

    # read
    #--- assume z
    if ndim == 1 :
        dat= nc[varname][:]
    #--- assume (y,x), (t,y,x), or (t,z,y,x)
    else :
        # read
        nc = netCDF4.Dataset(ncfile.replace('______',str(0).zfill(6)))
        if nc.variables[varname].ndim == 2 :
            if not 'dat' in locals() :
                dat = np.empty((ny,nx),dtype='float32')
            dat = nc[varname][:]
        elif nc.variables[varname].ndim == 3 :
            if not 'dat' in locals() :
                dat = np.empty((nt,ny,nx),dtype='float32')
            dat = nc[varname][:,:,:]
        elif nc.variables[varname].ndim == 4 :
            if iz >= 0 :
                if not 'dat' in locals() :
                    dat = np.empty((nt,ny,nx),dtype='float32')
                dat = nc[varname][:,iz,:,:]
            elif ix >=0 :
                if not 'dat' in locals() :
                    dat = np.empty((nt,nz,ny),dtype='float32')
                dat = nc[varname][:,:,:,ix]
            else :
                if not 'dat' in locals() :
                    dat = np.empty((nt,nz,ny,nx),dtype='float32')
                dat = nc[varname][:,:,:,:]

    return dat

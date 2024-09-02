'''
Bin the dataset by VPD (and EF) and save in process4_output
Including:
    def bin_VPD
    def bin_VPD_EF
    def write_var_VPD
    def write_var_VPD_EF
'''

__author__  = "Mengyuan Mu"
__version__ = "1.0 (05.01.2024)"
__email__   = "mu.mengyuan815@gmail.com"

#==============================================

import os
import gc
import sys
import glob
import copy
import numpy as np
import pandas as pd
import netCDF4 as nc
import multiprocessing as mp
from datetime import datetime, timedelta
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib.ticker as mticker
from PLUMBER2_VPD_common_utils import *

def find_longest_consecutive(input):

    current_start = 0
    current_len   = 0
    max_start     = 0
    max_len       = 0

    for i in np.arange(len(input)):
        if input[i] == 1:
            # if at current time step, Tair > 5
            current_len += 1
            if current_start == 0:
                current_start = i 
        else:
            # if at current time step, Tair < 5
            if current_len > max_len:
                max_len       = current_len
                max_start     = current_start
                current_start = 0
                current_len   = 0

    print('max_len',max_len,'max_start',max_start)

    return max_start, max_start + max_len - 1

def calc_growth_season(nc_path, var_name, site_name, smooth_window=7):

    # Read data 
    output_file = f'{nc_path}/{site_name}_ERA5_land_{var_name}.nc'
    f           = nc.Dataset(output_file, 'r+', format='NETCDF4')

    var         = f.variables['2t'][:]
    lat         = f.variables['lat'][:]

    time        = nc.num2date(f.variables['time'][:],f.variables['time'].units,
                    only_use_cftime_datetimes=False, only_use_python_datetimes=True)

    year_s      = time[0].year
    year_e      = time[-1].year
    
    nhours      = len(var)

    # Calculate total days
    if nhours%24 == 0:
        day_tot   = int(nhours/24)    
        var_daily = np.zeros(day_tot)
        # Calculate daily Tmean 
        for d in np.arange(day_tot):
            var_daily[d] = np.nanmean(var[d*24:d*24+24])
    else: 
        day_tot    = int(nhours/24)+1
        var_daily  = np.zeros(day_tot)
        # Calculate daily Tmean
        var_daily[0] = np.nanmean(var[0:nhours%24])
        for d in np.arange(1, day_tot):
            var_daily[d] = np.nanmean(var[d*24:d*24+24])

    # Smoothing
    var_smooth = np.zeros(day_tot)

    # Calculate xx-day smoothing
    half_length = int((smooth_window-1)/2)
    print('half window length is ', half_length)

    for d in np.arange(day_tot):
        if d < half_length:
            var_smooth[d] = np.nanmean(var_daily[0:d+d+1])
        elif d >= day_tot-half_length:
            var_smooth[d] = np.nanmean(var_daily[d-(day_tot-d-1):])
        else: 
            var_smooth[d] = np.nanmean(var_daily[d-half_length:d+half_length+1])

    # Tmean > 5 degree 
    #  !!!!!! be careful I changed here !!!!!!!
    # var_greater_5deg  = np.where(var_smooth >=5.0, 1, 0) 
    var_greater_5deg  = np.where(var_smooth >=5.0, 1, 0) 

    # calculate the fraction of total days within 15 continuous days that T>5
    frac_greater_5deg = np.zeros(day_tot)
    for d in np.arange(day_tot):
        if d < day_tot-1 - 14:
            frac_greater_5deg[d] = np.nanmean(var_greater_5deg[d:d+15])
        else: 
            frac_greater_5deg[d] = np.nanmean(var_greater_5deg[d:])

    # Whether at 10 days in the 15 days T>5
    greater_5deg  = np.where(frac_greater_5deg >= (10./15.), 1, 0) 

    # Adjust summer days if the site is at south hemisphere 
    if lat < 0: 
        # South hemisphere: search the beginning of grow season from the middle of year
        season_adjust = 183
    else:
        # North hemisphere: search the beginning of grow season from the beginning of year
        season_adjust = 0
    day_e         = season_adjust

    # Calculate growth season for each year 
    growth_season = np.zeros(day_tot)

    for yr in np.arange(year_s,year_e+1):
        
        if yr % 4 == 0:
            yr_day_tot = 366
        else:
            yr_day_tot = 365

        day_s             = day_e
        day_e             = day_s + yr_day_tot

        greater_5deg_tmp  = greater_5deg[day_s:day_e]

        growth_season_tmp = np.zeros(len(greater_5deg_tmp))

        # Calculate the longest consecutive T > 5 degree

        max_start, max_end   = find_longest_consecutive(greater_5deg_tmp)
        if max_end+15 < day_tot:
            growth_season_tmp[max_start:max_end+15] = 1
        else:
            growth_season_tmp[max_start:] = 1
        
        growth_season[day_s:day_e] = growth_season_tmp

    print(growth_season)

    # ============ Setting for plotting ============
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[4,4],sharex=False, sharey=False, squeeze=False)
    yr_s    = 0

    for yr in np.arange(year_s,year_e+1):
        if yr % 4 == 0:
            yr_day_tot = 366
        else:
            yr_day_tot = 365
            
        plot = ax[0,0].plot(growth_season[yr_s:yr_s+yr_day_tot]) 

        yr_s = yr_s+yr_day_tot

    fig.savefig(f"./plots/Check_{site_name}_growth_season.png",bbox_inches='tight',dpi=300) 
   
    # ============ Write the growing season ============
    # set time dimensions
    f.createDimension('days', day_tot)
    days                = f.createVariable('days', 'f4', ('days'))
    days.standard_name  = 'days'
    days.units          = f'days since {year_s}-01-01 00:00:00.0'
    days[:]             = np.arange(0,day_tot,1)

    # create variables
    Var_daily               = f.createVariable('t2_daily', 'f4', 'days')
    Var_daily.standard_name = 'daily mean 2m temperature'
    Var_daily.units         = 'C'
    Var_daily[:]            = var_daily

    Var_smooth               = f.createVariable('t2_daily_smooth', 'f4', 'days')
    Var_smooth.standard_name = f'smoothed daily mean 2m temperature ({smooth_window}d-window)'
    Var_smooth.units         = 'C'
    Var_smooth[:]            = var_smooth

    Var_greater_5deg               = f.createVariable('t2_daily_smooth_greater_5deg', 'f4', 'days')
    Var_greater_5deg.standard_name = f'smoothed daily mean 2m temperature >= 5 degree'
    Var_greater_5deg.units         = '1: yes, 0: no'
    Var_greater_5deg[:]            = var_greater_5deg

    Frac_greater_5deg               = f.createVariable('fraction_of_t2_daily_smooth_greater_5deg', 'f4', 'days')
    Frac_greater_5deg.standard_name = f'fraction of smoothed daily mean 2m temperature >= 5 degree within 15 days'
    Frac_greater_5deg.units         = 'fraction'
    Frac_greater_5deg[:]            = frac_greater_5deg

    Growth_season               = f.createVariable('Growth_season', 'f4', 'days')
    Growth_season.standard_name = f'Growth_season_or_not'
    Growth_season.units         = '1: the day is in growth season, 0: not'
    Growth_season[:]            = growth_season

    f.close()
    time  = None
    var   = None
    lat   = None

    return


if __name__ == "__main__":

    # Path of ERA 5-land dataset
    ERA_path       = "/g/data/zz93/era5-land/reanalysis"
    nc_path        = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_GPP_PLUMBER2/nc_files"
    var_name       = "2t" # 2m temperature, units K

    PLUMBER2_path  = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"
    all_site_path  = sorted(glob.glob(PLUMBER2_path+"/*.nc"))
    site_names     = [os.path.basename(site_path).split(".")[0] for site_path in all_site_path]
    smooth_window  = 7 # odd number only

    for site_name in site_names[1:2]:
        calc_growth_season(nc_path, var_name, site_name, smooth_window=smooth_window)
    
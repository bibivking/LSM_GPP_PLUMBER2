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

def calc_extremes(var_name, nc_path):

    # Read data 
    input_file  = nc_path + "/sites_ERA5_land_"+var_name+".nc"
    f           = nc.Dataset(input_file, 'r+', format='NETCDF4')

    for site_name in site_names: 
        var     = f.variables[site_name][:]

        time    = nc.num2date(f.variables['time'][:],f.variables['time'].units,
                    only_use_cftime_datetimes=False, only_use_python_datetimes=True)

        # Calculate total days
        day_tot   = len(var)/24
        var_daily = np.zeros(day_tot)

        # Calculate daily Tmean or Psum
        if var_name == '2t':
            for d in np.arange(day_tot):
                var_daily[d] = np.nanmean(var[d*24:d*24+24])
        elif var_name == 'tp':
            for d in np.arange(day_tot):
                var_daily[d] = np.nansum(var[d*24:d*24+24])

        var_smooth = np.zeros(day_tot)

        # Smoothing
        if var_name == '2t':
            # Calculate 7-day smoothing
            for d in np.arange(day_tot):
                if d < 3:
                    var_smooth[d] = np.nanmean(var_daily[0:d+d+1])
                elif d >= day_tot-3:
                    var_smooth[d] = np.nanmean(var_daily[d-(day_tot-d-1):])
                else: 
                    var_smooth[d] = np.nanmean(var_daily[d-3:d+4])        

        elif var_name == 'tp':
            # Calculate 31-day smoothing
            for d in np.arange(day_tot):
                if d < 15:
                    var_smooth[d] = np.nansum(var_daily[0:d+d+1])
                elif d >= day_tot-15:
                    var_smooth[d] = np.nansum(var_daily[d-(day_tot-d-1):])
                else: 
                    var_smooth[d] = np.nansum(var_daily[d-15:d+16])        
                    
        # remove leap year  
        day_e  = 0
        year_s = time[0].year
        year_e = time[-1].year
        for year in np.arange(year_s,year_e+1):
            if year % 4 == 0:
                day_s = day_e
                day_e = day_s+366 # the first day of next year
                var_smooth.pop(day_s+31+29-1) # remove 29th Feb 
            else:
                day_s = day_e
                day_e = day_s+365 # the first day of next year

        # Calculate daily climatology
        var_smooth_reshape = var_smooth.reshape((year_e-year_s+1, 365))
        
        # Calculate climatological 
        var_extremes = np.zeros(365)
        for d in np.arange(365):
            if var_name == '2t':
                var_extremes[d] = np.nanpercentile(var_smooth_reshape[:,d],85)
            elif var_name == 'tp':
                var_extremes[d] = np.nanpercentile(var_smooth_reshape[:,d],15)

        var_extremes_leap          = np.zeros(366)
        var_extremes_leap[0:31+28] = var_extremes[0:31+28] 
        var_extremes_leap[31+28]   = (var_extremes[31+27] +  var_extremes[31+28])/2
        var_extremes_leap[31+29:]  = var_extremes[31+28:] 

        # tell whether a day is going through extreme weathers
        var_extreme_or_not = np.zeros(day_tot)
        day_e  = 0
        for year in np.arange(year_s, year_e+1):
            if year%4 == 0:
                day_s = day_e
                day_e = day_s+366            
                if var_name == '2t':
                    var_extreme_or_not[day_s:day_e] = np.where(var_daily > var_extremes_leap, 1, 0)
                elif var_name == 'tp':
                    var_extreme_or_not[day_s:day_e] = np.where(var_daily < var_extremes_leap, 1, 0)
            else:
                day_s = day_e
                day_e = day_s+365
                var_extreme_or_not[day_s:day_e] = np.where(var_extremes)                
                if var_name == '2t':
                    var_extreme_or_not[day_s:day_e] = np.where(var_daily > var_extremes, 1, 0)
                elif var_name == 'tp':
                    var_extreme_or_not[day_s:day_e] = np.where(var_daily < var_extremes, 1, 0)
            
        # Save to nc file
        output_file = nc_path + "/ERA5_land_"+var_name+"_extremes_"+site_name+".nc"
        f           = nc.Dataset(output_file, 'w', format='NETCDF4')

        ### Create nc file ###
        f.history       = "Created by: %s" % (os.path.basename(__file__))
        f.creation_date = "%s" % (datetime.now())
        f.description   = var_name+ 'climatology extremes from ERA5-land at ' + str(len(site_names)) +' PLUMBER2 sites, made by MU Mengyuan'
        f.Conventions   = "CF-1.0"

        # set time dimensions
        f.createDimension('days_normal_year', 365)
        f.createDimension('days_leap_year', 366)
        f.createDimension('time', day_tot)

        # output var_daily
        days_normal_year                = f.createVariable('days_normal_year', 'f4', ('days_normal_year'))
        days_normal_year.standard_name  = 'time'
        days_normal_year.units          = f'days since {year_s}-01-01 00:00:00.0'
        days_normal_year[:]             = np.arange(0,365,1)

        # output var_extreme_or_not  
        days_normal_year                = f.createVariable('days_normal_year', 'f4', ('days_normal_year'))
        days_normal_year.standard_name  = 'time'
        days_normal_year.units          = f'days since {year_s}-01-01 00:00:00.0'
        days_normal_year[:]             = np.arange(0,365,1)

        # output var_extremes

        # output var_extremes_leap


        
        for site_name in site_names:

            print(site_name)
            
            f = nc.Dataset(output_file, 'r+', format='NETCDF4')
            # set model names dimension
            f.createDimension(site_name, 365)

            # create variables
            var               = f.createVariable(site_name, 'f4', 'time')
            if var_name == '2t':
                var.standard_name = var_name +' at '+site_name
                var.units         = var_units
            elif var_name == '2t':
            var[:]            = values[site_name]

            f.close()
            var   = None

    return


if __name__ == "__main__":

    # Path of ERA 5-land dataset
    ERA_path       = "/g/data/zz93/era5-land/reanalysis"
    nc_path        = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_GPP_PLUMBER2/nc_files"
    PLUMBER2_path  = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"

    year_s         = 1971
    year_e         = 2020
    variable       = ["2t"] # 2m temperature, units K

    all_site_path  = sorted(glob.glob(PLUMBER2_path+"/*.nc"))
    site_names     = [os.path.basename(site_path).split(".")[0] for site_path in all_site_path]

    # Extracting data
    for var_name in var_names:
        calc_extremes(var_name, ERA_path, nc_path, site_names, year_s, year_e)

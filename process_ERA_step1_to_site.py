'''
Bin the dataset by VPD (and EF) and save in process4_output
Including:

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

def extract_value_at_year(var_name, ERA_path, year, site_names):

    """
    Read the value of the ERA5-land variable at nearest pixel to lat and lon input.

    Parameters:
        nc_file (str): The path to the netCDF4 file.
        lat (float): The latitude of the site.
        lon (float): The longitude of the site.

    Returns:
        the extracted value at the nearest pixel.
    """

    # Set file path
    file_path  = f"{ERA_path}/{var_name}/{year}"

    # Get all file names
    file_names = sorted(glob.glob(file_path+"/*.nc"))
    # print('file_names',file_names)

    # Set the dictionary
    values     = {}

    # Get lats and lons
    PLUMBER2_met_path  = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met"
    lat_dict, lon_dict = read_lat_lon(site_names, PLUMBER2_met_path)

    for i, file_name in enumerate(file_names):

        # print('file_name', file_name)

        f         = nc.Dataset(file_name)
        latitude  = f.variables['latitude'][:]
        longitude = f.variables['longitude'][:]

        # print('latitude', latitude)
        # print('longitude', longitude)

        if var_name == '2t':
            print('var_name= ' , var_name, ' converts to C')
            var   = f.variables['t2m'][:,:,:] - 273.15
        else:
            var   = f.variables[var_name][:,:,:]

        for site_name in site_names:

            # print('site_name', site_name)

            # Get lat and lon
            lat       = lat_dict[site_name]
            lon       = lon_dict[site_name]

            # Find the indices of the nearest pixels to lat and lon.
            lat_idx   = np.argmin(np.abs(latitude - lat))
            lon_idx   = np.argmin(np.abs(longitude - lon))

            # print(site_name, 'lat_idx', lat_idx, 'lon_idx', lon_idx)

            # Read the climate_class value of the nearest pixel.
            if i == 0:
                if var_name == '2t':
                    values[site_name] = var[:, lat_idx, lon_idx]
                elif var_name == 'tp':
                    tmp               = var[:, lat_idx, lon_idx]
                    tmp[1:]           = tmp[1:] - tmp[0:-1]
                    values[site_name] = tmp
            else:
                if var_name == '2t':
                    values[site_name] = np.append(values[site_name], var[:, lat_idx, lon_idx])
                elif var_name == 'tp':
                    tmp               = var[:, lat_idx, lon_idx]
                    tmp[1:]           = tmp[1:] - tmp[0:-1]
                    values[site_name] = np.append(values[site_name], tmp)

        f.close()

    # print(values)

    return lat, lon, values

def extract_ERA5(var_name, ERA_path, nc_path, site_names, year_s, year_e):

    '''
    Extracting ERA5 data
    '''

    year_series = np.arange(year_s,year_e+1,1)
    ntime       = 0

    # Set units
    if var_name   == '2t':
        var_units = 'C'
    elif var_name == 'tp':
        var_units = 'mm/h'

    # parallel extract site data
    for i, year in enumerate(year_series):

        if i == 0:
            Lat, Lon, values      = extract_value_at_year(var_name, ERA_path, year, site_names)
        else:
            Lat, Lon, values_tmp  = extract_value_at_year(var_name, ERA_path, year, site_names)

            for site_name in site_names:
                values[site_name] = np.append(values[site_name], values_tmp[site_name])

        # add to total hours
        if year % 4 == 0:
            print(year, 'is a leap year')
            ntime = ntime + 366*24
        else:
            print(year, 'is not a leap year')
            ntime = ntime + 365*24

    for site_name in site_names:

        print(site_name,'len(values[site_name])',len(values[site_name]))

        # make output file
        output_file = f'{nc_path}/{site_name}_ERA5_land_{var_name}.nc'

        f           = nc.Dataset(output_file, 'w', format='NETCDF4')

        # set model names dimension
        f.createDimension(var_name, ntime)

        ### Create nc file ###
        f.history       = "Created by: %s" % (os.path.basename(__file__))
        f.creation_date = "%s" % (datetime.now())
        f.description   = f'ERA5-land {var_name} at {site_name} PLUMBER2 sites, made by MU Mengyuan'
        f.Conventions   = "CF-1.0"

        # set time dimensions
        f.createDimension('time', ntime)
        time                = f.createVariable('time', 'f4', ('time'))
        time.standard_name  = 'time'
        time.units          = f'hours since {year_s}-01-01 00:00:00.0'
        time[:]             = np.arange(0,ntime,1)

        # set lat dimensions
        f.createDimension('lat', 1)
        lat                = f.createVariable('lat', 'f4', ('lat'))
        lat.standard_name  = 'latitude'
        lat[:]             = Lat

        # set lon dimensions
        f.createDimension('lon', 1)
        lon                = f.createVariable('lon', 'f4', ('lon'))
        lon.standard_name  = 'longitude'
        lon[:]             = Lon

        # create variables
        var               = f.createVariable(var_name, 'f4', 'time')
        var.standard_name = var_name
        var.units         = var_units
        var[:]            = values[site_name]

        f.close()
        time  = None
        var   = None
        lat   = None
        lon   = None

    return

if __name__ == "__main__":

    # Path of ERA 5-land dataset
    ERA_path       = "/g/data/zz93/era5-land/reanalysis"
    nc_path        = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_GPP_PLUMBER2/nc_files"
    PLUMBER2_path  = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"

    year_s         = 1971
    year_e         = 2023
    var_names      = ["2t"] # 2m temperature, units K

    all_site_path  = sorted(glob.glob(PLUMBER2_path+"/*.nc"))
    site_names     = [os.path.basename(site_path).split(".")[0] for site_path in all_site_path]

    # Extracting data
    for var_name in var_names:
        extract_ERA5(var_name, ERA_path, nc_path, site_names, year_s, year_e)

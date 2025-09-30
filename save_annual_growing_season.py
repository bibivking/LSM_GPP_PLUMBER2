'''
'''

__author__  = "Mengyuan Mu"
__version__ = "1.0 (30.09.2024)"
__email__   = "mu.mengyuan815@gmail.com"

#==============================================
import os
import gc
import sys
import glob
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm  # Progress bar library
import multiprocessing as mp
from PLUMBER2_GPP_common_utils import *

def calculate_growing_season(site_name):

    secondly_to_hourly = 3600.

    # Prepare dataset
    PLUMBER2_path_site = f"/g/data/w97/mm3972/scripts/PLUMBER2/LSM_GPP_PLUMBER2/nc_files/{site_name}.nc"
    f                  = nc.Dataset(PLUMBER2_path_site, mode='r')
    var                = pd.DataFrame(f.variables['obs_GPP'][:].data, columns=['GPP'])

    # Read time
    time               = nc.num2date(f.variables['CABLE_time'][:], f.variables['CABLE_time'].units,
                         only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    ntime              = len(time)

    time_intervals = np.diff(time)[0] / timedelta(seconds=3600)
    day_tot_step   = int(24 / time_intervals)

    year           = np.zeros(ntime)
    month          = np.zeros(ntime)
    day            = np.zeros(ntime)

    for tt, t in enumerate(time):
        year[tt]  = t.year
        month[tt] = t.month
        day[tt]   = t.day

    var['year']   = year
    var['month']  = month
    var['day']    = day

    # Change units from gC/m2/s to gC/m2/time step
    var.loc[:, 'GPP'] = var['GPP'][:] * secondly_to_hourly * time_intervals
    var_daily         = var.groupby(['year', 'month', 'day']).sum(numeric_only=True).reset_index()

    # Calculate the 15-day rolling mean of GPP
    var_daily['GPP_15d_smooth'] = var_daily['GPP'].rolling(window=15, min_periods=1, center=True).mean()

    if site_name == 'SD-Dem':
        var_daily = var_daily[var_daily.year != 2006]
    elif site_name == 'US-PFa':
        var_daily = var_daily[var_daily.year != 1995]

    unique_year = np.unique(year)

    for uyr in unique_year:
        if np.sum(year == uyr) < 200:
            var_daily = var_daily[var_daily.year != uyr]
            print(site_name, uyr, 'only has ', np.sum(year == uyr), 'data, removed')

    # Calculate 95th percentile for each year
    percentile_95_GPP = var_daily.groupby('year')['GPP'].quantile(0.95).reset_index()
    percentile_95_GPP.rename(columns={'GPP': '95th_percentile_GPP'}, inplace=True)

    # Merge the 95th percentile values back into the var_daily DataFrame
    var_daily = pd.merge(var_daily, percentile_95_GPP, on='year')

    # Set GPP to np.nan if GPP is less than half of the 95th percentile of that year
    var_daily['GPP_growing_season_only'] = np.where( var_daily['GPP_15d_smooth'] < var_daily['95th_percentile_GPP'] / 2,
                                                      np.nan, var_daily['GPP'] )

    var_daily['is_growing_season']       = np.where(np.isnan(var_daily['GPP_growing_season_only']), 0, 1)

    var_daily.to_csv(f'./txt/growing_season/growing_season_{site_name}.csv', index=False)

    return

def save_growing_season_value(model_in, site_name, var_name, IGBP=None, clim_type=None, is_growing_season=True):

    secondly_to_hourly = 3600.

    models_calc_LAI    = ['ORC2_r6593','ORC2_r6593_CO2','ORC3_r7245_NEE','ORC3_r8120','GFDL','SDGVM','QUINCY','NoahMPv401']
    model_LAI_names    = {'ORC2_r6593':'lai','ORC2_r6593_CO2':'lai','ORC3_r7245_NEE':'lai','ORC3_r8120':'lai',
                          'GFDL':'lai', 'SDGVM':'lai','QUINCY':'LAI','NoahMPv401':'LAI'} #

    PLUMBER2_path_site = f"/g/data/w97/mm3972/scripts/PLUMBER2/LSM_GPP_PLUMBER2/nc_files/{site_name}.nc"
    PLUMBER2_met_path  = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"
    file_met_path      = glob.glob(PLUMBER2_met_path+"/*"+site_name+"*.nc")

    # prepare dataset
    f                  = nc.Dataset(PLUMBER2_path_site, mode='r')
    f_met              = nc.Dataset(file_met_path[0], mode='r')

    var_in             = pd.DataFrame(f.variables['obs_Tair'][:].data-273.15, columns=['Tair'])
    var_in['SWdown']   = f.variables['obs_SWdown'][:].data
    var_in['CO2']      = f_met.variables['CO2air'][:,0,0].data
    var_in['VPD']      = f_met.variables['VPD'][:,0,0].data
    var_in['Precip']   = f_met.variables['Precip'][:,0,0].data

    # Read time
    time   = nc.num2date(f.variables['CABLE_time'][:],f.variables['CABLE_time'].units,
                         only_use_cftime_datetimes=False,only_use_python_datetimes=True)
    ntime  = len(time)
    time_intervals = np.diff(time)[0]/timedelta(seconds=3600)
    day_tot_step   = int(24/time_intervals)

    year   = np.zeros(ntime)
    month  = np.zeros(ntime)
    day    = np.zeros(ntime)

    for tt,t in enumerate(time):
        year[tt]  = t.year
        month[tt] = t.month
        day[tt]   = t.day

    var_in['year']  = year
    var_in['month'] = month
    var_in['day']   = day

    if ((var_name == 'NEE' and (model_in == 'NoahMPv401' or model_in == 'GFDL' or model_in == 'STEMMUS-SCOPE'))
        or (var_name == 'GPP' and model_in == 'CHTESSEL_Ref_exp1')):
        try:
            var_in[var_name] = f.variables[f'{model_in}_{var_name}_gap_fill'][:].data*(-1)
        except:
            var_in[var_name] = f.variables[f'{model_in}_{var_name}'][:].data*(-1)
    else:
        try:
            var_in[var_name] = f.variables[f'{model_in}_{var_name}_gap_fill'][:].data
        except:
            var_in[var_name] = f.variables[f'{model_in}_{var_name}'][:].data

    if model_in in models_calc_LAI:
        if model_in == 'QUINCY' and site_name == 'US-Ha1':
            temp          = read_LAI_model(site_name, model_in, model_LAI_names[model_in])
            var_in['LAI'] = temp[8760:]
        else:
            var_in['LAI'] = read_LAI_model(site_name, model_in, model_LAI_names[model_in])
    else:
        var_in['LAI'] = read_LAI_obs(site_name, PLUMBER2_met_path)

    var_in['LAI'] = np.where(var_in['LAI'] < 0, np.nan, var_in['LAI'])

    try:
        var_in['SMtop1m'] = f.variables[f'{model_in}_SMtop1m'][:].data
    except:
        var_in['SMtop1m'] = f.variables['model_mean_SMtop1m'][:].data


    # from per second to per time step
    var_in.loc[:,'Precip'] = var_in['Precip'][:]*secondly_to_hourly*time_intervals

    # Change units from gC/m2/s to gC/m2/year
    if var_name in ['NEE','GPP']:
        var_in.loc[:,var_name] = var_in[var_name][:]*secondly_to_hourly*time_intervals

    # To daily data
    var_mean  = var_in.groupby(['year','month','day']).agg({
    'Tair'   : 'mean',  # Replace with the columns you want to average
    'SWdown' : 'mean',   # Replace with the columns you want to sum
    'CO2'    : 'mean',
    'VPD'    : 'mean',
    'Precip' : 'sum',
    var_name : 'sum',
    'LAI'    : 'mean',
    'SMtop1m': 'mean',
    }).reset_index()

    print('before remove nan year and year has less than 200 days data', var_mean)

    if site_name == 'SD-Dem':
        var_mean = var_mean[var_mean.year != 2006]
    elif site_name == 'US-PFa':
        var_mean = var_mean[var_mean.year != 1995]

    unique_year = np.unique(year)
    for uyr in unique_year:
        if np.sum(var_mean.year == uyr) < 200:
            var_mean = var_mean[var_mean.year != uyr]
            print(site_name, model_in, uyr, 'only has ', np.sum(year == uyr), 'data, removed')
    print('after remove nan year and year has less than 200 days data', var_mean)

    ### To annual growing season value
    growing_season = pd.read_csv(f'./txt/growing_season/growing_season_{site_name}.csv')

    try:
        if is_growing_season:
            var_tmp    = var_mean[growing_season['is_growing_season'].values == 1]
        else:
            var_tmp    = var_mean[growing_season['is_growing_season'].values == 0]

        var_out        = var_tmp.groupby(['year']).agg({
                                                        'Tair'   : 'mean',  # Replace with the columns you want to average
                                                        'SWdown' : 'mean',   # Replace with the columns you want to sum
                                                        'CO2'    : 'mean',
                                                        'VPD'    : 'mean',
                                                        'Precip' : 'mean',
                                                        var_name : 'mean',
                                                        'LAI'    : 'mean',
                                                        'SMtop1m': 'mean',
                                                        }).reset_index()

        var_out['site_name'] = site_name

        if IGBP != None:
            var_out['IGBP']  = IGBP
        if clim_type !=None:
            var_out['clim_type'] = clim_type

        if is_growing_season:
            directory = f'./txt/{var_name}_annual_growing_season/'
            check_directory_exist(directory)
            var_out.to_csv(f'{directory}/{var_name}_annual_growing_season_{model_in}_{site_name}.csv', index=False)
        else:
            directory = f'./txt/{var_name}_annual_non_growing_season/'
            check_directory_exist(directory)
            var_out.to_csv(f'{directory}/{var_name}_annual_non_growing_season_{model_in}_{site_name}.csv', index=False)

    except Exception as e:
            print(f"An error occurred: {e}, {site_name}, {model_in}")
            print('len(growing_season), len(var_in)', len(growing_season), len(var_mean))
            print('var_mean.index',var_mean.index)
            print('growing_season.index',growing_season.index)

    return

def save_growing_season_value_parallal(var_name, model_in, remain_sites, is_growing_season=True):

    PLUMBER2_met_path = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"
    sites_IGBP        = read_IGBP_veg_type(site_names, PLUMBER2_met_path)

    site_character_file = '/g/data/w97/mm3972/scripts/PLUMBER2/LSM_GPP_PLUMBER2/txt/site_character.csv'
    sites_clim          = pd.read_csv(site_character_file)

    # Create list of arguments to pass to the parallel function
    args_list         = [(model_in, site_name, var_name, sites_IGBP[site_name],
                          sites_clim.loc[sites_clim['site_name']=='AU-Tum', 'clim_type'].values[0],
                          is_growing_season) for site_name in remain_sites]

    # Run in parallel using starmap, which allows multiple arguments
    with mp.Pool(mp.cpu_count()) as pool:
        pool.starmap(save_growing_season_value, args_list)

    # =============== Save to one file ===============
    # Initialize an empty list to hold the dataframes
    all_var_means = []

    for site_name in remain_sites:

        if is_growing_season:
            var_out = pd.read_csv(f'./txt/{var_name}_annual_growing_season/{var_name}_annual_growing_season_{model_in}_{site_name}.csv')
        else:
            var_out = pd.read_csv(f'./txt/{var_name}_annual_non_growing_season/{var_name}_annual_non_growing_season_{model_in}_{site_name}.csv')

        all_var_means.append(var_out)

    # Concatenate all the var_mean dataframes
    var_means = pd.concat(all_var_means, ignore_index=True)

    if is_growing_season:
        var_means.to_csv(f'./txt/{var_name}_annual_growing_season/{var_name}_annual_growing_season_{model_in}.csv', index=False)
    else:
        var_means.to_csv(f'./txt/{var_name}_annual_non_growing_season/{var_name}_annual_non_growing_season_{model_in}.csv', index=False)
    return

if __name__ == "__main__":

    site_names, IGBP_types, clim_types, model_names = load_default_list()

    # Calculate remaining sites
    remove_site       = get_removed_site_names()
    site_names, IGBP_types, clim_types, model_names = load_default_list()
    set_site_names    = set(site_names)
    set_remove_site   = set(remove_site)
    remain_sites      = set_site_names - set_remove_site
    remain_sites      = list(remain_sites)

    # ================== Calculate growing season ====================
    if 0:
        for site_name in remain_sites:
            calculate_growing_season(site_name)

    # ================== Calculate annual values in growing season ====================
    if 1:
        PLUMBER2_path_site = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_GPP_PLUMBER2/nc_files/AU-How.nc"
        PLUMBER2_met_path  = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"

        is_growing_season  = False
        var_name           = 'NEE'
        f                  = nc.Dataset(PLUMBER2_path_site, mode='r')
        model_list         = f.variables[f'{var_name}_models'][:]
        model_list         = model_list.tolist()
        model_list.append('obs')
        f.close()
        for model_in in model_list:
            save_growing_season_value_parallal(var_name, model_in, remain_sites, is_growing_season=is_growing_season)

        # var_name           = 'GPP'
        # f                  = nc.Dataset(PLUMBER2_path_site, mode='r')
        # model_list         = f.variables[f'{var_name}_models'][:]
        # model_list         = model_list.tolist()
        # model_list.append('obs')
        # f.close()
        # for model_in in model_list:
        #     save_growing_season_value_parallal(var_name, model_in, remain_sites, is_growing_season=is_growing_season)
        #
        # is_growing_season  = True
        # var_name           = 'NEE'
        # f                  = nc.Dataset(PLUMBER2_path_site, mode='r')
        # model_list         = f.variables[f'{var_name}_models'][:]
        # model_list         = model_list.tolist()
        # model_list.append('obs')
        # f.close()
        # for model_in in model_list:
        #     save_growing_season_value_parallal(var_name, model_in, remain_sites, is_growing_season=is_growing_season)
        #
        # var_name           = 'GPP'
        # f                  = nc.Dataset(PLUMBER2_path_site, mode='r')
        # model_list         = f.variables[f'{var_name}_models'][:]
        # model_list         = model_list.tolist()
        # model_list.append('obs')
        # f.close()
        # for model_in in model_list:
        #     save_growing_season_value_parallal(var_name, model_in, remain_sites, is_growing_season=is_growing_season)

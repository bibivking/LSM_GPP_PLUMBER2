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

def get_annual_value(model_in, site_name, var_name, IGBP=None, clim_type=None):
    
    secondly_to_annually = 3600*24*365.
    
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
    for tt,t in enumerate(time):
        year[tt] = t.year

    var_in['year'] = year
    
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

    print(model_in, "before fix LAI<0, np.any(var_in['LAI']<0)", np.any(var_in['LAI']<0))
    var_in['LAI'] = np.where(var_in['LAI'] < 0, np.nan, var_in['LAI'])
    print(model_in, "after fix LAI<0, np.sum(np.isnan(var_in['LAI']))", np.sum(np.isnan(var_in['LAI'])))

    try: 
        var_in['SMtop1m'] = f.variables[f'{model_in}_SMtop1m'][:].data
    except: 
        var_in['SMtop1m'] = f.variables['model_mean_SMtop1m'][:].data
        
    # Change units from gC/m2/s to gC/m2/year
    if var_name in ['NEE','GPP']:
        var_in.loc[:,var_name] = var_in[var_name][:]*secondly_to_annually
    
    var_mean  = var_in.groupby(['year']).mean(numeric_only=True)
    
    var_mean['site_name']= site_name

    if IGBP != None:
        var_mean['IGBP'] = IGBP
    if clim_type !=None:
        var_mean['clim_type'] = clim_type

    print('before remove nan year and year has less than 200 days data', var_mean)

    if site_name == 'SD-Dem':
        var_mean = var_mean[var_mean.index != 2006]
    elif site_name == 'US-PFa':
        var_mean = var_mean[var_mean.index != 1995]
    
    unique_year = np.unique(year)
    for uyr in unique_year:
        if np.sum(year == uyr) < 200 * day_tot_step:
            var_mean = var_mean[var_mean.index != uyr]
            print(site_name, model_in, uyr, 'only has ', np.sum(year == uyr), 'data, removed')

    print('after remove nan year and year has less than 200 days data', var_mean)

    var_mean.to_csv(f'./txt/{var_name}_annual_value/{var_name}_annual_value_{model_in}_{site_name}.csv')
    
    return var_mean

def save_annual_value(var_name, model_in):

    remove_site        = get_removed_site_names()

    site_names, IGBP_types, clim_types, model_names = load_default_list()

    PLUMBER2_met_path  = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"
    sites_IGBP         = read_IGBP_veg_type(site_names, PLUMBER2_met_path)

    # Initialize an empty list to hold the dataframes
    all_var_means      = []

    # Loop over site names
    for site_name in site_names:
        if site_name not in remove_site:
            try:
                var_mean         = get_annual_value(model_in, site_name, var_name)
                var_mean['IGBP'] = sites_IGBP[site_name]
                all_var_means.append(var_mean)
            except Exception as e:
                print(f'Error occurred while processing {model_in} at {site_name}: {str(e)}')                
                
    # Concatenate all the var_mean dataframes
    var_means = pd.concat(all_var_means, ignore_index=True)
    var_means.to_csv(f'./txt/{var_name}_annual_value/{var_name}_annual_value_{model_in}.csv')

    return

def save_annual_value_parallal(var_name, model_in):

    remove_site       = get_removed_site_names()
    site_names, IGBP_types, clim_types, model_names = load_default_list()

    # Calculate remaining sites
    set_site_names    = set(site_names)
    set_remove_site   = set(remove_site)
    remain_sites      = set_site_names - set_remove_site
    remain_sites      = list(remain_sites)

    PLUMBER2_met_path = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"
    sites_IGBP        = read_IGBP_veg_type(site_names, PLUMBER2_met_path)
    
    site_character_file = '/g/data/w97/mm3972/scripts/PLUMBER2/LSM_GPP_PLUMBER2/txt/site_character.csv'
    sites_clim          = pd.read_csv(site_character_file)

    print("np.array(sites_clim.loc[sites_clim['site_name']=='AU-Tum', 'clim_type'].values)",
          sites_clim.loc[sites_clim['site_name']=='AU-Tum', 'clim_type'].values[0])

    # Create list of arguments to pass to the parallel function
    args_list         = [(model_in, site_name, var_name, sites_IGBP[site_name], 
                          sites_clim.loc[sites_clim['site_name']=='AU-Tum', 'clim_type'].values[0])
                          for site_name in remain_sites]

    # Run in parallel using starmap, which allows multiple arguments
    with mp.Pool(mp.cpu_count()) as pool:
        pool.starmap(get_annual_value, args_list)

    # Initialize an empty list to hold the dataframes
    all_var_means = []

    # Loop over site names
    for site_name in remain_sites:
        try:
            var_mean = pd.read_csv(f'./txt/{var_name}_annual_value/{var_name}_annual_value_{model_in}_{site_name}.csv')
            all_var_means.append(var_mean)
        except Exception as e:
            print(f'Error occurred while processing {model_in} at {site_name}: {str(e)}')                
                
    # Concatenate all the var_mean dataframes
    var_means = pd.concat(all_var_means, ignore_index=True)
    var_means.to_csv(f'./txt/{var_name}_annual_value/{var_name}_annual_value_{model_in}.csv')

    return

if __name__ == "__main__":

    var_name           = 'NEE'

    PLUMBER2_path_site = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_GPP_PLUMBER2/nc_files/AU-How.nc"
    PLUMBER2_met_path  = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"

    f                  = nc.Dataset(PLUMBER2_path_site, mode='r')
    model_list         = f.variables[f'{var_name}_models'][:]
    model_list         = model_list.tolist()
    model_list.append('obs')
    f.close()

    for model_in in model_list:
        print('Model is ', model_in)
        save_annual_value_parallal(var_name, model_in)

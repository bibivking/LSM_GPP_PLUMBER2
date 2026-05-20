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
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta
import multiprocessing as mp
from PLUMBER2_GPP_common_utils import *


import pandas as pd
import pingouin as pg

# Example of loading your data
# Replace 'your_data.csv' with your actual data file or load from a DataFrame
data = pd.read_csv('your_data.csv')

# Columns in your data
# Assume the columns are named 'GPP', 'wind_speed', 'air_temperature', 'radiation', 'CO2'
gpp = data['GPP']
factors = ['wind_speed', 'air_temperature', 'radiation', 'CO2']

# Create a DataFrame for storing partial correlation results
partial_corr_results = pd.DataFrame(columns=['Factor', 'Partial Corr', 'p-value'])

# Loop over each factor to calculate partial correlation with GPP, controlling for the others
for factor in factors:
    control_vars = [f for f in factors if f != factor]
    
    # Calculate partial correlation
    p_corr = pg.partial_corr(data=data, x='GPP', y=factor, covar=control_vars, method='pearson')
    
    # Store the result
    partial_corr_results = partial_corr_results.append({
        'Factor': factor,
        'Partial Corr': p_corr['r'].values[0],
        'p-value': p_corr['p-val'].values[0]
    }, ignore_index=True)

# Display the results
print(partial_corr_results)


def save_annual_mean(var_name, model_in, per_LAI=False):

    var       = np.zeros(170)
    Site_name = [""] * 170    # Creates a list with 170 empty strings
    lat       = np.zeros(170)
    lon       = np.zeros(170)

    for i, site_name in enumerate(remain_sites): 
        print(site_name)
        Site_name[i]       = site_name

        PLUMBER2_path_site = f"/srv/ccrc/LandAP/z5218916/script/PLUMBER2/LSM_GPP_PLUMBER2/nc_files/{site_name}.nc"
        PLUMBER2_met_path  = "/srv/ccrc/LandAP/z5218916/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"
        PLUMBER2_flux_path = "/srv/ccrc/LandAP/z5218916/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Flux/"
        file_path          = glob.glob(PLUMBER2_flux_path+"/*"+site_name+"*.nc")

        with nc.Dataset(PLUMBER2_path_site, mode='r') as f:
            try:
                if var_name == 'NEE':
                    if model_in == 'NoahMPv401' or model_in == 'GFDL' or model_in == 'STEMMUS-SCOPE':
                        var_tmp = f.variables[model_in + '_NEE'][:].data*(-1)
                    else:
                        var_tmp = f.variables[model_in + '_NEE'][:].data
                elif var_name == 'GPP':
                    var_tmp = f.variables[model_in + '_GPP'][:].data
            except:
                print(model_in, site_name, 'not exists')
                continue

            if per_LAI:
                if model_in in models_calc_LAI:
                    # print('in models_calc_LAI', model_in)
                    LAI = read_LAI_model(site_name, model_in, model_LAI_names[model_in])
                else:
                    # print('not in models_calc_LAI', model_in)
                    LAI = read_LAI_obs(site_name, PLUMBER2_met_path)
                tmp = np.where(LAI == 0, np.nan, var_tmp/LAI)
                var[i] = np.nanmean(tmp)*365*24*3600
            else:
                var[i] = np.nanmean(var_tmp)*365*24*3600


        with nc.Dataset(file_path[0], mode='r') as f_flux:

            lat[i] = f_flux.variables['latitude'][0,0] 
            lon[i] = f_flux.variables['longitude'][0,0] 

    #             if var_name == 'NEE':
    #                 var_qc = f_flux.variables['NEE_qc'][:,0,0]
    #             elif var_name == 'GPP':
    #                 var_qc = f_flux.variables['GPP_qc'][:,0,0]

    #             time   = nc.num2date(f_flux.variables['time'][:],f_flux.variables['time'].units,
    #                                  only_use_cftime_datetimes=False,only_use_python_datetimes=True)

    #             fig1 = plt.figure(figsize=(10, 5))
    #             plot = plt.plot(time,var_qc)
    #             plt.savefig(f'./plots/{var_name}_qc_obs_{site_name}.png',dpi=300)

    var_out              = pd.DataFrame(var, columns=[var_name])
    var_out['lat']       = lat
    var_out['lon']       = lon
    var_out['site_name'] = Site_name

    if per_LAI:
        var_out.to_csv(f'./txt/{var_name}_annual_mean_per_LAI/{var_name}_annual_mean_per_LAI_{model_in}.csv', index=False)
    else:
        var_out.to_csv(f'./txt/{var_name}_annual_mean/{var_name}_annual_mean_{model_in}.csv', index=False)

def save_annual_mean_parallal(var_name, per_LAI=False):
    
    PLUMBER2_path_site = "/srv/ccrc/LandAP/z5218916/script/PLUMBER2/LSM_GPP_PLUMBER2/nc_files/AU-How.nc"
    f                  = nc.Dataset(PLUMBER2_path_site, mode='r')
    model_list         = f.variables[f'{var_name}_models'][:]
    model_list         = model_list.tolist()
    model_list.append('obs')
    f.close()

    # Create a pool of workers (28 CPUs)
    with mp.Pool() as pool:
        # Distribute the tasks across CPUs
        pool.starmap(save_annual_mean, [(var_name, model_in, per_LAI) for model_in in model_list])

if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    PLUMBER2_path      = "/srv/ccrc/LandAP/z5218916/data/PLUMBER2/"
    PLUMBER2_flux_path = "/srv/ccrc/LandAP/z5218916/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Flux/"
    PLUMBER2_met_path  = "/srv/ccrc/LandAP/z5218916/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"

    site_names, IGBP_types, clim_types, model_names = load_default_list()

    remove_site        = get_removed_site_names()

    models_calc_LAI   = ['ORC2_r6593','ORC2_r6593_CO2','ORC3_r7245_NEE','ORC3_r8120','GFDL','SDGVM','QUINCY','NoahMPv401']
    model_LAI_names   = {'ORC2_r6593':'lai','ORC2_r6593_CO2':'lai','ORC3_r7245_NEE':'lai','ORC3_r8120':'lai',
                        'GFDL':'lai', 'SDGVM':'lai','QUINCY':'LAI','NoahMPv401':'LAI'} #

    # Calculate remaining sites
    set_site_names      = set(site_names)
    set_remove_site     = set(remove_site)
    remain_sites        = set_site_names - set_remove_site
    remain_sites        = list(remain_sites)

    # var_name = 'NEE'
    # per_LAI  = False
    # save_annual_mean_parallal(var_name, per_LAI=per_LAI)

    var_name = 'NEE'
    per_LAI  = True
    save_annual_mean_parallal(var_name, per_LAI=per_LAI)

    # var_name = 'GPP'
    # per_LAI  = False
    # save_annual_mean_parallal(var_name, per_LAI=per_LAI)

    # var_name = 'GPP'
    # per_LAI  = True
    # save_annual_mean_parallal(var_name, per_LAI=per_LAI)
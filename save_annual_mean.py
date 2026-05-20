'''
per_LAI shouldn't turn on, since the annual mean of NEE or GPP divided by LAI means nothing 
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
                if ((var_name == 'NEE' and (model_in == 'NoahMPv401' or model_in == 'GFDL' or model_in == 'STEMMUS-SCOPE'))
                    or (var_name == 'GPP' and model_in == 'CHTESSEL_Ref_exp1')):                                                
                    try:
                        var_tmp = f.variables[f'{model_in}_{var_name}_gap_fill'][:].data*(-1)
                    except:
                        var_tmp = f.variables[f'{model_in}_{var_name}'][:].data*(-1)
                else:
                    try:
                        var_tmp = f.variables[f'{model_in}_{var_name}_gap_fill'][:].data
                    except:
                        var_tmp = f.variables[f'{model_in}_{var_name}'][:].data
            except:
                print(model_in, site_name, 'not exists')
                continue

            if site_name in ['SD-Dem', 'US-PFa']:
                time    = nc.num2date(f.variables['CABLE_time'][:], f.variables['CABLE_time'].units,
                                      only_use_cftime_datetimes=False,only_use_python_datetimes=True)
                var_tmp = set_nan_for_special_site(site_name, var_tmp, time)

            if per_LAI:
                if model_in in models_calc_LAI:
                    if model_in == 'QUINCY' and site_name == 'US-Ha1':
                        temp = read_LAI_model(site_name, model_in, model_LAI_names[model_in])
                        LAI  = temp[8760:]
                    else:
                        LAI  = read_LAI_model(site_name, model_in, model_LAI_names[model_in])
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

    var_name = 'NEE'
    per_LAI  = False
    save_annual_mean_parallal(var_name, per_LAI=per_LAI)

    # var_name = 'GPP'
    # per_LAI  = False
    # save_annual_mean_parallal(var_name, per_LAI=per_LAI)

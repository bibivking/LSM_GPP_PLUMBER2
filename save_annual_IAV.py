'''

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

def save_IAV(var_name, model_in):

    '''
    remove per_LAI, since it makes no sense to calculate the annual mean of the NEE/LAI
    '''
    
    nsite    = len(remain_sites)
    Site_name= [""] * nsite    # Creates a list with 170 empty strings
    Variance = np.zeros(nsite)
    lat      = np.zeros(nsite)
    lon      = np.zeros(nsite)
    
    for s, site_name in enumerate(remain_sites[:]):
        
        Site_name[s] =  site_name
        
        PLUMBER2_path_site = f"/g/data/w97/mm3972/scripts/PLUMBER2/LSM_GPP_PLUMBER2/nc_files/{site_name}.nc"
        PLUMBER2_met_path  = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Flux/"
        file_path          = glob.glob(PLUMBER2_met_path+"/*"+site_name+"*.nc")

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
                
                time    = nc.num2date(f.variables['CABLE_time'][:], f.variables['CABLE_time'].units,
                                        only_use_cftime_datetimes=False,only_use_python_datetimes=True)

                if site_name in ['SD-Dem', 'US-PFa']:
                    var_tmp = set_nan_for_special_site(site_name, var_tmp, time)
                    thres_num   = 6
                else:
                    thres_num   = 5
                var = pd.DataFrame(var_tmp, columns=[var_name])

                ntime      = len(time)
                year       = np.zeros(ntime)

                for i,t in enumerate(time):
                    year[i] = t.year

                var['year'] = year[:]
                var         = var.groupby(['year']).mean(numeric_only=True).reset_index()*365*24*3600
                
                if len(var) >= thres_num:
                    Variance[s] = np.nanvar(var[var_name])
                    with nc.Dataset(file_path[0], mode='r') as f_flux:
                        lat[s] = f_flux.variables['latitude'][0,0] 
                        lon[s] = f_flux.variables['longitude'][0,0]
            except:
                print(model_in, site_name, 'not exists')
                continue

    # Convert NEE to a numpy array (for easier manipulation)
    Variance             = np.array(Variance)
    
    var_out              = pd.DataFrame(Site_name, columns=['site_name'])
    var_out['variance']  = Variance
    var_out['lat']       = lat
    var_out['lon']       = lon

    # Get the directory path
    directory = f'./txt/{var_name}_IAV'
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)     
    var_out.to_csv(f'{directory}/{var_name}_IAV_{model_in}.csv', index=False)

    return

# Define a function to generate each plot
def save_IAV_parallal(var_name):
    
    PLUMBER2_path_site = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_GPP_PLUMBER2/nc_files/AU-How.nc"
    f                  = nc.Dataset(PLUMBER2_path_site, mode='r')
    model_list         = f.variables[f'{var_name}_models'][:]
    model_list         = model_list.tolist()
    model_list.append('obs')
    f.close()

    # Create a pool of workers (28 CPUs)
    with mp.Pool() as pool:
        # Distribute the tasks across CPUs
        pool.starmap(save_IAV, [ (var_name, model_in) for model_in in model_list])

    return

if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    PLUMBER2_path      = "/g/data/w97/mm3972/data/PLUMBER2/"
    PLUMBER2_flux_path = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Flux/"
    PLUMBER2_met_path  = "/g/data/w97/mm3972/data/Fluxnet_data/Post-processed_PLUMBER2_outputs/Nc_files/Met/"

    site_names, IGBP_types, clim_types, model_names = load_default_list()

    remove_site        = get_removed_site_names()

    models_calc_LAI    = ['ORC2_r6593','ORC2_r6593_CO2','ORC3_r7245_NEE','ORC3_r8120','GFDL','SDGVM','QUINCY','NoahMPv401']
    model_LAI_names    = {'ORC2_r6593':'lai','ORC2_r6593_CO2':'lai','ORC3_r7245_NEE':'lai','ORC3_r8120':'lai',
                          'GFDL':'lai', 'SDGVM':'lai','QUINCY':'LAI','NoahMPv401':'LAI'} #

    # Calculate remaining sites
    set_site_names      = set(site_names)
    set_remove_site     = set(remove_site)
    remain_sites        = set_site_names - set_remove_site
    remain_sites        = list(remain_sites)

    var_name='NEE'
    save_IAV_parallal(var_name)

    var_name='GPP'
    save_IAV_parallal(var_name)
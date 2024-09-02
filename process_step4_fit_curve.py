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

def bin_VPD(var_plot, model_out_list, uncertain_type='UCRTN_percentile'):

    # Set up the VPD bins
    vpd_top      = 10.001 #7.04
    vpd_bot      = 0.001 #0.02
    vpd_interval = 0.1 #0.04

    vpd_series   = np.arange(vpd_bot,vpd_top,vpd_interval)

    # Set up the values need to draw
    vpd_tot      = len(vpd_series)
    model_tot    = len(model_out_list)
    vpd_num      = np.zeros((model_tot, vpd_tot))
    var_vals     = np.zeros((model_tot, vpd_tot))
    var_vals_top = np.zeros((model_tot, vpd_tot))
    var_vals_bot = np.zeros((model_tot, vpd_tot))

    # Binned by VPD
    for j, vpd_val in enumerate(vpd_series):

        mask_vpd       = (var_plot['VPD'] > vpd_val-vpd_interval/2) & (var_plot['VPD'] < vpd_val+vpd_interval/2)

        if np.any(mask_vpd):

            var_masked = var_plot[mask_vpd]

            # Draw the line for different models
            for i, model_out_name in enumerate(model_out_list):

                if 'obs' in model_out_name:
                    head = ''
                else:
                    head = 'model_'

                # calculate mean value
                var_vals[i,j] = var_masked[head+model_out_name].mean(skipna=True)

                # # calculate median value
                # var_vals[i,j] = var_masked[head+model_out_name].median(skipna=True)

                vpd_num[i,j]  = np.sum(~np.isnan(var_masked[head+model_out_name]))
                #print('model_out_name=',model_out_name,'j=',j,'vpd_num[i,j]=',vpd_num[i,j])

                if uncertain_type=='UCRTN_one_std':
                    # using 1 std as the uncertainty
                    var_std   = var_masked[head+model_out_name].std(skipna=True)
                    var_vals_top[i,j] = var_vals[i,j] + var_std
                    var_vals_bot[i,j] = var_vals[i,j] - var_std

                elif uncertain_type=='UCRTN_percentile':
                    # using percentile as the uncertainty
                    var_temp  = var_masked[head+model_out_name]
                    mask_temp = ~ np.isnan(var_temp)
                    if np.any(mask_temp):
                        var_vals_top[i,j] = np.percentile(var_temp[mask_temp], 75)
                        var_vals_bot[i,j] = np.percentile(var_temp[mask_temp], 25)
                    else:
                        var_vals_top[i,j] = np.nan
                        var_vals_bot[i,j] = np.nan

                elif uncertain_type=='UCRTN_bootstrap':
                    # using bootstrap to get the confidence interval for the unknown distribution dataset

                    var_temp  = var_masked[head+model_out_name]
                    mask_temp = ~ np.isnan(var_temp)

                    # Generate confidence intervals for the SAMPLE MEAN with bootstrapping:
                    var_vals_bot[i,j], var_vals_top[i,j] = bootstrap_ci(var_temp[mask_temp], np.mean, n_samples=1000)

                    # # Generate confidence intervals for the SAMPLE MEAN with bootstrapping:
                    # var_vals_bot[i,j], var_vals_top[i,j] = bootstrap_ci(var_temp[mask_temp], np.median, n_samples=1000)

        else:
            print('In bin_VPD, binned by VPD, var_masked = np.nan. Please check why the code goes here')
            print('j=',j, ' vpd_val=',vpd_val)

            var_vals[:,j]     = np.nan
            vpd_num[:,j]      = np.nan
            var_vals_top[:,j] = np.nan
            var_vals_bot[:,j] = np.nan

    return vpd_series, vpd_num, var_vals, var_vals_top, var_vals_bot

def write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=None, bounds=None,
                  day_time=False, summer_time=False, IGBP_type=None, time_scale=None,
                  clim_type=None, energy_cor=False,VPD_num_threshold=None, LAI_range=None,
                  uncertain_type='UCRTN_percentile', models_calc_LAI=None, veg_fraction=None,
                  clarify_site={'opt':False,'remove_site':None}, standardize=None,
                  remove_strange_values=True, country_code=None,
                  hours_precip_free=None, method='CRV_bins',dist_type='Linear'):

    '''
    1. bin the dataframe by percentile of obs_EF
    2. calculate var series against VPD changes
    3. write out the var series
    '''

    # save data
    if var_name == 'NEE':
        var_name = 'NEP'

    # Get model lists
    if var_name == 'Gs':
        site_names, IGBP_types, clim_types, model_names = load_default_list()
        model_out_list = model_names['model_select_new']
    else:
        model_out_list = get_model_out_list(var_name)

    # Read in the selected raw data
    var_input = pd.read_csv(f'./txt/process3_output/curves/{file_input}',na_values=[''])
    site_num  = len(np.unique(var_input["site_name"]))

    # ============ Set the output file name ============
    folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, IGBP_type=IGBP_type,
                                                clim_type=clim_type, time_scale=time_scale, standardize=standardize,
                                                country_code=country_code, selected_by=selected_by, bounds=bounds,
                                                veg_fraction=veg_fraction, LAI_range=LAI_range, method=method,
                                                uncertain_type=uncertain_type, clarify_site=clarify_site)

    # Checks if a folder exists and creates it if it doesn't
    if not os.path.exists(f'./txt_out/{folder_name}'):
        os.makedirs(f'./txt_out/{folder_name}')

    # ============ Choosing fitting or binning ============

    # if method == 'CRV_bins':
    #     # ============ Bin by VPD ============
    #     vpd_series, vpd_num, var_vals, var_vals_top, var_vals_bot = bin_VPD(var_input, model_out_list, uncertain_type)

    #     # ============ Creat the output dataframe ============
    #     var = pd.DataFrame(vpd_series, columns=['vpd_series'])

    #     for i, model_out_name in enumerate(model_out_list):

    #         var[model_out_name+'_vpd_num'] = vpd_num[i,:]

    #         if VPD_num_threshold == None:
    #             var[model_out_name+'_vals'] = var_vals[i,:]
    #             var[model_out_name+'_top']  = var_vals_top[i,:]
    #             var[model_out_name+'_bot']  = var_vals_bot[i,:]
    #         else:
    #             var[model_out_name+'_vals'] = np.where(var[model_out_name+'_vpd_num'] >= VPD_num_threshold,
    #                                               var_vals[i,:], np.nan)
    #             var[model_out_name+'_top']  = np.where(var[model_out_name+'_vpd_num'] >= VPD_num_threshold,
    #                                               var_vals_top[i,:], np.nan)
    #             var[model_out_name+'_bot']  = np.where(var[model_out_name+'_vpd_num'] >= VPD_num_threshold,
    #                                               var_vals_bot[i,:], np.nan)

    #     var['site_num']    = site_num

    #     var.to_csv(f'./txt_out/{folder_name}/{var_name}{file_message}.csv')

    if method == 'CRV_fit_GAM_simple' or method == 'CRV_fit_GAM_complex':

        '''
        fitting GAM curve
        '''

        # ============ Check whether the folder save GAM_fit data exist ============
        if not os.path.exists(f'./txt_out/{folder_name}/GAM_fit'):
            os.makedirs(f'./txt_out/{folder_name}/GAM_fit')

        # ============ Creat the output dataframe ============
        # Use multiprocessing to fit GAM models in parallel
        with mp.Pool() as pool:
            pool.starmap(fit_GAM_for_model, [(folder_name, file_message, var_name, model_in, 
                         var_input['VPD'], var_input['obs_Tair'],var_input[model_in + "_EF"], 
                         var_input[get_header(model_in) + model_in], method, dist_type)
                         for model_in in model_out_list[:2]])

    return

def fit_GAM_for_model(folder_name, file_message, var_name, model_in, vpd, tair, 
                      EF, Qle, method='CRV_fit_GAM_simple', dist_type='Linear'):

    # Remove nan values
    vpd_tmp  = copy.deepcopy(vpd)
    tair_tmp = copy.deepcopy(tair)
    EF_tmp   = copy.deepcopy(EF)
    Qle_tmp  = copy.deepcopy(Qle)

    # copy.deepcopy: creates a complete, independent copy of an object
    # and its entire internal structure, including nested objects and any
    # references they contain.
    nonnan_mask = (~np.isnan(vpd_tmp)) & (~np.isnan(tair_tmp))  & (~np.isnan(EF_tmp))  & (~np.isnan(Qle_tmp))
    x1_values   = vpd_tmp[nonnan_mask]
    x2_values   = tair_tmp[nonnan_mask]    
    x3_values   = EF_tmp[nonnan_mask]    
    y_values    = Qle_tmp[nonnan_mask]

    if len(x1_values) <= 10:
        print("Alarm! Not enought sample")
        return 
    x_interval  = 100

    # Set x_series
    x1_series   = np.arange(np.min(x1_values), np.max(x1_values), x_interval)
    x2_series   = np.arange(np.min(x2_values), np.max(x2_values), x_interval)
    x3_series   = np.arange(np.min(x3_values), np.max(x3_values), x_interval)

    # Define grid search parameters
    lam        = np.logspace(-3, 3, 11) # Smoothing parameter range
    n_splines  = np.arange(4, 11, 1)    # Number of splines per smooth term range

    # Set up KFold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize empty lists for storing results
    models = []
    scores = []

    # Perform grid search
    for train_index, test_index in kf.split(x1_values):
        print(model_in, 'len(x1_values)',len(x1_values),
                        'len(train_index)',len(train_index),
                        'len(test_index)',len(test_index))
        X1_train, X1_test = x1_values[train_index], x1_values[test_index]
        X2_train, X2_test = x2_values[train_index], x2_values[test_index]
        X3_train, X3_test = x3_values[train_index], x3_values[test_index]

        y_train, y_test   = y_values[train_index], y_values[test_index]

        X1_train = X1_train.reshape(-1, 1)
        X2_train = X2_train.reshape(-1, 1)
        X3_train = X3_train.reshape(-1, 1)

        # Define and fit GAM model
        if dist_type=='Gamma':
            for n_spline in n_splines:
                gam = GammaGAM(l(0) + l(1) + l(2)).gridsearch([X1_train, X2_train, X3_train].T, y_train, lam=lam)
                models.append(gam)
                # s(0, spline_order=3, n_splines=n_spline) 
                # Evaluate model performance (replace with your preferred metric)
                score = gam.score([X1_test, X2_test, X3_test].T, y_test) # gam.score: compute the explained deviance for a trained model for a given X data and y labels
                scores.append(score)

    # Find the best model based on average score
    # For gam.score which calcuate deviance, the best model's score is closet to 0
    best_model_index = np.argmin(np.abs(scores))
    best_model       = models[best_model_index]
    best_score       = scores[best_model_index]

    print('scores',scores)
    print('best_score',best_score)
    print('best_model_index',best_model_index)
    print('best_model',best_model)
    print(f"Best model parameters: {best_model.lam}, {best_model.n_splines}")

    # Save the best model using joblib
    joblib.dump(best_model, f"./txt_out/GAM_fit/bestGAM_{var_name}{file_message}_{model_out_name}_{dist_type}.pkl")

    # Further analysis of the best model (e.g., plot smoothers, analyze interactions)
    y_pred = best_model.predict([x1_series,x2_series,x3_series].T)

    norm   = plt.Normalize(vmin=round(np.min(y_pred)), vmax=round(np.max(y_pred)))

    # Note that The code calculates 95% confidence intervals, but remember that confidence
    #           intervals should generally be calculated on the actual test data points,
    #           not a new set of equally spaced values like x_series
    y_int  = best_model.confidence_intervals([x1_series,x2_series,x3_series].T, width=.95)

    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"), constrained_layout=1)

    # # Create the scatter plot for X and Y
    # plt.scatter(x_values, y_values, s=0.5, facecolors='none', edgecolors='blue',  alpha=0.5, label='data points')

    # # Plot the line for X_predict and Y_predict
    # plt.plot_surface(x1_series, x2_series, x3_series, color=y_pred, label='Predicted line')
    # plt.fill_between(x_series,y_int[:,1],y_int[:,0], color='red', edgecolor="none", alpha=0.1) #  .
    ax.scatter(x1_series, x2_series, x3_series, c='blue', marker='o', s=50)
    ax.plot_surface(x1_series, x2_series, x3_series, facecolors=plt.cm.plasma(norm(y_pred)))
    m = mp.cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
    plt.colorbar(m)

    # Add labels and title
    plt.xlabel('VPD')
    plt.ylabel('Tair')
    plt.zlabel('EF')
    plt.title('Check the GAM fitted curve')
    # plt.xlim(0, 10)  # Set x-axis limits
    # plt.ylim(0, 800)  # Set y-axis limits

    # Add legend
    plt.legend()

    plt.savefig(f'./plots/check_{var_name}_{model_out_name}_GAM_fitted_curve_{dist_type}.png',dpi=600)

    return


if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    PLUMBER2_path  = "/g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2/nc_files/"

    site_names, IGBP_types, clim_types, model_names = load_default_list()

    # ======================= Default setting (dont change) =======================
    var_name       = 'Qle'       #'TVeg'
    time_scale     = 'hourly'   #'daily'
    selected_by    = 'EF_model' # 'EF_model'
                                # 'EF_obs'
    standardize    = None       # 'None'
                                # 'STD_LAI'
                                # 'STD_annual_obs'
                                # 'STD_monthly_obs'
                                # 'STD_monthly_model'
                                # 'STD_daily_obs'
    LAI_range      = None
    veg_fraction   = None   #[0.7,1]

    clarify_site   = {'opt': True,
                     'remove_site': ['AU-Rig','AU-Rob','AU-Whr','CA-NS1','CA-NS2','CA-NS4','CA-NS5','CA-NS6',
                     'CA-NS7','CA-SF1','CA-SF2','CA-SF3','RU-Che','RU-Zot','UK-PL3','US-SP1']}
    models_calc_LAI= ['ORC2_r6593','ORC2_r6593_CO2','ORC3_r7245_NEE','ORC3_r8120','GFDL','SDGVM','QUINCY','NoahMPv401']

    day_time       = False  # False for daily
                            # True for half-hour or hourly

    if time_scale == 'hourly':
        day_time   = True

    energy_cor     = False
    if var_name == 'NEE':
        energy_cor = False

    # Set regions/country
    country_code   = None#'AU'
    if country_code != None:
        site_names = load_sites_in_country_list(country_code)

    # ====================== Custom setting ========================
    var_name       = 'Qle'
    uncertain_type = 'UCRTN_bootstrap'
    selected_by    = 'EF_model'
    method         = 'CRV_fit_GAM_complex'
    dist_type      = 'Gamma' # None #'Linear' #'Poisson' # 'Gamma'

    # 0 < EF < 0.2
    bounds         = [0.8,1.0] #30
    folder_name, file_message = decide_filename(day_time=day_time, energy_cor=energy_cor, time_scale=time_scale,
                                                standardize=standardize, country_code=country_code,
                                                selected_by=selected_by, bounds=bounds, veg_fraction=veg_fraction,
                                                LAI_range=LAI_range, clarify_site=clarify_site) #
    file_input     = 'raw_data_'+var_name+'_VPD'+file_message+'.csv'
    write_var_VPD_parallel(var_name, site_names, file_input, PLUMBER2_path, selected_by=selected_by,
                bounds=bounds, day_time=day_time, clarify_site=clarify_site,
                standardize=standardize, time_scale=time_scale, uncertain_type=uncertain_type,
                models_calc_LAI=models_calc_LAI, veg_fraction=veg_fraction, LAI_range=LAI_range,
                country_code=country_code, energy_cor=energy_cor, method=method, dist_type=dist_type)
    gc.collect()

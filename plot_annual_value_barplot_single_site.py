'''

'''

#==============================================

import os
import pandas as pd
from netCDF4 import Dataset
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.interpolate import griddata
import matplotlib.ticker as mticker
from multiprocessing import Pool
from PLUMBER2_GPP_common_utils import *

def get_annual_value(var_name, site_name, model_in, met=False, grow_season=0):

    '''
    Get all site annual mean NEE, GPP, RS
    '''

    if grow_season == 0: # annual value
        message = 'annual_value'
    elif grow_season == 1: # growing season
        message = 'annual_growing_season'
    elif grow_season == -1: # none growing season
        message = 'annual_non_growing_season'

    if met == True:
        df_out = pd.read_csv(f'./txt/{var_name}_{message}/{var_name}_{message}_{model_in}_{site_name}.csv')
    else:
        df_out = pd.read_csv(f'./txt/{var_name}_{message}/{var_name}_{message}_{model_in}_{site_name}.csv', usecols=[var_name])

    df_out = df_out.rename(columns={var_name: f'{var_name}_{model_in}'})

    return df_out

def get_annual_value_for_plot_all(var_name, site_name, model_in, met=False, grow_season=0):

    '''
    Get all site annual mean NEE, GPP, RS
    '''

    if grow_season == 0: # annual value

        if met == True:
            df_out = pd.read_csv(f'./txt/{var_name}_annual_value/{var_name}_annual_value_{model_in}_{site_name}.csv')
            df_out.loc[:, var_name] = df_out[var_name]/365
        else:
            df_out = pd.read_csv(f'./txt/{var_name}_annual_value/{var_name}_annual_value_{model_in}_{site_name}.csv', usecols=[var_name])
            df_out.loc[:, var_name] = df_out[var_name]/365

    elif grow_season == 1: # growing season
        if met == True:
            df_out = pd.read_csv(f'./txt/{var_name}_annual_growing_season/{var_name}_annual_growing_season_{model_in}_{site_name}.csv')
        else:
            df_out = pd.read_csv(f'./txt/{var_name}_annual_growing_season/{var_name}_annual_growing_season_{model_in}_{site_name}.csv', usecols=[var_name])
    elif grow_season == -1: # none growing season
        if met == True:
            df_out = pd.read_csv(f'./txt/{var_name}_annual_non_growing_season/{var_name}_annual_non_growing_season_{model_in}_{site_name}.csv')
        else:
            df_out = pd.read_csv(f'./txt/{var_name}_annual_non_growing_season/{var_name}_annual_non_growing_season_{model_in}_{site_name}.csv', usecols=[var_name])

    df_out = df_out.rename(columns={var_name: f'{var_name}_{model_in}'})

    return df_out

def plot_annual_value_barplot_single_site(var_name, site_name, grow_season=0):

    secondly_to_annually = 3600*24*365.

    PLUMBER2_path_site = "/srv/ccrc/LandAP/z5218916/script/PLUMBER2/LSM_GPP_PLUMBER2/nc_files/AR-SLu.nc"
    f                  = nc.Dataset(PLUMBER2_path_site, mode='r')
    model_list         = f.variables[f'{var_name}_models'][:]
    model_list         = model_list.tolist()
    model_list.append('obs')
    # print(model_list)

    # Loop through models in model_list
    for i, model_in in enumerate(model_list):
        if i == 0:
            df     = get_annual_value(var_name, site_name, model_in, met=True, grow_season=grow_season)
        else:
            df_tmp = get_annual_value(var_name, site_name, model_in, met=False, grow_season=grow_season)
            df     = pd.concat([df, df_tmp], axis=1)

    if grow_season == 0:
        df.loc[:, 'Precip'] = df['Precip'] * secondly_to_annually

    ##### Plotting #####
    met_vars  = ['Precip', 'Tair', 'VPD', 'LAI', 'SWdown', 'SMtop1m']  # Add 'CO2' if needed

    # Set up the figure with subplots
    fig, axes = plt.subplots(nrows=len(met_vars) + 1, ncols=1, figsize=(10, 20), sharex=True)

    # --- First Panel: NEE box-and-whisker plot with model values as dots ---
    x = np.sort(df['year'].unique()) #np.arange(min(df['year']), max(df['year']) + 1)  # Numeric values for the years
    print(x)

    # Prepare boxplot data: group NEE values by year for all models
    boxplot_data  = []
    years         = df['year'].unique()

    for year in years:
        # Collect all NEE values for this year across all models
        year_data = []
        for model_in in model_list[:-1]:
            model_data = df[df['year'] == year][f'{var_name}_{model_in}'].dropna()
            year_data.extend(model_data)  # Combine values for this year
        boxplot_data.append(year_data)  # Add combined year data

    try:
        # Create the box plot
        axes[0].boxplot(boxplot_data, positions=years, widths=0.5, patch_artist=True,
                        showfliers=False,
                        boxprops=dict(facecolor='none', edgecolor='black'))

        axes[0].axhline(y=0, ls='dashed', color='gray')
    except Exception as e:
        print('An error occurred:', e)
        print(f'Length of years: {len(years)}, Length of boxplot_data: {len(boxplot_data)}')
        print('Years:', years)
        print('Boxplot data:', boxplot_data)
        # Gracefully handle or log the error without using exit
        return  # or log the error and continue program execution
    # Add scatter points for individual model values
    for i, model_in in enumerate(model_list[:-1]):

        # Check if the length of x matches the length of the model data
        if len(x) != len(df[f'{var_name}_{model_in}']):

            # Print error details
            print(f"Error for model: {model_in}, Site: {site_name}, "
                  f"Length of x: {len(x)}, x:{x}, length of df[f'{var_name}_{model_in}']: {len(df[f'{var_name}_{model_in}'])}, df[f'{var_name}_{model_in}']: {df[f'{var_name}_{model_in}']}")

            # Raise an error to stop the entire program
            raise ValueError(f"Mismatched lengths for model: {model_in} in site: {site_name}")

        axes[0].scatter(x, df[f'{var_name}_{model_in}'], edgecolor=model_colors[model_in],facecolor='none', label=model_in, alpha=0.9)

    axes[0].scatter(x, df[f'{var_name}_obs'], edgecolor='gray',facecolor='gray', label='obs', alpha=0.9)

    # Customize first panel
    axes[0].set_ylabel(f'{var_name}')
    axes[0].set_title(f'{var_name} Model Comparison')
    axes[0].legend(loc='best', fontsize=8, ncol=6, frameon=False)

    # Set x-ticks and limits
    axes[0].set_xticks(x)  # Set x-ticks to the range of years
    axes[0].set_xticklabels(x.astype(int), rotation=45)  # Ensure labels match the years
    axes[0].set_xlim(x[0] - 0.5, x[-1] + 0.5)  # Set limits based on years
    # axes[0].set_ylim(-2000, 4000)  # Set limits based on years

    # --- Remaining Panels ---
    for i, var in enumerate(met_vars, 1):  # 1 means starting at 1 because 0 is used by NEE
        axes[i].bar(x, df[var], width=0.6, color='b', label=var)  # Adjust bar width for consistency
        axes[i].set_ylabel(var)
        # axes[i].legend(loc='best')

    # Set x-axis label
    axes[-1].set_xlabel('Year')

    # Adjust layout for better display
    plt.tight_layout()
    plt.show()
    if grow_season == 1:
        plt.savefig(f'./plots/Barplot_annual_sites/barplot_yearly_grow_season_changes_{var_name}_{site_name}.png', dpi=100)
    elif grow_season == 0:
        plt.savefig(f'./plots/Barplot_annual_sites/barplot_yearly_changes_{var_name}_{site_name}.png', dpi=100)
    elif grow_season == -1:
        plt.savefig(f'./plots/Barplot_annual_sites/barplot_yearly_non_grow_season_changes_{var_name}_{site_name}.png', dpi=100)

    return

def plot_all_barplot_single_site(var_name, site_name):

    secondly_to_daily  = 3600*24.

    PLUMBER2_path_site = "/srv/ccrc/LandAP/z5218916/script/PLUMBER2/LSM_GPP_PLUMBER2/nc_files/AR-SLu.nc"
    f                  = nc.Dataset(PLUMBER2_path_site, mode='r')
    model_list         = f.variables[f'{var_name}_models'][:]
    model_list         = model_list.tolist()
    model_list.append('obs')
    # print(model_list)

    # Loop through models in model_list
    for i, model_in in enumerate(model_list):
        if i == 0:
            df_annual       = get_annual_value_for_plot_all(var_name, site_name, model_in, met=True, grow_season=0)
            df_grow         = get_annual_value_for_plot_all(var_name, site_name, model_in, met=True, grow_season=1)
            df_non_grow     = get_annual_value_for_plot_all(var_name, site_name, model_in, met=True, grow_season=-1)
        else:
            df_annual_tmp   = get_annual_value_for_plot_all(var_name, site_name, model_in, met=False, grow_season=0)
            df_grow_tmp     = get_annual_value_for_plot_all(var_name, site_name, model_in, met=False, grow_season=1)
            df_non_grow_tmp = get_annual_value_for_plot_all(var_name, site_name, model_in, met=False, grow_season=-1)
            df_annual       = pd.concat([df_annual, df_annual_tmp], axis=1)
            df_grow         = pd.concat([df_grow, df_grow_tmp], axis=1)
            df_non_grow     = pd.concat([df_non_grow, df_non_grow_tmp], axis=1)

    df_annual.loc[:, 'Precip']   = df_annual['Precip'] * secondly_to_daily
    df_grow.loc[:, 'Precip']     = df_grow['Precip']
    df_non_grow.loc[:, 'Precip'] = df_non_grow['Precip']

    ##### Plotting #####
    met_vars  = ['Precip', 'Tair', 'VPD', 'LAI', 'SWdown', 'SMtop1m']  # Add 'CO2' if needed

    # Create subplots: len(met_vars) + 1 (for the NEE, GPP, RS panel)
    num_panels = len(met_vars) + 1

    # Define the relative heights: the first panel gets a ratio of 5, the others get 3
    heights    = [5] + [3] * len(met_vars)

    fig, axes  = plt.subplots(nrows=num_panels, ncols=1, figsize=(10, 2*num_panels),
                          sharex=True, gridspec_kw={'height_ratios': heights})

    # --- First Panel: NEE box-and-whisker plot with model values as dots ---
    x   = np.sort(df_annual['year'].unique())

    # Prepare boxplot data: group NEE values by year for all models
    boxplot_annual   = []
    boxplot_grow     = []
    boxplot_non_grow = []

    years         = df_annual['year'].unique()

    year_avail_annual   = []
    year_avail_grow     = []
    year_avail_non_grow = []

    for year in years:
        # Collect all NEE values for this year across all models
        year_annual   = []
        year_grow     = []
        year_non_grow = []

        for model_in in model_list[:-1]:
            model_annual   = df_annual[df_annual['year'] == year][f'{var_name}_{model_in}'].dropna()
            year_annual.extend(model_annual)  # Combine values for this year
            if ~np.all(np.isnan(model_annual)):
                year_avail_annual.append(year)

            model_grow     = df_grow[df_grow['year'] == year][f'{var_name}_{model_in}'].dropna()
            year_grow.extend(model_grow)  # Combine values for this year
            if ~np.all(np.isnan(model_grow)):
                year_avail_grow.append(year)

            model_non_grow = df_non_grow[df_non_grow['year'] == year][f'{var_name}_{model_in}'].dropna()
            year_non_grow.extend(model_non_grow)  # Combine values for this year
            if len(model_non_grow) > 0:
                if ~np.all(np.isnan(model_non_grow)):
                    year_avail_non_grow.append(year)

        boxplot_annual.append(year_annual)  # Add combined year data
        boxplot_grow.append(year_grow)  # Add combined year data
        boxplot_non_grow.append(year_non_grow)  # Add combined year data

    year_avail_annual   = np.sort(np.unique(year_avail_annual))
    year_avail_grow     = np.sort(np.unique(year_avail_grow))
    year_avail_non_grow = np.sort(np.unique(year_avail_non_grow))

    width = 0.3

    # Create the box plot
    axes[0].boxplot(boxplot_annual, positions=x-width,widths=width, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor='none', edgecolor='black', linewidth=1.2),
                    whiskerprops=dict(color='black', linewidth=1.2),
                    capprops=dict(color='black', linewidth=1.2),
                    medianprops=dict(color='black', linewidth=1.2) )

    axes[0].boxplot(boxplot_grow, positions=x,widths=width, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor='none', edgecolor='darkgreen',linewidth=1.2),
                    whiskerprops=dict(color='darkgreen', linewidth=1.2),
                    capprops=dict(color='darkgreen', linewidth=1.2),
                    medianprops=dict(color='darkgreen', linewidth=1.2))

    axes[0].boxplot(boxplot_non_grow, positions=x+width,widths=width, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor='none', edgecolor='brown',linewidth=1.2),
                    whiskerprops=dict(color='brown', linewidth=1.2),
                    capprops=dict(color='brown', linewidth=1.2),
                    medianprops=dict(color='brown', linewidth=1.2))

    axes[0].axhline(y=0, ls='dashed', color='gray')

    # Add scatter points for individual model values
    for i, model_in in enumerate(model_list[:-1]):

        annual_exist, grow_exist, non_grow_exist = True, True, True

        # Check if the length of x matches the length of the model data
        if (   (len(x) != len(df_annual[f'{var_name}_{model_in}']))
            or (len(x) != len(df_grow[f'{var_name}_{model_in}']))
            or (len(x) != len(df_non_grow[f'{var_name}_{model_in}'])) ):

            # Print error details
            print(f"Error for model: {model_in}, Site: {site_name}, "
                  f"Length of x: {len(x)}, x:{x}, length of df_annual[f'{var_name}_{model_in}']: {len(df_annual[f'{var_name}_{model_in}'])}",
                                                f"length of df_grow[f'{var_name}_{model_in}']: {len(df_grow[f'{var_name}_{model_in}'])}",
                                                f"length of df_non_grow[f'{var_name}_{model_in}']: {len(df_non_grow[f'{var_name}_{model_in}'])}")
            print( 'year_avail_annual, year_avail_grow, year_avail_non_grow', year_avail_annual, year_avail_grow, year_avail_non_grow)

#             # Raise an error to stop the entire program
#             raise ValueError(f"Mismatched lengths for model: {model_in} in site: {site_name}")

            if len(df_annual[f'{var_name}_{model_in}']) == 0:
                annual_exist   = False
            if len(df_grow[f'{var_name}_{model_in}']) == 0:
                grow_exist     = False
            if len(df_non_grow[f'{var_name}_{model_in}']) == 0:
                non_grow_exist = False

        if annual_exist:
            axes[0].scatter(year_avail_annual-width,   df_annual[f'{var_name}_{model_in}'],   edgecolor=model_colors[model_in], facecolor='none', label=model_in, alpha=0.9)
        if grow_exist:
            axes[0].scatter(year_avail_grow,           df_grow[f'{var_name}_{model_in}'],     edgecolor=model_colors[model_in], facecolor='none', alpha=0.9)
        if non_grow_exist:
            axes[0].scatter(year_avail_non_grow+width, df_non_grow[f'{var_name}_{model_in}'], edgecolor=model_colors[model_in], facecolor='none', alpha=0.9)

    if annual_exist:
        axes[0].scatter(year_avail_annual-width, df_annual[f'{var_name}_obs'], edgecolor='none',facecolor='black', label='obs', alpha=0.9)
    if grow_exist:
        axes[0].scatter(year_avail_grow,       df_grow[f'{var_name}_obs'], edgecolor='none',facecolor='black',  alpha=0.9)
    if non_grow_exist:
        axes[0].scatter(year_avail_non_grow+width, df_non_grow[f'{var_name}_obs'], edgecolor='none',facecolor='black',  alpha=0.9)

    # Customize first panel
    axes[0].set_ylabel(var_name)
    axes[0].set_title(f'{var_name} Model Comparison')
    axes[0].legend(loc='best', fontsize=8, ncol=6, frameon=False, columnspacing=0.5)

    # Set x-ticks and limits
    axes[0].set_xticks(x)  # Set x-ticks to the range of years
    axes[0].set_xticklabels(x.astype(int), rotation=45)  # Ensure labels match the years
    axes[0].set_xlim(x[0] - 0.5, x[-1] + 0.5)  # Set limits based on years
    # axes[0].set_ylim(-6000, 4000)  # Set limits based on years

    # --- Remaining Panels ---
    for i, var in enumerate(met_vars, 1):  # 1 means starting at 1 because 0 is used by NEE
        if annual_exist:
            axes[i].bar(year_avail_annual-width,   df_annual[var],   width=width, color='dodgerblue',  label=var)  # Adjust bar width for consistency
        if grow_exist:
            axes[i].bar(year_avail_grow,           df_grow[var],     width=width, color='seagreen', label=var)  # Adjust bar width for consistency
        if non_grow_exist:
            axes[i].bar(year_avail_non_grow+width, df_non_grow[var], width=width, color='coral', label=var)  # Adjust bar width for consistency

        axes[i].set_ylabel(var)
        # axes[i].legend(loc='best')

    # Set x-axis label
    axes[-1].set_xlabel('Year')

    # Adjust layout for better display
    plt.tight_layout()
    plt.show()


    plt.savefig(f'./plots/Barplot_annual_sites/barplot_annual_vs_grow_non_grow_{var_name}_{site_name}.png', dpi=100)

    return


if __name__ == "__main__":

    # Path of PLUMBER 2 dataset
    site_names, IGBP_types, clim_types, model_names \
                 = load_default_list()
    model_colors = set_model_colors_with_ML()
    remove_site  = get_removed_site_names()

    # Calculate remaining sites
    set_site_names  = set(site_names)
    set_remove_site = set(remove_site)
    remain_sites    = set_site_names - set_remove_site
    remain_sites    = list(remain_sites)

    models_calc_LAI = ['ORC2_r6593','ORC2_r6593_CO2','ORC3_r7245_NEE','ORC3_r8120','GFDL','SDGVM','QUINCY','NoahMPv401']
    model_LAI_names = {'ORC2_r6593':'lai','ORC2_r6593_CO2':'lai','ORC3_r7245_NEE':'lai','ORC3_r8120':'lai',
                       'GFDL':'lai', 'SDGVM':'lai','QUINCY':'LAI','NoahMPv401':'LAI'}

    # grow_season = -1
    # var_name     = 'NEE'
    # with Pool() as pool:
    #     pool.starmap(plot_annual_value_barplot_single_site,
    #                 [(var_name, site_name, grow_season) for site_name in remain_sites])
    #
    # var_name     = 'GPP'
    # with Pool() as pool:
    #     pool.starmap(plot_annual_value_barplot_single_site,
    #                 [(var_name, site_name, grow_season) for site_name in remain_sites])


    var_name     = 'NEE'
    with Pool() as pool:
        pool.starmap(plot_all_barplot_single_site,
                    [(var_name, site_name) for site_name in remain_sites])

    var_name     = 'GPP'
    with Pool() as pool:
        pool.starmap(plot_all_barplot_single_site,
                    [(var_name, site_name) for site_name in remain_sites])

"""
===============================================
   Extreme Events Diagnostics - Plotting
===============================================

This script loads the processed outputs from the
extreme-event computation pipeline and generates:

    • Spatial maps (days, intensity, threshold, mean state)
    • Temporal time series (monthly intensity, frequency)

It supports three event types (but is easily extendable):
    - MHW (Marine Heatwaves) - variable: to
    - LOX / DEO (Deoxygenation extremes) - variable: o2
    - OAX (Ocean Acidification extremes) - variable: pH (hi)

The workflow is designed for:
    - High-resolution ocean model output (4D: t,z,lat,lon)
    - Multiple depths
    - Publication-quality figures
    - Full automation across variables

Input: 
    NetCDFs produced by `calculate_extremes_computation.sh`
Output:
    PNG figures stored under:
    ./extremes_out/<event>_<var>/depth<depth>/

Author:
    Cristina Radin (2025)
    cristina.radin@uni-hamburg.de
    AI4PEX (https://ai4pex.org/)
    University of Hamburg

Notes:
    - Colorbar limits are automatically determined
      using percentile-based robust scaling.
    - Land is masked using the "max value" trick
      produced by the HPC script.
    - Time series include padded ylim for clarity.

================================================
"""

import xarray as xr
import matplotlib.pyplot as plt
import os
import re
import subprocess
import tempfile
import glob
from datetime import datetime
import numpy as np
import netCDF4 as nc


# -------------------------------
# SET PARAMETERS
# -------------------------------

variables = [
    {"name": "to", "name2":"to", "event_type": "mhw",  "event_name": "MHW", "label": "Temperature (ºC)"}, 
    {"name": "o2", "name2":"o2",  "event_type": "deo",  "event_name": "LOX", "label": "Oxygen (kmol O2 m-3)"},  
    {"name": "hi", "name2":"pH",  "event_type": "oax", "event_name": "OAX", "label": "pH"}  
]


idx_var = variables[0]     # 0: to-MHW, 1: o2-DEO, 2: hi-OAX
var = idx_var["name2"]
var_path = idx_var["name"]
extreme = idx_var["event_type"]
extreme_label = idx_var["event_name"]
extreme_label_full = idx_var["label"]
depth = 8 #8, 51, 104, 186.5, 489 m
 


path = '/***/extremes_out/' + extreme + '_'+ var_path + '/depth' + str(depth) + '/tmp/'
path_out = '/****/extremes_out/' + extreme + '_'+ var_path + '/depth' + str(depth) + '/' 



# -------------------------------
# DATA LOADING
# -------------------------------

# 1) Days (total, monthly, yearly)
ds_days_total = xr.open_dataset(path +'days_total.nc')
days_total = ds_days_total[f'{var}'] 

ds_days_yearly = xr.open_dataset(path+'days_yearly.nc')
days_yearly = ds_days_yearly[f'{var}'] 
days_mean=days_yearly.mean(dim='time')

ds_days_monthly = xr.open_dataset(path+'days_monthly.nc')
days_monthly = ds_days_monthly[f'{var}']

# 2) Yearly and monthly Max Intensity
ds_intensity_max_yearly = xr.open_dataset(path+'intensity_max_yearly.nc')
intensity_max_yearly = ds_intensity_max_yearly[f'{var}']
intensity_max_yearly_mean = intensity_max_yearly.mean(dim='time')

ds_intensity_max_monthly = xr.open_dataset(path+'intensity_max_monthly.nc')
intensity_max_monthly = ds_intensity_max_monthly[f'{var}']

# 3) Threshold
ds_thres_monthly = xr.open_dataset(path+'threshold_monmean.nc')
thres_monthly = ds_thres_monthly[f'{var}']
thres_mean = thres_monthly.mean(dim='time')

# 4) Variable mean
ds_var_monthly = xr.open_dataset(path+'all_data_monmean.nc')
data_var_monthly = ds_var_monthly[f'{var}']
var_mean = data_var_monthly.mean(dim='time')



# -------------------------------
# MASK LAND 
# -------------------------------

max_val = float(days_total.max())

land_mask = xr.where(days_total.isel(time=0) != max_val, 1, np.nan)  

days_total_masked = np.where(days_total == max_val, np.nan, days_total)
days_yearly_masked = days_mean * land_mask
intensity_max_yearly_mean_masked = intensity_max_yearly_mean * land_mask
thres_mean_masked = thres_mean * land_mask
var_mean_masked = var_mean * land_mask
intensity_max_monthly_masked= intensity_max_monthly * land_mask
days_monthly_masked= days_monthly * land_mask



# -------------------------------
# SPATIAL PLOTS
# -------------------------------

lats = ds_days_total['lat'][:]
lons = ds_days_total['lon'][:]

lon2d, lat2d = np.meshgrid(lons, lats)

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
plot_title = f"Spatial Patterns of {extreme_label} Metrics for {var} at Depth {depth} m"

fig.suptitle(plot_title, fontsize=16)

# --- Panel 1: DAYS ---
im1 = axs[0, 0].pcolormesh(lon2d, lat2d, days_yearly_masked.squeeze(),
                        shading='auto', vmin=np.nanpercentile(days_yearly_masked,10), vmax=np.nanpercentile(days_yearly_masked, 90))
axs[0, 0].set_title("Annual days")
axs[0, 0].set_xlabel("Longitude")
axs[0, 0].set_ylabel("Latitude")
plt.colorbar(im1, ax=axs[0, 0], label="Days (ref 1985–2014)")

# --- Panel 2: INTENSITY ---
im2 = axs[0, 1].pcolormesh(lon2d, lat2d, intensity_max_yearly_mean_masked.squeeze(),
                        shading='auto', vmin=np.nanpercentile(intensity_max_yearly_mean_masked, 5), vmax=np.nanpercentile(intensity_max_yearly_mean_masked, 95))
axs[0, 1].set_title("Annual Max Intensity")
axs[0, 1].set_xlabel("Longitude")
axs[0, 1].set_ylabel("Latitude")
plt.colorbar(im2, ax=axs[0, 1], label=f"{idx_var['label']}")


# --- Panel 3: Threshold ---
im3 = axs[1, 0].pcolormesh(lon2d, lat2d, thres_mean_masked.squeeze(),
                        shading='auto', vmin=np.nanpercentile(var_mean_masked, 5), vmax=np.nanpercentile(var_mean_masked, 95))
axs[1, 0].set_title("Threshold")
axs[1, 0].set_xlabel("Longitude")
axs[1, 0].set_ylabel("Latitude")
plt.colorbar(im3, ax=axs[1, 0], label=f"{idx_var['label']}")


# --- Panel 4: Variable Mean ---
im4 = axs[1, 1].pcolormesh(lon2d, lat2d, var_mean_masked.squeeze(),
                        shading='auto', vmin=np.nanpercentile(var_mean_masked, 5), vmax=np.nanpercentile(var_mean_masked, 95))
axs[1, 1].set_title(f"{var} Mean")
axs[1, 1].set_xlabel("Longitude")
axs[1, 1].set_ylabel("Latitude")
plt.colorbar(im4, ax=axs[1, 1], label=f" Mean {var}")
plt.tight_layout()

plt.savefig(path_out + "spatialpatterns.png", dpi=300)
plt.show()



# -------------------------------
# TEMPORAL PLOTS
# -------------------------------

# --- Intensity (monthly max) ---

intensity_max_monthly_masked_mean = intensity_max_monthly_masked.mean(dim=['lat','lon'])


low, high = np.nanmin(intensity_max_monthly_masked_mean), np.nanmax(intensity_max_monthly_masked_mean)
padding = 0.05 * (high - low)  

plt.figure(figsize=(12,4))
plt.plot(intensity_max_monthly_masked_mean['time'], intensity_max_monthly_masked_mean, label='Monthly Max Intensity', color='red')
plt.ylim(low - padding, high + padding)
plt.ylabel("Intensity")
plt.xlabel('Time')
plt.title(f'{extreme_label} Monthly Intensity (Depth {depth} m)')
plt.legend()
plt.grid()
plt.savefig(path_out + "time_series_intensity.png", dpi=300)   
plt.show()


# --- Frequency (monthly days) ---

days_monthly_masked_mean = days_monthly_masked.mean(dim=['lat','lon'])

low, high = np.nanmin(days_monthly_masked_mean), np.nanmax(days_monthly_masked_mean)
padding = 0.05 * (high - low)  

plt.figure(figsize=(12,4))
plt.plot(days_monthly_masked_mean['time'], days_monthly_masked_mean, label='Monthly Days', color='green')
plt.ylim(low - padding, high + padding)
plt.ylabel("Frequency (days)")
plt.xlabel('Time')
plt.title(f'{extreme_label} Monthly Frequency (Depth {depth} m)')
plt.legend()
plt.grid()
plt.savefig(path_out + "time_series_frequency.png", dpi=300)   
plt.show()

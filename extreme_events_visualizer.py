"""
===============================================
   Extreme Events - Diagnostic Visualization
===============================================

This script generates diagnostic plots for extreme event analysis.
It loads processed data from the extreme-event computation pipeline and produces:

Workflow:
    1. Load land mask and data (days, intensity, threshold, mean state)
    2. Apply land mask for ocean-only processing
    3. Generate time series plots (frequency and intensity)
    4. Generate spatial pattern plots with consistent colormaps
    5. Save all figures to output directory


Input: 
    Processed NetCDFs from:
    ./extremes_out_final/<event>_<var>/depth<depth>/tmp/

Output:
    PNG figures stored under:
    ./extremes_out_final/<event>_<var>/depth<depth>/

    Generated plots include:
    1. Temporal time series:
        - Monthly frequency of extreme days
        - Monthly maximum intensity

    2. Spatial patterns (2x2 grid):
        - Annual extreme days
        - Annual maximum intensity  
        - Climatological threshold
        - Mean state of the variable


Event types supported:
    - MHW (Marine Heatwaves) - variable: to
    - LOX/DEO (Low Oxygen extremes) - variable: o2  
    - OAX (Ocean Acidification extremes) - variable: hi


Author:
    Cristina Radin (2025)
    cristina.radin@uni-hamburg.de
    AI4PEX (https://ai4pex.org/)
    University of Hamburg

Notes:
    - Uses consistent colormaps across all spatial plots
    - Land is masked as gray areas
    - Time series include adaptive y-axis limits
    - Spatial plots use 5th-95th percentiles for color scaling

================================================
"""

"""
===============================================
   Extreme Events - Diagnostic Visualization
===============================================

"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
import matplotlib.dates as mdates
from xarray.coders import CFDatetimeCoder
import sys


LAND_MASK_3D_PATH = '___________/land_mask_3d.nc'



# Event type configuration

VARIABLES_CONFIG = {
    'to': {
        'name': 'to',
        'name2': 'to',
        'event_type': 'mhw',
        'event_name': 'MHW',
        'label': 'Temperature (ºC)',
        'cmap': 'RdBu_r',
        'units': 'days'
    },
    'o2': {
        'name': 'o2',
        'name2': 'o2',
        'event_type': 'deo',
        'event_name': 'LOX',
        'label': 'Oxygen (μmol O2 m-3)',
        'cmap': 'YlOrRd',
        'units': 'days'
    },
    'hi': {
        'name': 'hi',
        'name2': 'hi',
        'event_type': 'oax',
        'event_name': 'OAX',
        'label': '[H+] (μmol m-3)',
        'cmap': 'YlOrRd',  
        'units': 'days'
    }
}


def load_global_land_mask():
    try:
        return xr.open_dataarray(LAND_MASK_3D_PATH)
    except:
        mask_ds = xr.open_dataset(LAND_MASK_3D_PATH)
        return mask_ds['land_mask']

def get_land_mask_for_depth(land_mask_3d, depth_value):
    if isinstance(depth_value, str):
        depth_value = float(depth_value)
    
    mask_2d = land_mask_3d.sel(depth=depth_value, method='nearest')
    print(f"  Mask: {mask_2d.depth.values}m (requested: {depth_value}m)")
    return mask_2d



# Read command-line arguments
if len(sys.argv) >= 3:
    var_from_bash = sys.argv[1]  # "to", "o2", "hi"
    depth_arg = sys.argv[2]      # "8", "51", "186.5"
else:
    var_from_bash = "to"
    depth_arg = "8"

# Format depth for file paths
try:
    depth_value = float(depth_arg)
except:
    depth_value = 8.0

depth_for_path = depth_arg
if depth_for_path.endswith('.0'):
    depth_for_path = depth_for_path[:-2]


# Find variable configuration
if var_from_bash not in VARIABLES_CONFIG:
    raise ValueError(f"Variable '{var_from_bash}' no valid. Use: to, o2, hi")

config = VARIABLES_CONFIG[var_from_bash]
var = config['name2']
var_path = config['name']
extreme = config['event_type']
extreme_label = config['event_name']
extreme_label_full = config['label']
cmap_to_use = config['cmap']



# Build paths
path = f'___________/extremes_out_final/{extreme}_{var_path}/depth{depth_for_path}/tmp/'
path_out = f'___________/extremes_out_final/{extreme}_{var_path}/depth{depth_for_path}/'

import os
os.makedirs(path_out, exist_ok=True)

print("="*70)
print(f"Extreme event and variable selected: {extreme_label} ({var})")
print(f"Depth: {depth_arg}m")
print("="*70)


print("\n1. LOADING LAND MASK...")
land_mask_3d = load_global_land_mask()
land_mask = get_land_mask_for_depth(land_mask_3d, depth_value)

lats = land_mask['lat'].values
lons = land_mask['lon'].values
lon2d, lat2d = np.meshgrid(lons, lats)


print("\n2. LOADING DATA...")

# 1) Yearly and monthly days
ds_duration_yearly = xr.open_dataset(path + 'days_yearly.nc')
days_yearly = ds_duration_yearly['days_total']
days_yearly_mean = days_yearly.mean(dim="time")

ds_duration_monthly = xr.open_dataset(path + 'days_monthly.nc')
days_monthly = ds_duration_monthly['days_total']

# 2) Intensity data
ds_intensity_max_yearly = xr.open_dataset(path + 'intensity_max_yearly.nc')
intensity_max_yearly = ds_intensity_max_yearly[var]
intensity_max_yearly_mean = intensity_max_yearly.mean(dim='time')

ds_intensity_max_monthly = xr.open_dataset(path + 'intensity_max_monthly.nc')
intensity_max_monthly = ds_intensity_max_monthly[var]

# 3) Threshold data
time_coder = CFDatetimeCoder(use_cftime=True)
ds_thres_monthly = xr.open_dataset(path + 'threshold_monmean.nc', decode_times=time_coder)
thres_monthly = ds_thres_monthly[var]
thres_mean = thres_monthly.mean(dim='time')

# 4) Variable mean data
ds_all = xr.open_dataset(path + 'all_data_monmean.nc')
data_var_all = ds_all[var]
data_var_mean = data_var_all.mean(dim='time')


print("\n3. APPLYING LAND MASK...")

intensity_max_monthly_masked = intensity_max_monthly.values * land_mask.values
intensity_max_yearly_mean_masked = intensity_max_yearly_mean.values * land_mask.values
days_yearly_mean_masked = days_yearly_mean.values * land_mask.values
days_monthly_masked = days_monthly.values * land_mask.values
thres_mean_masked = thres_mean.values * land_mask.values
data_var_mean_masked = data_var_mean.values * land_mask.values

# Monthly and yearly means for time series
days_monthly_mean_masked = np.nanmean(days_monthly_masked, axis=(1, 2))
intensity_max_monthly_mean_masked = np.nanmean(intensity_max_monthly_masked, axis=(1, 2, 3))

# -------------------------------
# PLOT 1: Frequency (monthly days)
# -------------------------------
print("\n4. CREATING FREQUENCY PLOT...")

low, high = np.nanmin(days_monthly_mean_masked), np.nanmax(days_monthly_mean_masked)
padding = 0.05 * (high - low)

plt.figure(figsize=(12,4))
plt.plot(ds_duration_monthly['time'], days_monthly_mean_masked, label='Monthly Days', color='green')

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_minor_locator(mdates.YearLocator(1))
ax.grid(True, which='minor', alpha=0.3, linestyle=':')
ax.grid(True, which='major', alpha=0.5)

plt.ylim(low - padding, high + padding)
plt.ylabel("Frequency (days)")
plt.xlabel('Time')
plt.title(f'{extreme_label} Monthly Frequency (Depth {depth_arg} m)')
plt.legend()
plt.tight_layout()

plt.savefig(path_out + "time_series_frequency.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: time_series_frequency.png")


# -------------------------------
# PLOT 2: Intensity (monthly max)
# -------------------------------
print("\n5. CREATING INTENSITY PLOT...")

low, high = np.nanmin(intensity_max_monthly_mean_masked), np.nanmax(intensity_max_monthly_mean_masked)
padding = 0.05 * (high - low)

plt.figure(figsize=(12, 4))
plt.plot(ds_duration_monthly['time'], 
         intensity_max_monthly_mean_masked, 
         label='Monthly Max Intensity', color='red', linewidth=1)

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_minor_locator(mdates.YearLocator(1))
ax.grid(True, which='major', alpha=0.4, linestyle='-', linewidth=0.5)
ax.grid(True, which='minor', alpha=0.2, linestyle=':', linewidth=0.3)

plt.ylim(low - padding, high + padding)
plt.ylabel("Intensity")
plt.xlabel('Year')
plt.title(f'{extreme_label} Monthly Intensity (Depth {depth_arg} m)')
plt.legend()
plt.tight_layout()

plt.savefig(path_out + "time_series_intensity.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: time_series_intensity.png")


# -------------------------------
# PLOT 3: Spatial patterns (2x2) 
# -------------------------------
print("\n6. CREATING SPATIAL PLOT...")

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
plot_title = f"Spatial Patterns of {extreme_label} Metrics for {var} at Depth {depth_arg} m"
fig.suptitle(plot_title, fontsize=16)

def create_masked_plot(ax, data, title, vmin=None, vmax=None, cmap=None, label=None):
    im = ax.pcolormesh(lon2d, lat2d, data.squeeze(),
                       shading='auto', 
                       vmin=vmin,
                       vmax=vmax,
                       cmap=cmap)
    
    im.cmap.set_bad('gray', alpha=0.7)
    
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    cbar = plt.colorbar(im, ax=ax)
    if label:
        cbar.set_label(label)
    
    return im

# Panel 1: DAYS 
im1 = create_masked_plot(axs[0, 0], 
                        days_yearly_mean_masked,
                        "Annual days",
                        vmin=np.nanpercentile(days_yearly_mean_masked, 10),
                        vmax=np.nanpercentile(days_yearly_mean_masked, 90),
                        cmap=cmap_to_use,
                        label="Days (ref 1985–2014)")

# Panel 2: INTENSITY 
im2 = create_masked_plot(axs[0, 1], 
                        intensity_max_yearly_mean_masked,
                        "Annual Max Intensity",
                        vmin=np.nanpercentile(intensity_max_yearly_mean_masked, 5),
                        vmax=np.nanpercentile(intensity_max_yearly_mean_masked, 95),
                        cmap=cmap_to_use,
                        label=extreme_label_full)

# Panel 3: Threshold 
im3 = create_masked_plot(axs[1, 0], 
                        thres_mean_masked,
                        "Threshold",
                        vmin=np.nanpercentile(data_var_mean_masked, 5),
                        vmax=np.nanpercentile(data_var_mean_masked, 95),
                        cmap=cmap_to_use,
                        label=extreme_label_full)

# Panel 4: Variable Mean 
im4 = create_masked_plot(axs[1, 1], 
                        data_var_mean_masked,
                        f"{var} Mean",
                        vmin=np.nanpercentile(data_var_mean_masked, 5),
                        vmax=np.nanpercentile(data_var_mean_masked, 95),
                        cmap=cmap_to_use,
                        label=f"Mean {var}")

plt.tight_layout()
plt.savefig(path_out + "spatialpatterns.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: spatialpatterns.png")

print("\n" + "="*70)

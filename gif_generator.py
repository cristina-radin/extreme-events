"""
===============================================
   Extreme Events - Diagnostic Visualization
===============================================

This script generates diagnostic plots for extreme event analysis.
It loads processed data from the extreme-event computation pipeline and produces:

    1. Temporal time series:
        - Monthly frequency of extreme days
        - Monthly maximum intensity

    2. Spatial patterns (2x2 grid):
        - Annual extreme days
        - Annual maximum intensity  
        - Climatological threshold
        - Mean state of the variable

    3. Individual spatial maps:
        - Annual extreme days only

Workflow:
    1. Load land mask and data (days, intensity, threshold, mean state)
    2. Apply land mask for ocean-only processing
    3. Generate time series plots (frequency and intensity)
    4. Generate spatial pattern plots with consistent colormaps
    5. Save all figures to output directory

Event types supported:
    - MHW (Marine Heatwaves) - variable: to
    - LOX/DEO (Low Oxygen extremes) - variable: o2  
    - OAX (Ocean Acidification extremes) - variable: hi

Input: 
    Processed NetCDFs from:
    ./extremes_out_final/<event>_<var>/depth<depth>/tmp/

Output:
    PNG figures stored under:
    ./extremes_out_final/<event>_<var>/depth<depth>/

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



import xarray as xr
import numpy as np
import glob
import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


# Global configuration
VARIABLES_CONFIG = {
    'to': {
        'name': 'to',
        'name2': 'to',
        'event_type': 'mhw',
        'event_name': 'MHW',
        'label': 'Temperature (ºC)',
        'cmap': 'RdBu_r',
        'units': 'days',
        'intensity_units': 'ºC'
    },
    'o2': {
        'name': 'o2',
        'name2': 'o2',
        'event_type': 'deo',
        'event_name': 'LOX',
        'label': 'Oxygen (μmol O2 m-3)',
        'cmap': 'YlOrRd',
        'units': 'days',
        'intensity_units': 'μmol O2 m-3'
    },
    'hi': {
        'name': 'hi',
        'name2': 'hi',
        'event_type': 'oax',
        'event_name': 'OAX',
        'label': '[H+] (μmol m-3)',
        'cmap': 'YlOrRd',
        'units': 'days',
        'intensity_units': 'μmol m-3'
    }
}

DEFAULT_LAND_MASK_PATH = '___________/land_mask_3d.nc'


def load_global_land_mask(mask_path):
    # Load 3D global land mask from NetCDF file.

    if mask_path is None or not os.path.exists(mask_path):
        print(f"Mask file not founded: {mask_path}")
        return None
    
    try:
        print(f"Loading land mask from: {mask_path}")
        ds = xr.open_dataset(mask_path)
        
        if 'land_mask' in ds:
            mask = ds['land_mask']
        else:
            mask = ds[list(ds.data_vars)[0]]
        
        print(f"  Mask loaded: dims={mask.dims}, shape={mask.shape}")
        return mask
    except Exception as e:
        print(f"Error loading mask: {e}")
        return None

def get_land_mask_for_depth(land_mask_3d, depth_value, target_lat=None, target_lon=None):
    # Extract 2D land mask for specific depth.

    if land_mask_3d is None:
        return None
    
    try:
        if isinstance(depth_value, str):
            try:
                depth_value = float(depth_value)
            except:
                pass
        
        if 'depth' in land_mask_3d.dims:
            depth_values = land_mask_3d.depth.values
            idx = np.abs(depth_values - depth_value).argmin()
            nearest_depth = depth_values[idx]
            
            mask_2d = land_mask_3d.sel(depth=nearest_depth, method='nearest')
            print(f"  Mask: {nearest_depth}m (requested: {depth_value}m)")
            
        elif 'z' in land_mask_3d.dims:
            mask_2d = land_mask_3d.sel(z=depth_value, method='nearest')
        else:
            mask_2d = land_mask_3d
        
        if target_lat is not None and target_lon is not None:
            if 'lat' in mask_2d.dims and 'lon' in mask_2d.dims:
                mask_2d = mask_2d.interp(lat=target_lat, lon=target_lon, method='nearest')
        
        return mask_2d
        
    except Exception as e:
        print(f"Error obtaining mask for depth {depth_value}: {e}")
        return None

def apply_land_mask(data_array, land_mask):
    # Apply land mask to data array.
    if land_mask is None:
        return data_array
    try:
        masked_data = data_array * land_mask.values
        
        masked_da = xr.DataArray(
            masked_data,
            dims=data_array.dims,
            coords=data_array.coords,
            attrs=data_array.attrs
        )
        
        masked_da.attrs['land_mask_applied'] = 1  
        
        return masked_da
        
    except Exception as e:
        print(f"Error applying mask: {e}")
        return data_array


# Combine data from different depths.

def combine_depth_data(var_name, base_path, land_mask_path=None, verbose=True):
    
    if var_name not in VARIABLES_CONFIG:
        raise ValueError(f"Variable: '{var_name}' no valid. Use: {list(VARIABLES_CONFIG.keys())}")
    
    config = VARIABLES_CONFIG[var_name]
    
    if verbose:
        print(f"="*60)
        print(f"PROCESSING: {config['event_name']} ({var_name})")
        print(f"="*60)
    
    land_mask_3d = None
    if land_mask_path:
        land_mask_3d = load_global_land_mask(land_mask_path)
    
    pattern = f"{config['event_type']}_{config['name']}/depth*/tmp/days_yearly.nc"
    search_path = os.path.join(base_path, pattern)
    
    if verbose:
        print(f"Searching for files in: {search_path}")
    

    all_files = sorted(glob.glob(search_path))
    
    if not all_files:
        raise FileNotFoundError(f"Files for {var_name} not found in {base_path}")
    
    if verbose:
        print(f"Found {len(all_files)} files")
        print("-"*40)
    

    depth_data = []
    depths = []
    
    # Process each file
    for file_path in all_files:
        for part in Path(file_path).parts:
            if part.startswith('depth'):
                depth_str = part.replace('depth', '')
                break
        
        try:
            depth_val = float(depth_str)
        except ValueError:
            depth_val = depth_str
        
        if verbose:
            print(f" Depth {depth_val}m: {file_path}")
        
        try:
            # Data loading
            ds = xr.open_dataset(file_path)
            
            if 'lat' in ds.dims and 'lon' in ds.dims:
                target_lat = ds.lat.values
                target_lon = ds.lon.values
            else:
                target_lat = target_lon = None
            
            days_yearly = ds['days_total']
            days_yearly_mean = days_yearly.mean(dim="time")
            
            if land_mask_3d is not None:
                if verbose:
                    print(f"    Applying land mask...")
                
                # Obtain mask for this depth
                land_mask_2d = get_land_mask_for_depth(
                    land_mask_3d, depth_val, target_lat, target_lon
                )
                
                if land_mask_2d is not None:
                    days_yearly_mean_masked = apply_land_mask(days_yearly_mean, land_mask_2d)
                    
                    days_yearly_mean = days_yearly_mean_masked
            
            depth_data.append(days_yearly_mean)
            depths.append(depth_val)
            
            ds.close()
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    
    # Sorter by depth (ascending)
    sorted_indices = np.argsort(depths)
    depths_sorted = [depths[i] for i in sorted_indices]
    depth_data_sorted = [depth_data[i] for i in sorted_indices]
    
    # Create depth coordinate
    depth_coord = xr.DataArray(
        depths_sorted,
        dims=['depth'],
        name='depth',
        attrs={'long_name': 'Depth', 'units': 'm'}
    )
    
    combined_data = xr.concat(depth_data_sorted, dim=depth_coord)
    
    # Final dataset
    ds_combined = xr.Dataset({
        'days_yearly_mean': combined_data
    })
    
    ds_combined.attrs['variable'] = var_name
    ds_combined.attrs['event_type'] = config['event_type']
    ds_combined.attrs['event_name'] = config['event_name']
    ds_combined.attrs['depths_included'] = str(depths_sorted)
    ds_combined.attrs['land_mask_applied'] = 1 if land_mask_3d is not None else 0
    
    if verbose:
        print("-"*40)
        print(f"Processed {len(depth_data)} depths")
        print(f"  Depths: {depths_sorted}")
        print(f"  Value range: {float(ds_combined['days_yearly_mean'].min().values):.2f} to "
              f"{float(ds_combined['days_yearly_mean'].max().values):.2f} days")
        if land_mask_3d is not None:
            print(f" Land mask applied")
        else:
            print(f" Land mask NOT applied")
    
    return ds_combined

# -------------------------------
# MERGE ALL INTENSITIES
# -------------------------------

def combine_intensity_data(var_name, base_path, land_mask_path=None, verbose=True):

    
    if var_name not in VARIABLES_CONFIG:
        raise ValueError(f"Variable '{var_name}' not valid. Use: {list(VARIABLES_CONFIG.keys())}")
    
    config = VARIABLES_CONFIG[var_name]
    
    if verbose:
        print(f"="*60)
        print(f"Processing intensities: {config['event_name']} ({var_name})")
        print(f"="*60)
    
    land_mask_3d = None
    if land_mask_path:
        land_mask_3d = load_global_land_mask(land_mask_path)
    
    pattern = f"{config['event_type']}_{config['name']}/depth*/tmp/intensity_max_yearly.nc"
    search_path = os.path.join(base_path, pattern)
    
    if verbose:
        print(f"Looking for files in: {search_path}")
    
    all_files = sorted(glob.glob(search_path))
    
    if not all_files:
        print(f" Files not found for intensity_max_yearly.nc")
        pattern = f"{config['event_type']}_{config['name']}/depth*/tmp/*intensity*.nc"
        search_path = os.path.join(base_path, pattern)
        all_files = sorted(glob.glob(search_path))
        
    if not all_files:
        raise FileNotFoundError(f"Files not found for {var_name} in {base_path}")
    
    if verbose:
        print(f"Found {len(all_files)} files")
        print("-"*40)
    
    depth_data = []
    depths = []
    
    for file_path in all_files:
        for part in Path(file_path).parts:
            if part.startswith('depth'):
                depth_str = part.replace('depth', '')
                break
        
        try:
            depth_val = float(depth_str)
        except ValueError:
            depth_val = depth_str
        
        if verbose:
            print(f" Depth {depth_val}m: {file_path}")
        
        try:
            ds = xr.open_dataset(file_path)
            
            intensity_var = None
            for var in ds.data_vars:
                if 'intensity' in var.lower() or var == config['name']:
                    intensity_var = var
                    break
            
            if intensity_var is None:
                intensity_var = list(ds.data_vars)[0]
            
            if 'lat' in ds.dims and 'lon' in ds.dims:
                target_lat = ds.lat.values
                target_lon = ds.lon.values
            else:
                target_lat = target_lon = None
            
            intensity_yearly = ds[intensity_var]
            intensity_yearly_mean = intensity_yearly.mean(dim="time")
            
            if land_mask_3d is not None:
                if verbose:
                    print(f"    Applying land mask...")
                
                land_mask_2d = get_land_mask_for_depth(
                    land_mask_3d, depth_val, target_lat, target_lon
                )
                
                if land_mask_2d is not None:
                    intensity_yearly_mean_masked = apply_land_mask(intensity_yearly_mean, land_mask_2d)
                    
                    intensity_yearly_mean = intensity_yearly_mean_masked
            
            depth_data.append(intensity_yearly_mean)
            depths.append(depth_val)
            
            ds.close()
            
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
    
    if not depth_data:
        raise ValueError(f"No intensity data could be loaded for {var_name}")

    sorted_indices = np.argsort(depths)
    depths_sorted = [depths[i] for i in sorted_indices]
    depth_data_sorted = [depth_data[i] for i in sorted_indices]
    
    depth_coord = xr.DataArray(
        depths_sorted,
        dims=['depth'],
        name='depth',
        attrs={'long_name': 'Depth', 'units': 'm'}
    )
    
    combined_data = xr.concat(depth_data_sorted, dim=depth_coord)
    
    ds_combined = xr.Dataset({
        'intensity_yearly_mean': combined_data
    })
    
    ds_combined.attrs['variable'] = var_name
    ds_combined.attrs['event_type'] = config['event_type']
    ds_combined.attrs['event_name'] = config['event_name']
    ds_combined.attrs['depths_included'] = str(depths_sorted)
    ds_combined.attrs['land_mask_applied'] = 1 if land_mask_3d is not None else 0
    ds_combined.attrs['data_type'] = 'intensity'
    
    if verbose:
        print("-"*40)
        print(f"  Processed {len(depth_data)} depths (intensities)")
        print(f"  Depths: {depths_sorted}")
        print(f"  Value range: {float(ds_combined['intensity_yearly_mean'].min().values):.3f} to "
              f"{float(ds_combined['intensity_yearly_mean'].max().values):.3f} {config['intensity_units']}")
        if land_mask_3d is not None:
            print(f" Land mask applied")
        else:
            print(f"  Land mask NOT applied")
    
    return ds_combined

# -------------------------------
#  SAVING COMBINED DATA
# -------------------------------

def save_combined_data(ds_combined, output_dir='./', data_type='days', verbose=True):

    var_name = ds_combined.attrs['variable']
    event_type = ds_combined.attrs['event_type']
    
    if data_type == 'intensity':
        output_filename = f"{event_type}_{var_name}_intensity_yearly_mean_all_depths.nc"
    else:
        output_filename = f"{event_type}_{var_name}_days_yearly_mean_all_depths.nc"
    
    output_path = os.path.join(output_dir, output_filename)
    
    os.makedirs(output_dir, exist_ok=True)
    
    ds_to_save = ds_combined.copy()
    
    for key, value in list(ds_to_save.attrs.items()):
        if isinstance(value, bool):
            ds_to_save.attrs[key] = int(value)
        elif isinstance(value, np.bool_):
            ds_to_save.attrs[key] = int(value)
    
    for var in ds_to_save.data_vars:
        for key, value in list(ds_to_save[var].attrs.items()):
            if isinstance(value, bool) or isinstance(value, np.bool_):
                ds_to_save[var].attrs[key] = int(value)
    
    ds_to_save.to_netcdf(output_path)
    
    if verbose:
        print(f"Data saved in: {output_path}")
    
    return output_path

# -------------------------------
# VISUALIZATION
# -------------------------------

def create_summary_plot(ds_combined, output_dir='./', data_type='days', verbose=True):
    
    var_name = ds_combined.attrs['variable']
    event_type = ds_combined.attrs['event_type']
    event_name = ds_combined.attrs['event_name']
    config = VARIABLES_CONFIG[var_name]
    
    if data_type == 'intensity':
        data_var = 'intensity_yearly_mean'
        title_suffix = 'Intensity'
        cbar_label = config['intensity_units']
    else:
        data_var = 'days_yearly_mean'
        title_suffix = 'Mean Yearly Days'
        cbar_label = 'Days'
    
    data = ds_combined[data_var]
    
    if verbose:
        print(f"Creating summary plot for {event_name} ({title_suffix})...")
    
    # Subplots
    n_depths = len(data.depth)
    n_cols = min(3, n_depths)
    n_rows = int(np.ceil(n_depths / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    fig.suptitle(f'{event_name} - {title_suffix}', fontsize=16, fontweight='bold')
    
    mask_applied = ds_combined.attrs.get('land_mask_applied', 0)
  
    if n_depths == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    vmin = float(data.quantile(0.05).values)
    vmax = float(data.quantile(0.95).values)

    for idx, depth in enumerate(data.depth.values):
        ax = axes[idx]
        
        data_slice = data.sel(depth=depth)
        
        im = data_slice.plot(
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cmap=config['cmap'],
            add_colorbar=False,
            robust=True
        )
        
        if mask_applied == 1 or mask_applied is True:
            im.cmap.set_bad('gray', alpha=0.7)
        
        ax.set_title(f'Depth: {depth}m', fontsize=10)
    
    for idx in range(n_depths, len(axes)):
        axes[idx].axis('off')
    
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label=cbar_label)
    
    bottom_margin = 0.02 if (mask_applied == 1 or mask_applied is True) else 0
    plt.tight_layout(rect=[0, bottom_margin, 0.9, 0.95])
    
    if data_type == 'intensity':
        output_png = os.path.join(output_dir, f'{event_type}_{var_name}_intensity_summary.png')
    else:
        output_png = os.path.join(output_dir, f'{event_type}_{var_name}_summary.png')
    
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"Summary plot saved in: {output_png}")
    
    return output_png



def create_depth_animation_simple(ds_combined, output_dir='./', data_type='days', fps=1, dpi=150, verbose=True):

    
    var_name = ds_combined.attrs['variable']
    event_type = ds_combined.attrs['event_type']
    event_name = ds_combined.attrs['event_name']
    config = VARIABLES_CONFIG[var_name]
    
    if data_type == 'intensity':
        data_var = 'intensity_yearly_mean'
        title_suffix = 'Intensity'
        cbar_label = config['intensity_units']
    else:
        data_var = 'days_yearly_mean'
        title_suffix = 'Yearly Days'
        cbar_label = 'Days'

    data = ds_combined[data_var]
    n_frames = len(data.depth)

    if verbose:
        print(f"Creating simple animation for {event_name} ({title_suffix})...")
        print(f"  NNumber of frames: {n_frames}")

    if n_frames == 0:
        print("No depth data available.")
        return None

    vmin = float(data.quantile(0.05).values)
    vmax = float(data.quantile(0.95).values)

    fig, ax = plt.subplots(figsize=(12, 8))

    first_slice = data.isel(depth=0)
    quadmesh = first_slice.plot(
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cmap=config['cmap'],
        add_colorbar=False,
        robust=True
    )

    mask_applied = ds_combined.attrs.get('land_mask_applied', 0)
    if mask_applied == 1 or mask_applied is True:
        quadmesh.cmap.set_bad('gray', alpha=0.7)

    cbar = fig.colorbar(quadmesh, ax=ax)
    cbar.set_label(cbar_label, rotation=90, labelpad=20)

    depth0 = data.depth.values[0]
    title = f'{event_name} - Depth: {depth0}m - {title_suffix}'
    ax.set_title(title, fontsize=14, fontweight='bold')

    def update(frame):
        depth = data.depth.values[frame]
        data_slice = data.isel(depth=frame)

        quadmesh.set_array(data_slice.values.ravel())

        title = f'{event_name} - Depth: {depth}m - {title_suffix}'
        ax.set_title(title, fontsize=14, fontweight='bold')

        return quadmesh,

    ani = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=n_frames,
        interval=1000//fps,
        blit=False,
        repeat=True
    )

    if data_type == 'intensity':
        output_gif = os.path.join(output_dir, f'{event_type}_{var_name}_intensity_animation.gif')
    else:
        output_gif = os.path.join(output_dir, f'{event_type}_{var_name}_animation.gif')

    try:
        ani.save(output_gif, writer='pillow', fps=fps, dpi=dpi)
        plt.close(fig)

        if verbose:
            print(f"Animation saved: {output_gif}")

        return output_gif

    except Exception as e:
        print(f"Error saving GIF: {e}")
        plt.close(fig)
        return None

def create_frequency_plot_simple(ds_combined, land_mask_path, output_dir='./', data_type='days', verbose=True):

    
    var_name = ds_combined.attrs['variable']
    event_type = ds_combined.attrs['event_type']
    event_name = ds_combined.attrs['event_name']
    config = VARIABLES_CONFIG[var_name]
    
    if verbose:
        print(f"\nCreating frequency plot {'(INTENSITIES)' if data_type == 'intensity' else '(DAYS)'}...")
        print(f"  Variable: {event_name}")
    
    base_path = '/work/bg1446/u241379/extremes_out_final/'
    var_path = config['name']
    
    land_mask_3d = None
    if land_mask_path:
        land_mask_3d = load_global_land_mask(land_mask_path)
    
    if data_type == 'intensity':
        pattern1 = os.path.join(base_path, f"{event_type}_{var_path}", "depth*", "tmp", "intensity_max_monthly.nc")
        pattern2 = os.path.join(base_path, f"{event_type}_{var_path}", "depth*", "tmp", "*intensity*monthly*.nc")
        
        monthly_files = sorted(glob.glob(pattern1))
        if not monthly_files:
            monthly_files = sorted(glob.glob(pattern2))
    else:
        pattern = os.path.join(base_path, f"{event_type}_{var_path}", "depth*", "tmp", "days_monthly.nc")
        monthly_files = sorted(glob.glob(pattern))
    
    if not monthly_files:
        print(f" Monthly files not found for {data_type}")
        if data_type == 'intensity':
            pattern = os.path.join(base_path, f"{event_type}_{var_path}", "depth*_*", "tmp", "intensity_max_monthly.nc")
        else:
            pattern = os.path.join(base_path, f"{event_type}_{var_path}", "depth*_*", "tmp", "days_monthly.nc")
        
        monthly_files = sorted(glob.glob(pattern))
        if not monthly_files:
            print(f" Monthly files not found for {data_type}")
            return None
    
    if verbose:
        print(f"  Found {len(monthly_files)} files")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    depth_files_dict = {}
    
    for file_path in monthly_files:
        try:
            parts = Path(file_path).parts
            depth_value = None
            for part in parts:
                if part.startswith('depth'):
                    depth_str = part.replace('depth', '')
                    if '_' in depth_str:
                        depth_str = depth_str.replace('_', '.')
                    try:
                        depth_value = float(depth_str)
                    except:
                        try:
                            depth_value = int(depth_str)
                        except:
                            depth_value = depth_str
                    break
            
            if depth_value is None:
                continue
            
            if depth_value not in depth_files_dict:
                depth_files_dict[depth_value] = file_path
            else:
                if verbose:
                    print(f"  Depth {depth_value}m found twice, using: {depth_files_dict[depth_value]}")
            
        except Exception as e:
            if verbose:
                print(f" Error extracting depth from {file_path}: {str(e)[:80]}")
    
    plot_data = []
    
    for depth_value, file_path in sorted(depth_files_dict.items()):
        try:
            if verbose:
                print(f"  Processing depth {depth_value}m...")
            
            ds = xr.open_dataset(file_path)
            
            data_var = None
            if data_type == 'intensity':
                for var in ds.data_vars:
                    if 'intensity' in var.lower() or var == config['name']:
                        data_var = var
                        break
                if data_var is None:
                    data_var = list(ds.data_vars)[0]
            else:
                data_var = 'days_total'
            
            if data_var not in ds:
                print(f"  '{data_var}' not found in {file_path}")
                ds.close()
                continue
                
            monthly_data = ds[data_var]
            
            if 'lat' in ds.dims and 'lon' in ds.dims:
                target_lat = ds.lat.values
                target_lon = ds.lon.values
            else:
                target_lat = target_lon = None
            
            if land_mask_3d is not None:
                land_mask_2d = get_land_mask_for_depth(
                    land_mask_3d, depth_value, target_lat, target_lon
                )
                
                if land_mask_2d is not None:
                    masked_data = []
                    for t in range(len(monthly_data.time)):
                        time_slice = monthly_data.isel(time=t)
                        masked_slice = apply_land_mask(time_slice, land_mask_2d)
                        masked_data.append(masked_slice.values)
                    
                    monthly_data.values = np.array(masked_data)
            
            try:
                spatial_mean = monthly_data.mean(dim=['lat', 'lon'], skipna=True)
            except:
                try:
                    spatial_mean = monthly_data.mean(dim=['latitude', 'longitude'], skipna=True)
                except:
                    spatial_mean = monthly_data.mean(skipna=True)
            
            times = spatial_mean['time'].values
            values = spatial_mean.values
            
            dates = []
            for t in times:
                try:
                    if isinstance(t, np.datetime64):
                        dates.append(pd.Timestamp(t).to_pydatetime())
                    elif hasattr(t, 'year'):
                        dates.append(t)
                    else:
                        dates.append(t)
                except:
                    dates.append(t)
            
            plot_data.append({
                'depth': depth_value,
                'dates': dates,
                'values': values,
                'masked': land_mask_3d is not None
            })
            
            if verbose:
                mask_status = " (masked)" if land_mask_3d is not None else ""
                print(f" Processed: {len(values)} points{mask_status}")
            
            ds.close()
            
        except Exception as e:
            if verbose:
                print(f"   Error in {file_path}: {str(e)[:80]}")
    
    if not plot_data:
        print("No data could be loaded from any file")
        return None
    
    plot_data_sorted = sorted(plot_data, key=lambda x: x['depth'])
    
    if verbose:
        print(f" Processed {len(plot_data_sorted)} unique depths")
    
    n_depths = len(plot_data_sorted)
    
    if n_depths <= 10:
        custom_colors = [
            '#1f77b4',  # Blue
            '#ff7f0e',  # Orange
            '#2ca02c',  # Green
            '#d62728',  # Red
            '#9467bd',  # Purple
            '#8c564b',  # Brown
            '#e377c2',  # Pink
            '#7f7f7f',  # Dark gray
            '#bcbd22',  # Lime
            '#17becf',  # Cyan
        ]
        colors = custom_colors[:n_depths]
        
    else:
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_depths))
    
    for i, data in enumerate(plot_data_sorted):
        ax.plot(data['dates'], data['values'],
               label=f"{data['depth']}m",
               color=colors[i],
               linewidth=2.0,
               alpha=0.85)
    
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    
    if data_type == 'intensity':
        ax.set_ylabel(f'Intensity ({config["intensity_units"]})', fontsize=12, fontweight='bold')
        title = f'{event_name} - Monthly Intensity by Depth'
    else:
        ax.set_ylabel('Days per Month', fontsize=12, fontweight='bold')
        title = f'{event_name} - Monthly Frequency by Depth'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    if n_depths <= 10:
        ncol = 2
        loc = 'upper left'
        legend_fontsize = 10
    elif n_depths <= 20:
        ncol = 3
        loc = 'upper left'
        legend_fontsize = 9
    else:
        ncol = 4
        loc = 'upper center'
        legend_fontsize = 8
    
    legend = ax.legend(title='Depth (m)', fontsize=legend_fontsize, 
                       title_fontsize=legend_fontsize+1,
                       loc=loc, ncol=ncol, 
                       framealpha=1.0,
                       frameon=True,
                       edgecolor='black',
                       facecolor='white')
    
    for line in legend.get_lines():
        line.set_linewidth(3.0)
    
    plt.tight_layout()
    
    if data_type == 'intensity':
        output_png = os.path.join(output_dir, f'{event_type}_{var_name}_intensity_frequency.png')
    else:
        output_png = os.path.join(output_dir, f'{event_type}_{var_name}_frequency.png')
    
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f" Saved plot: {output_png}")
    
    return output_png

# -------------------------------
# MAIN FUNCTION WITH EVENT ORGANIZATION
# -------------------------------

def main():
    
    parser = argparse.ArgumentParser(
        description='Combine data by depth with land mask and intensities, organized by events.'
    )
    
    parser.add_argument('variable', 
                       choices=['to', 'o2', 'hi', 'all'],
                       help='Variable to process: to (Temperature), o2 (Oxygen), hi (Heat Index), or all')
    
    parser.add_argument('--base-path', 
                       default='___________/extremes_out_final/',
                       help='Base path for the data')
    
    parser.add_argument('--output-dir', 
                       default='./summary_results',
                       help='Main output directory')
    
    parser.add_argument('--land-mask-path',
                       default=None,
                       help=f'Path to the land mask (default: {DEFAULT_LAND_MASK_PATH})')
    
    parser.add_argument('--no-gif', 
                       action='store_true',
                       help='Do not create GIF animation')
    
    parser.add_argument('--no-plot', 
                       action='store_true',
                       help='Do not create summary plot')
    
    parser.add_argument('--no-freq', 
                       action='store_true',
                       help='Do not create frequency plot')
    
    parser.add_argument('--no-intensity',
                       action='store_true',
                       help='Do not process intensity data')
    
    parser.add_argument('--no-mask',
                       action='store_true',
                       help='Do not apply land mask')
    
    parser.add_argument('--quiet', 
                       action='store_true',
                       help='Silent mode')
    
    args = parser.parse_args()
    
    if args.no_mask:
        land_mask_path_to_use = None
    else:
        land_mask_path_to_use = args.land_mask_path if args.land_mask_path else DEFAULT_LAND_MASK_PATH
    
    verbose = not args.quiet
    
    if verbose:
        print("="*60)
        print("Combination of Depth Data with Land Mask and Intensities")
        print("Organization by Events")
        print("="*60)
        if land_mask_path_to_use:
            print(f"Land mask: {land_mask_path_to_use}")
        else:
            print("Land mask: DISABLED")
        if args.no_intensity:
            print("Intensities: DISABLED")
    
    if args.variable == 'all':
        variables_to_process = ['to', 'o2', 'hi']
    else:
        variables_to_process = [args.variable]
    
    all_results = []
    
    for var in variables_to_process:
        if verbose:
            print(f"\n{'='*60}")
            print(f"PROCESSING VARIABLE: {var}")
            print('='*60)
        
        config = VARIABLES_CONFIG[var]
        event_name = config['event_name']
        event_type = config['event_type']
        
        # -------------------------------
        # CREATE FOLDER STRUCTURE FOR THIS EVENT
        # -------------------------------
        # Main event directory
        event_dir = os.path.join(args.output_dir, event_name)
        
        # Subdirectories within the event
        event_data_dir = os.path.join(event_dir, 'data')
        event_plots_dir = os.path.join(event_dir, 'plots')
        
        # Create directories
        dirs_to_create = [event_dir, event_data_dir, event_plots_dir]
        
        for directory in dirs_to_create:
            os.makedirs(directory, exist_ok=True)
            if verbose:
                print(f" Directory created: {directory}")
        
        # -------------------------------
        # 1. DAILY DATA PROCESSING
        # -------------------------------
        days_results = {'variable': var, 'type': 'days', 'success': False}
        
        try:
            # 1. Saving daily data (with/sin mask)
            ds_days = combine_depth_data(var, args.base_path, land_mask_path_to_use, verbose)
            
            # 2. Saving NetCDF of daily data in event's data/ folder
            nc_file_days = save_combined_data(ds_days, event_data_dir, 'days', verbose)
            
            # 3. Creating GIF of daily data (if requested) in event's plots/ folder
            gif_file_days = None
            if not args.no_gif:
                try:
                    gif_file_days = create_depth_animation_simple(ds_days, event_plots_dir, 'days', verbose=verbose)
                except Exception as e:
                    print(f" Error creating GIF : {e}")
            
            # 4. Creating spatial plot of days (if requested) in event's plots/ folder
            plot_file_days = None
            if not args.no_plot:
                try:
                    plot_file_days = create_summary_plot(ds_days, event_plots_dir, 'days', verbose=verbose)
                except Exception as e:
                    print(f" Error creating days plot: {e}")

            # 5. Creating frequency plot of days (if requested) in event's plots/ folder
            freq_file_days = None
            if not args.no_freq:
                try:
                    freq_file_days = create_frequency_plot_simple(ds_days, land_mask_path_to_use, event_plots_dir, 'days', verbose=verbose)
                except Exception as e:
                    print(f" Error creating frequency plot of days: {e}")

            days_results.update({
                'success': True,
                'event_dir': event_dir,
                'nc_file': nc_file_days,
                'gif_file': gif_file_days,
                'plot_file': plot_file_days,
                'freq_file': freq_file_days,
                'mask_applied': land_mask_path_to_use is not None and ds_days.attrs.get('land_mask_applied', 0) == 1
            })
            
        except Exception as e:
            print(f" Error processing days for {var}: {e}")
            days_results['error'] = str(e)
        
        all_results.append(days_results)
        
        # -------------------------------
        # 2. PROCESSING INTENSITY DATA (if requested)
        # -------------------------------
        if not args.no_intensity:
            intensity_results = {'variable': var, 'type': 'intensity', 'success': False}
            
            try:
                # 1. Combination of intensity data (with/sin mask)
                ds_intensity = combine_intensity_data(var, args.base_path, land_mask_path_to_use, verbose)
                
                # 2. Saving NetCDF of intensity data in event's data/ folder
                nc_file_intensity = save_combined_data(ds_intensity, event_data_dir, 'intensity', verbose)
                
                # 3. Creating GIF of intensity data (if requested) in event's plots/ folder
                gif_file_intensity = None
                if not args.no_gif:
                    try:
                        gif_file_intensity = create_depth_animation_simple(ds_intensity, event_plots_dir, 'intensity', verbose=verbose)
                    except Exception as e:
                        print(f"  Error creating GIF of intensity: {e}")

                # 4. Creating spatial plot of intensity (if requested) in event's plots/ folder
                plot_file_intensity = None
                if not args.no_plot:
                    try:
                        plot_file_intensity = create_summary_plot(ds_intensity, event_plots_dir, 'intensity', verbose=verbose)
                    except Exception as e:
                        print(f"  Error creating intensity plot: {e}")

                # 5. Creating frequency plot of intensity (if requested) in event's plots/ folder
                freq_file_intensity = None
                if not args.no_freq:
                    try:
                        freq_file_intensity = create_frequency_plot_simple(ds_intensity, land_mask_path_to_use, event_plots_dir, 'intensity', verbose=verbose)
                    except Exception as e:
                        print(f"  Error creating frequency plot of intensity: {e}")

                intensity_results.update({
                    'success': True,
                    'event_dir': event_dir,
                    'nc_file': nc_file_intensity,
                    'gif_file': gif_file_intensity,
                    'plot_file': plot_file_intensity,
                    'freq_file': freq_file_intensity,
                    'mask_applied': land_mask_path_to_use is not None and ds_intensity.attrs.get('land_mask_applied', 0) == 1
                })
                
            except Exception as e:
                print(f"Error processing intensities for {var}: {e}")
                intensity_results['error'] = str(e)
            
            all_results.append(intensity_results)
    
    # Final summary
    if verbose:
        print("\n" + "="*60)
        print(" COMPLETE SUMMARY OF THE EVENT ORGANIZATION")
        print("="*60)
        
        for res in all_results:
            var_type = "INTENSITY" if res['type'] == 'intensity' else "DAYS"
            if res['success']:
                mask_status = " ( masked)" if res.get('mask_applied', False) else " (no mask)"
                print(f"\n{res['variable'].upper()} - {var_type}: ✓ COMPLETED{mask_status}")
                print(f"  Event directory: {res['event_dir']}")
                if res.get('nc_file'):
                    print(f"    • NetCDF (data/): {os.path.basename(res['nc_file'])}")
                if res.get('gif_file'):
                    print(f"    • GIF (plots/): {os.path.basename(res['gif_file'])}")
                if res.get('plot_file'):
                    print(f"    • Plot (plots/): {os.path.basename(res['plot_file'])}")
                if res.get('freq_file'):
                    print(f"    • Frequency (plots/): {os.path.basename(res['freq_file'])}")
            else:
                print(f"\n{res['variable'].upper()} - {var_type}: ✗ FAILED")
                print(f"  Error: {res.get('error', 'Unknown')}")
        
        print(f"\nFOLDER STRUCTURE CREATED:")
        print(f"  Main directory: {os.path.abspath(args.output_dir)}")
        
        # Show folder structure
        if os.path.exists(args.output_dir):
            for event in os.listdir(args.output_dir):
                event_path = os.path.join(args.output_dir, event)
                if os.path.isdir(event_path):
                    print(f"\n  {event}/")
                    # Subdirectories
                    for subdir in ['data', 'plots']:
                        subdir_path = os.path.join(event_path, subdir)
                        if os.path.exists(subdir_path):
                            files = os.listdir(subdir_path)
                            if files:
                                print(f"    ├── {subdir}/ ({len(files)} files)")
                                file_types = {}
                                for f in files:
                                    ext = os.path.splitext(f)[1].lower()
                                    if ext not in file_types:
                                        file_types[ext] = []
                                    file_types[ext].append(f)
                                
                                for ext in sorted(file_types.keys()):
                                    file_list = file_types[ext]
                                    if len(file_list) <= 3:
                                        for f in sorted(file_list):
                                            print(f"    │   • {f}")
                                    else:
                                        for f in sorted(file_list)[:2]:
                                            print(f"    │   • {f}")
                                        print(f"    │   • ... and {len(file_list)-2} more")
                            else:
                                print(f"    ├── {subdir}/ (empty)")
        
        print("="*60)

if __name__ == "__main__":
    main()

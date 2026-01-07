"""
Land Mask Creation from Temperature Data
========================================

Creates a 3D land-sea mask using temperature data from multiple depths.
The mask identifies ocean pixels (value=1) and land pixels (value=NaN)
based on temperature data availability.

Author:
    Cristina Radin (2025)
    cristina.radin@uni-hamburg.de
    AI4PEX (https://ai4pex.org/)
    University of Hamburg
"""


import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt



# Configuration

DEPTHS = [8, 51, 104, 186.5, 489]
BASE_PATH = '___________/extremes_out_final'  
OUTPUT_FILE = '___________/land_mask_3d.nc'

# -------------------------------
# CREAR MÁSCARA 3D
# -------------------------------

print("="*70)
print("CREATING LAND MASK (using temperature 'to' data)")
print("="*70)

masks = []
valid_depths = []

for depth in DEPTHS:
    depth_str = str(depth)
    if depth_str.endswith('.0'):
        depth_str = depth_str[:-2]
    
    data_file = f'{BASE_PATH}/mhw_to/depth{depth_str}/tmp/all_data_monmean.nc'
    
    if os.path.exists(data_file):
        try:
            ds = xr.open_dataset(data_file)
            temperature_data = ds['to']
            
            data_sum = abs(temperature_data).sum(dim='time')
            mask_2d = xr.where(data_sum == 0, np.nan, 1)
            mask_2d = mask_2d.squeeze()
            
            mask_2d = mask_2d.expand_dims(depth=[depth])
            
            masks.append(mask_2d)
            valid_depths.append(depth)
            
            ocean_pixels = np.sum(~np.isnan(mask_2d.values))
            total_pixels = mask_2d.size
            ocean_percent = (ocean_pixels / total_pixels) * 100
            
            if isinstance(depth, float) and not depth.is_integer():
                depth_display = f"{depth:5.1f}"
            else:
                depth_display = f"{int(depth):5d}"
            
            print(f"Depth {depth_display}m: {ocean_pixels:7,} ocean pixels ({ocean_percent:5.1f}%)")
            
        except Exception as e:
            print(f"Depth {depth:5.1f}m: Error - {str(e)[:80]}")
    else:
        print(f"File not found: {data_file}")  

# -------------------------------
# COMBINE AND SAVE MASK
# -------------------------------

if masks:
    mask_3d = xr.concat(masks, dim='depth')
    mask_3d.name = 'land_mask'
    
    mask_3d.attrs.update({
        'long_name': 'Land-sea mask',
        'units': '1',
        'description': '1 = sea, NaN = land',
        'source_variable': 'to (temperature)',
        'depths': str(valid_depths)
    })
    
    mask_3d.to_netcdf(OUTPUT_FILE)
    
    print("\n" + "="*70)
    print(f"MASK SAVED: {OUTPUT_FILE}")
    print(f"   Depths: {valid_depths}")
    print(f"   Dimensions: {mask_3d.shape}")
    print("="*70)


    # -------------------------------
    # VISUALIZATION
    # -------------------------------
    try:
        loaded_mask = xr.open_dataarray(OUTPUT_FILE)
        
        n_depths = len(valid_depths)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (depth, ax) in enumerate(zip(valid_depths, axes)):
            if i < len(axes):
                mask_slice = loaded_mask.sel(depth=depth)
                mask_2d = mask_slice.values if hasattr(mask_slice, 'values') else mask_slice
                
                cmap = plt.cm.Blues
                cmap.set_bad('lightgray', 1.0)
                
                im = ax.imshow(mask_2d, cmap=cmap, vmin=0.9, vmax=1.1, 
                              origin='lower', aspect='auto')
                
                ocean_pixels = np.sum(~np.isnan(mask_2d))
                total_pixels = mask_2d.size
                ocean_percent = (ocean_pixels / total_pixels) * 100
                
                if isinstance(depth, float) and not depth.is_integer():
                    title = f"Depth {depth:.1f}m\n{ocean_percent:.1f}% ocean"
                else:
                    title = f"Depth {int(depth)}m\n{ocean_percent:.1f}% ocean"
                
                ax.set_title(title, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
        
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        
        plt.suptitle(f'Land Mask - {n_depths} depths', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_file = OUTPUT_FILE.replace('.nc', '.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"\n Plot saved: {plot_file}")
        plt.close()
        
    except Exception as e:
        print(f"\n Could not create plot: {e}")
        print("   Mask was saved successfully.")
    
else:
    print("\n ERROR: Could not create any masks")
    print(f"   Verify files exist in:")
    print(f"   {BASE_PATH}/mhw_to/depth[XX]/tmp/all_data_monmean.nc")

    print("\nDirectories existing in mhw_to:")
    mhw_path = f'{BASE_PATH}/mhw_to'
    if os.path.exists(mhw_path):
        for item in os.listdir(mhw_path):
            if item.startswith('depth'):
                print(f"  - {item}")




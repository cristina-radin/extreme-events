#!/usr/bin/env bash

#SBATCH -J thresholds
#SBATCH --mail-user=___________
#SBATCH -p shared
#SBATCH --output ___________/extremes_out_final/job%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -A bg1446
#SBATCH --time=12:00:00
#SBATCH --mem=128G

####################################################################################################
# Project: AI4PEX (https://ai4pex.org/)
# Script: extreme_events_computation.sh
# Author: Cristina Radin (University of Hamburg)
# Contact: cristina.radin@uni-hamburg.de
# Based on script from Danai Filippou
# Date: 12.2025
#
# Description:
# This script computes extreme events (MHW, DEO, OAX) from ICON-COAST model output.
# It performs a full CDO-based preprocessing and extreme-event detection pipeline:
#
#   1. Preprocessing & Grid Remapping
#      - Extracts region, depth, and variable-of-interest (to, o2, hi).
#      - Converts units (o2, hi to μmol m-3)
#      - Shifts timestamps, remaps to target grid.
#      - Processes files in blocks for efficiency.
#
#   2. Merging
#      - Merges processed blocks into a continuous time series.
#
#   3. Reference Period
#      - Selects reference climatology period (1985-2014).
#      - Converts to non-leap calendar (deletes Feb 29).
#
#   4. Threshold Calculation
#      - Produces climatological thresholds (percentile-based).
#      - Computes monthly means for the threshold, reference and entire period.
#
#   5. Extreme Event Detection
#      - Computes daily anomalies relative to threshold.
#      - Generates binary extreme mask (≥ or ≤ threshold depending on variable).
#      - Computes consecutive-day duration, filters events by minimum duration 
#        (5 days, following Hobday et al., 2016) and stores duration.
#      - Computes intensity of extreme events.
#      - Mean and maximum intensities (total, yearly, monthly).
#
#   6. Post-processing & Visualization
#      - Computes a filtered mask with extreme events (1=day in an Extreme Event with >5 days).
#      - Total, yearly, and monthly number of extreme days with cdo. 
#      - Generates spatial and temporal plots.
#
# Output Structure:
#   ./extremes_out_final/<extreme>_<var>/depth<depth>/
#   ├── tmp/          # Temporary processing files
#   ├── *.nc          # Final output NetCDFs
#   └── *.png         # Diagnostic plots
#
# Notes:
#   - Adjust configuration parameters (variable, depth, paths) in CONFIG section.
#   - Requires CDO 2.5.0+ and access to target grid file.
#   - Uses Python environment with xarray, numpy, matplotlib, numba.
#   - All temporary files are stored in tmp/ directory and removed progressively.
#   - Memory intensive: uses 128GB RAM and 12h walltime.
#
#
# References:
#   Hobday et al. (2016) - A hierarchical approach to defining marine heatwaves
#   ICON-COAST model documentation
#
####################################################################################################



module load cdo/2.5.0-gcc-11.2.0
module load nco 2>/dev/null || echo "Note: nco module not available"


#######################################
# CONFIG
#######################################

base_path="___________"
target_grid="___________"


var="hi" #to, hi, o2 
depth=489 #8, 51, 104, 186.5, 489 m

if [ "$var" = "to" ]; then
    percentile=90
    comparison="gec"
    extreme="mhw"

elif [ "$var" = "o2" ]; then
    percentile=10
    comparison="lec"
    extreme="deo"

elif [ "$var" = "hi" ]; then
    percentile=90
    comparison="gec"
    extreme="oax"
fi

echo "FINAL CONFIGURATION: "
echo "var: $var"
echo "depth: $depth"
echo "percentile: $percentile"
echo "comparison: $comparison"

out_dir="./extremes_out_final/${extreme}_${var}/depth${depth}"
tmp_dir="$out_dir/tmp"
mkdir -p "$out_dir" "$tmp_dir"

ref_start=1985
ref_end=2014

block_size=30
CDO_CORES=1  

# REGION
lon_min=-80
lon_max=20
lat_min=0
lat_max=70



#######################################
# CDO FUNCTION
#######################################

run_cdo() {
    echo "[CDO] $*"
    "$@"
}
start=$(date +%s)



#######################################
# 1. PREPROCESSING & GRID REMAPPING
#######################################

merged_blocks=()

for chunk in "$base_path"/chunk*; do
    [ ! -d "$chunk" ] && continue

    files=($(ls "$chunk"___________daily_[0-9][0-9]*T*.nc 2>/dev/null | sort))
    [ ${#files[@]} -eq 0 ] && continue

    tmp_list=()
    count=0

    for f in "${files[@]}"; do
        tmp_out=$(mktemp --tmpdir="$tmp_dir" --suffix=".nc")

            run_cdo cdo \
                -sellonlatbox,$lon_min,$lon_max,$lat_min,$lat_max \
                -sellevel,$depth \
                -selvar,$var \
                -shifttime,-1day \
                -remapnn,"$target_grid" \
                "$f" "$tmp_out"

            if [ "$var" = "o2" ]; then
                tmp_out_units=$(mktemp --tmpdir="$tmp_dir" --suffix=".nc")  

                run_cdo cdo -expr,"o2=o2*1e9" "$tmp_out" "$tmp_out_units"

                if command -v ncatted &>/dev/null; then
                    ncatted -a units,o2,o,c,"micromol m-3" \
                            -a long_name,o2,o,c,"oxygen concentration" \
                            "$tmp_out_units" 2>/dev/null || true
                    if [ $count -eq 1 ]; then  
                        echo "Metadatos actualizados (units: micromol O2 m-3)"
                    fi
                fi

                mv "$tmp_out_units" "$tmp_out"
                
            elif [ "$var" = "hi" ]; then
                tmp_out_units=$(mktemp --tmpdir="$tmp_dir" --suffix=".nc")
                    
                run_cdo cdo -expr,"hi=hi*1e9" "$tmp_out" "$tmp_out_units"
                    
                if command -v ncatted &>/dev/null; then
                    ncatted -a units,hi,o,c,"micromol m-3" \
                            -a long_name,hi,o,c,"hydrogen ion concentration" \
                            "$tmp_out_units" 2>/dev/null || true
                    if [ $count -eq 1 ]; then
                            echo "HI: Convertido a  (units: micromol O2 m-3)"
                    fi
                fi
                    
                    mv "$tmp_out_units" "$tmp_out"
            fi

        tmp_list+=("$tmp_out")
        count=$((count+1))
        duration=$SECONDS
        echo "-> File processed in $duration seconds"

        if (( count % block_size == 0 )); then
            merged=$(mktemp --tmpdir="$tmp_dir" --suffix=".nc")
            run_cdo cdo mergetime "${tmp_list[@]}" "$merged"
            merged_blocks+=("$merged")
            rm -f "${tmp_list[@]}"
            tmp_list=()
            count=0
        fi
    done

    if [ ${#tmp_list[@]} -gt 0 ]; then
        merged=$(mktemp --tmpdir="$tmp_dir" --suffix=".nc")
        echo "=== Final merge of partial block -> $merged ==="
        run_cdo cdo mergetime "${tmp_list[@]}" "$merged"
        merged_blocks+=("$merged")
        rm -f "${tmp_list[@]}"
    fi
done



#######################################
# 2. MERGING
#######################################

tmp_all_data_merged="$tmp_dir/tmp_all_data_merged.nc"
run_cdo cdo mergetime "${merged_blocks[@]}" "$tmp_all_data_merged"



#######################################
# 3. REFERENCE PERIOD PROCESSING
#######################################

tmp_ref="$tmp_dir/tmp_ref.nc"
ref_nl="$tmp_dir/ref_${ref_start}_${ref_end}.nc"
all_data_nl="$tmp_dir/all_data.nc"

run_cdo cdo selyear,$ref_start/$ref_end "$tmp_all_data_merged" "$tmp_ref"

run_cdo cdo delete,month=2,day=29 "$tmp_ref" "$ref_nl"
run_cdo cdo delete,month=2,day=29 "$tmp_all_data_merged" "$all_data_nl"

rm -f "$tmp_all_data_merged"
rm -f "$tmp_ref" 
rm -f "${merged_blocks[@]}"



#######################################
# 4. THRESHOLD CALCULATION
#######################################

min="$tmp_dir/min.nc"
max="$tmp_dir/max.nc"
tmp_thresh="$tmp_dir/tmp_thresh.nc"
thresh="$tmp_dir/threshold.nc"
thresh_monmean="$tmp_dir/threshold_monmean.nc"
ref_monmean="$tmp_dir/ref_${ref_start}_${ref_end}_monmean.nc"
all_data_monmean="$tmp_dir/all_data_monmean.nc"


run_cdo cdo -P 32 -ydrunmin,5 "$ref_nl" "$min"
run_cdo cdo -P 32 -ydrunmax,5 "$ref_nl" "$max"

run_cdo cdo -P 32 -ydrunpctl,${percentile},5 "$ref_nl" "$min" "$max" "$tmp_thresh"
run_cdo cdo -settaxis,0001-01-01,00:00:00,1day "$tmp_thresh" "$thresh"


run_cdo cdo -P 32 monmean "$thresh" "$thresh_monmean"
run_cdo cdo -P 32 monmean "$ref_nl" "$ref_monmean"
run_cdo cdo -P 32 monmean "$all_data_nl" "$all_data_monmean"


rm -f "$tmp_thresh"



###################################
# 5. EXTREME EVENT DETECTION
###################################

tmp_anom="$tmp_dir/tmp_anomalies.nc" 
binary="$tmp_dir/binary.nc" 
tmp_count="$tmp_dir/tmp_count.nc" 
tmp_duration="$tmp_dir/tmp_duration.nc" 
tmp_duration2="$tmp_dir/tmp_duration2.nc" 
duration="$tmp_dir/duration.nc" 
intensity="$tmp_dir/intensity.nc" 


run_cdo cdo -P 100 sub "$all_data_nl" "$thresh" "$tmp_anom" 
run_cdo cdo -P 100 ${comparison},0 "$tmp_anom" "$binary"  
run_cdo cdo -P 100 consecsum "$binary" "$tmp_count"
run_cdo cdo -P 100 gec,5 "$tmp_count" "$tmp_duration" 
run_cdo cdo -P 100 mul "$tmp_duration" "$tmp_count" "$tmp_duration2" 
run_cdo cdo -P 100 setmissval,0  "$tmp_duration2"  "$duration" 


run_cdo cdo -P 100 mul "$binary" "$tmp_anom" "$intensity"


rm -f "$tmp_count" "$tmp_duration" "$tmp_anom" "$tmp_duration2"


###################################


intensity_mean_total="$tmp_dir/intensity_mean_total.nc"
intensity_mean_yearly="$tmp_dir/intensity_mean_yearly.nc"
intensity_mean_monthly="$tmp_dir/intensity_mean_monthly.nc"

intensity_max_total="$tmp_dir/intensity_max_total.nc"
intensity_max_yearly="$tmp_dir/intensity_max_yearly.nc"
intensity_max_monthly="$tmp_dir/intensity_max_monthly.nc"


run_cdo cdo -P 100 timmean "$intensity" "$intensity_mean_total"
run_cdo cdo -P 100 yearmean "$intensity" "$intensity_mean_yearly"
run_cdo cdo -P 100 monmean "$intensity" "$intensity_mean_monthly"

run_cdo cdo -P 100 timmax "$intensity" "$intensity_max_total"
run_cdo cdo -P 100 yearmax "$intensity" "$intensity_max_yearly"
run_cdo cdo -P 100 monmax "$intensity" "$intensity_max_monthly"


###################################
# 6. POST-PROCESSING & VISUALIZATION
###################################


# 6.1. Activate our virtual environment
source ___________/bin/activate
echo "Activated environment"
echo "Python: $(which python)"
echo "Path:___________/bin/python"

# 6.2. Numba & CPUs
if [ -z "$SLURM_CPUS_PER_TASK" ] || [ "$SLURM_CPUS_PER_TASK" = "" ]; then
    export NUMBA_NUM_THREADS=4
    echo "SLURM_CPUS_PER_TASK no defined, using 4 threads"
else
    export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
    echo "Using $NUMBA_NUM_THREADS threads (SLURM_CPUS_PER_TASK)"
fi

# 6.3. Verify installed packages 
echo ""
echo "Verifying Pytho packages:"
python -c "
import sys
print(f'Python {sys.version.split()[0]}')
import numpy as np; print(f'numpy {np.__version__}')
import xarray as xr; print(f'xarray {xr.__version__}')
import numba; print(f'numba {numba.__version__}')
print(f'NUMBA using {numba.config.NUMBA_NUM_THREADS} threads')
"

# 6.4. Run scripts for duration computation and plotting
echo ""
echo "Running extreme_events_visualizer.py..."
echo "========================================"

python extreme_events_duration.py "$var" "$depth"
python extreme_events_visualizer.py "$var" "$depth"

# 6.5. Deactivate environment
deactivate


echo "========================================"
echo "Final date: $(date)"
echo "Job completed"
echo "========================================"




echo "-- COMPLETED SH --"

echo "=== END OF THE COMPUTATION ==="
end=$(date +%s)
elapsed=$(( end - start ))
echo "Total execution time: $(( elapsed / 3600 )) hours"

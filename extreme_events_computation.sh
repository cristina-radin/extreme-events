#!/usr/bin/env bash

#SBATCH -J extremes
#SBATCH --mail-user=****@****.de
#SBATCH -p shared
#SBATCH --output ./extremes_out/job%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -A *****
#SBATCH --time=12:00:00
#SBATCH --mem=128G

####################################################################################################
# Project: AI4PEX (https://ai4pex.org/)
# Script: calcul_extremes_computation.sh
# Author: Cristina Radin (University of Hamburg)
# Contact: cristina.radin@uni-hamburg.de
# Adapted from Danai Filippou's work. 
# Date: 11.2025
#
# Description:
# This script computes extreme events (MHW, DEO, OAX) from ICON-COAST model output.
# It performs a full CDO-based preprocessing and extreme-event detection pipeline:
#
#   1. Preprocessing & Grid Remapping
#      - Extracts region, depth, and variable-of-interest (to, o2, hi→pH).
#      - Converts HI to pH if required.
#      - Shifts timestamps, remaps to target grid.
#      - Processes files in blocks for efficiency.
#
#   2. Merging
#      - Merges processed blocks into a continuous time series.
#
#   3. Reference Period
#      - Selects reference climatology for non-leap calendar.
#
#   4. Threshold Calculation
#      - Produces climatological thresholds.
#      - Computes monthly means for the threshold, reference and entire period.
#
#   5. Extreme Event Detection
#      - Computes daily anomalies relative to threshold.
#      - Generates binary extreme mask (≥ or ≤ threshold depending on variable).
#      - Computes consecutive-day duration, filters events by minimum duration (5 days).
#      - Computes intensity of extreme events.
#      - Mean and maximum intensities (total, yearly, monthly).
#      - Total, yearly, and monthly number of extreme days.
#
#
# Notes:
#   - Adjust configuration parameters (variable, depth, paths) as needed.
#   - This script is extendible for other datasets and configurations. 
#   - This script assumes ICON-COAST output is structured as daily files per chunk using a triangular grid.
#   - All temporary files are stored in a dedicated tmp/ directory and removed progressively.
#
####################################################################################################



module load cdo/2.5.0-gcc-11.2.0


#######################################
# CONFIG
#######################################

base_path="******"
target_grid="****"


var="to" #to, hi, o2 
depth=8 #8, 51, 104, 186.5, 489 m

if [ "$var" = "to" ]; then
    percentile=90
    comparison="gec"
    calculate_ph=0
    extreme="mhw"

elif [ "$var" = "o2" ]; then
    percentile=10
    comparison="lec"
    calculate_ph=0
    extreme="deo"

elif [ "$var" = "hi" ]; then
    percentile=10
    comparison="lec"
    calculate_ph=1
    extreme="oax"
fi

echo "FINAL CONFIGURATION: "
echo "var: $var"
echo "depth: $depth"
echo "percentile: $percentile"
echo "comparison: $comparison"
echo "calculate_ph: $calculate_ph"

out_dir="./extremes_out/${extreme}_${var}/depth${depth}"
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
# 1. PREPROCESSING FILES AND MERGE BY BLOCKS
#######################################

merged_blocks=()

for chunk in "$base_path"/chunk*; do
    [ ! -d "$chunk" ] && continue

    files=($(ls "$chunk"/hamocc_era5_244_cerosinc_pco2_cAR9_daily_[0-9][0-9]*T*.nc 2>/dev/null | sort))
    [ ${#files[@]} -eq 0 ] && continue

    tmp_list=()
    count=0

    for f in "${files[@]}"; do
        tmp_out=$(mktemp --tmpdir="$tmp_dir" --suffix=".nc")

            if [ "$var" = "hi" ]; then
                tmp_ph=$(mktemp --tmpdir="$tmp_dir" --suffix=".nc")
                run_cdo cdo -O -P 20 -expr,"pH=-log10(hi)" "$f" "$tmp_ph"
                input_for_processing="$tmp_ph"
                select_var="pH"
            else
                input_for_processing="$f"
                select_var="$var"
            fi

            run_cdo cdo \
                -sellonlatbox,$lon_min,$lon_max,$lat_min,$lat_max \
                -sellevel,$depth \
                -selvar,$select_var \
                -shifttime,-1day \
                -remapnn,"$target_grid" \
                "$input_for_processing" "$tmp_out"

            if [ "$var" = "hi" ]; then
                rm -f "$tmp_ph"
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
# 2. FINAL MERGE
#######################################

tmp_all_data_merged="$tmp_dir/tmp_all_data_merged.nc"
run_cdo cdo mergetime "${merged_blocks[@]}" "$tmp_all_data_merged"



#######################################
# 3. NO-LEAP + SUBSET REF PERIOD
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

run_cdo cdo -P 32 -ydrunpctl,90,5 "$ref_nl" "$min" "$max" "$tmp_thresh"
run_cdo cdo -settaxis,0001-01-01,00:00:00,1day "$tmp_thresh" "$thresh"


run_cdo cdo -P 32 monmean "$thresh" "$thresh_monmean"
run_cdo cdo -P 32 monmean "$ref_nl" "$ref_monmean"
run_cdo cdo -P 32 monmean "$all_data_nl" "$all_data_monmean"


rm -f "$tmp_thresh"



###################################
# 5. EXTREME EVENTS
###################################

tmp_anom="$tmp_dir/tmp_anomalies.nc" 
binary="$tmp_dir/binary.nc" 
tmp_count="$tmp_dir/tmp_count.nc" 
tmp_duration="$tmp_dir/tmp_duration.nc" 
duration="$tmp_dir/duration.nc" 
intensity="$tmp_dir/intensity.nc" 


run_cdo cdo -P 100 sub "$all_data_nl" "$thresh" "$tmp_anom" 
run_cdo cdo -P 100 ${comparison},0 "$tmp_anom" "$binary"  
run_cdo cdo -P 100 runsum,5 "$binary" "$tmp_count"
run_cdo cdo -P 100 gec,5 "$tmp_count" "$tmp_duration" 
run_cdo cdo -P 100 runmax,5 "$tmp_duration" "$duration" 
run_cdo cdo -P 100 mul "$binary" "$tmp_anom" "$intensity"


rm -f "$tmp_count" "$tmp_duration" "$tmp_anom"


###################################
days_total="$tmp_dir/days_total.nc"
days_yearly="$tmp_dir/days_yearly.nc"
days_monthly="$tmp_dir/days_monthly.nc"

intensity_mean_total="$tmp_dir/intensity_mean_total.nc"
intensity_mean_yearly="$tmp_dir/intensity_mean_yearly.nc"
intensity_mean_monthly="$tmp_dir/intensity_mean_monthly.nc"

intensity_max_total="$tmp_dir/intensity_max_total.nc"
intensity_max_yearly="$tmp_dir/intensity_max_yearly.nc"
intensity_max_monthly="$tmp_dir/intensity_max_monthly.nc"

run_cdo cdo -P 100 timsum "$duration" "$days_total"
run_cdo cdo -P 100 yearsum "$duration" "$days_yearly"
run_cdo cdo -P 100 monsum "$duration" "$days_monthly"

run_cdo cdo -P 100 timmean "$intensity" "$intensity_mean_total"
run_cdo cdo -P 100 yearmean "$intensity" "$intensity_mean_yearly"
run_cdo cdo -P 100 monmean "$intensity" "$intensity_mean_monthly"

run_cdo cdo -P 100 timmax "$intensity" "$intensity_max_total"
run_cdo cdo -P 100 yearmax "$intensity" "$intensity_max_yearly"
run_cdo cdo -P 100 monmax "$intensity" "$intensity_max_monthly"




echo "-- COMPLETED --"

echo "=== END OF COMPUTATION ==="
end=$(date +%s)
elapsed=$(( end - start ))
echo "Total execution time: $(( elapsed / 3600 )) hours"

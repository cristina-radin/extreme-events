# extreme-events

Extreme Events Computation Pipeline

This repository contains the workflow and scripts used to compute marine extreme events from high-resolution ocean model outputs. The workflow includes threshold computation, event detection, temporal aggregation, masking, and visualization.


📌 Overview
The pipeline performs the following steps:
1. Load SST or variable of interest from NetCDF files at all available depths.
2. Compute daily pixel-wise thresholds (e.g., 90th / 95th percentile) for each grid cell.
3. Identify extreme events by comparing with the corresponding threshold.
4. Generate time series of frequency, intensity and spatial patterns.



🚀 How to Run
1. Make the job script executable
chmod +x extreme_events_computation.sh
2. Submit the job
sbatch extreme_events_computation.sh



📞 Contact / Support
For questions regarding the processing pipeline, HPC execution, or data structure, don't hesitate to contact me. 

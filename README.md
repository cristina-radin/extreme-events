# extreme-events

Extreme Events Computation Pipeline

This repository contains the workflow and scripts used to detect and analize marine extreme events (Marine Heatwaves, Low Oxygen, Ocean Acidification) from high-resolution ICON-COAST model output.



📦 Quick Installation
- Clone repository
`git clone https://github.com/cristina-radin/extreme-events.git`

`cd extreme-events`

- Create Python environment
`python -m venv env_extremes`

`source env_extremes/bin/activate`

- Install dependencies
`pip install -r requirements.txt`




📌 Overview
The pipeline performs the following steps:
1. Load SST or variable of interest from NetCDF files at all available depths.
2. Compute daily pixel-wise thresholds (e.g., 90th / 95th percentile) for each grid cell.
3. Identify extreme events by comparing with the corresponding threshold.
4. Generate time series of frequency, intensity and spatial patterns.




🚀 How to Run
1. Submit the job:
`sbatch extreme_events_computation.sh`

2. Generate land mask (required once)
`python land_mask.py`


3. Generate animations (optional)

`python gif_generator.py to 8`                # Create GIF for temperature event




📞 Contact / Support
For questions regarding the processing pipeline, HPC execution, or data structure, don't hesitate to contact me. 

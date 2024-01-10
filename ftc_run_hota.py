# Take FISH.txt from current folder and move to the folder structure required

import os

file="FISH-yulun.txt"
TrackerName=file.split(".")[0]
tracker_folder=f"data/trackers/mot_challenge/FISH-train/{TrackerName}"
result_folder = os.path.join(tracker_folder, "data")

# Copy file to tracker folder
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
cmd=f'cp {file} "{result_folder}/FISH.txt"'
os.system(cmd)

# Run HOTA; See the results in the tracker folder (see above)
cmd=f'python scripts/run_mot_challenge.py --BENCHMARK FISH --SPLIT_TO_EVAL train \
    --TRACKERS_TO_EVAL {TrackerName} --METRICS HOTA --USE_PARALLEL False --NUM_PARALLEL_CORES 1 \
        --DO_PREPROC False'
os.system(cmd)

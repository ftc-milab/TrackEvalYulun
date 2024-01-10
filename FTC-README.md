# Create conda environment and install requirements as needed
```
conda create -n ftc310 python=3.10
pip install <WHATEVER NEEDED>
```

# Run HOTA
1. Copy your file name to the root folder
2. Change FILE in ftc_run_hota.py to your desired file
3. Run ftc_run_hota.py `python ftc_run_hota.py`
4. See the results in the shell; Details in the tracker_folder (jpg, pdf, summary, etc) `data/trackers/mot_challenge/FISH-train/<TrackerName>/`
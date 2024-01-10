# Take FISH.txt from current folder and move to the folder structure required

import os

file="FISH-yulun.txt"
TrackerName=file.split(".")[0]

os.makedirs(f"data/trackers/mot_challenge/FISH-train/{TrackerName}")
# #Perfect tracker
cmd=f'cp {file} "data/trackers/mot_challenge/FISH-train/{TrackerName}/FISH.txt"'
os.system(cmd)




def create_folders(exp_id=None,\
                        max_frames=50, \
                        weights_fn='best-organizers.pt',\
                        tracker_type = 'sort', \
                        max_age=1, \
                        min_hits=3, \
                        iou_threshold=0.3):
    # FISH folder
    
    
    #PerfectTracker folder
    fn=f"TrackEval/data/trackers/mot_challenge/FISH{exp_id}-train/PerfectTracker/data"
    if not os.path.exists(fn):
        os.makedirs(fn)

    # gt
    # TrackEval/data/gt/mot_challenge/FISH-train/FISH/gt/gt.txt
    fn=os.path.join(FISH,"gt/gt.txt")
    with open("TrackEval/data/gt/mot_challenge/FISH-train/FISH/gt/gt.txt","r") as f:
        with open(fn,"w") as g:
            frames=0
            for line in f:
                g.write(line)
                frames+=1
                if frames> max_frames*10-1:
                    break

    # #Perfect tracker
    cmd=f'cp {FISH}/gt/gt.txt "TrackEval/data/trackers/mot_challenge/FISH{exp_id}-train/PerfectTracker/data/FISH{exp_id}.txt"'
    os.system(cmd)
    

    #TrackEval/data/gt/mot_challenge/FISH-train/FISH/seqinfo.ini
    fn=os.path.join(FISH, "seqinfo.ini")
    with open(fn,"w") as f:
        f.write("[Sequence]\n")
        f.write(f"name=FISH{exp_id}\n")
        f.write(f"seqLength={max_frames}")

    # seqmaps
    seqmaps= "TrackEval/data/gt/mot_challenge/seqmaps"
    fns=[os.path.join(seqmaps, f'FISH{exp_id}-all.txt'),\
        os.path.join(seqmaps, f'FISH{exp_id}-test.txt'),\
        os.path.join(seqmaps, f'FISH{exp_id}-train.txt')]
    for fn in fns:
        with open(fn,"w") as f:
            f.write("NAME\n")
            f.write(f"FISH{exp_id}")

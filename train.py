"""
 Training audio samples should be placed in the "train" directory as such:
 > train
   > Midwest
       > Illinois
           Illinois 1.wav
           Illinois 2.wav
       > Indiana
           Indiana 1.wav
   > Northeast
       > Connecticut
           Connecticut 1.wav

"""

import os
import shutil
import glob
from pyAudioAnalysis import audioTrainTest as aT

# Moves all files to temporary directory
os.mkdir("tempTrain")
regions = glob.glob("train/*")
for region in regions:
    states = glob.glob(region + "/*")
    os.mkdir("tempTrain" + region[region.find("/"):])
    for state in states:
        paths = glob.glob(state + "/*")
        for path in paths:
            shutil.move(path, "tempTrain" + region[region.find("/"):])

# Gets list of directories to be trained on
dirs = ["tempTrain/" + directory for directory in os.listdir("tempTrain")]

# Trains model using randomforest
aT.extract_features_and_train(dirs, 1.0, 1.0, 0.1, 0.1, "randomforest", "model", False)

# Moves all files back to original directory
regions = glob.glob("tempTrain/*")
for region in regions:
    paths = glob.glob(region + "/*")
    for path in paths:
        end = path.find(" ")
        if not path[end+1:end+2].isdigit():
            end = path[end+1:].find(" ") + end + 1
        if end != -1:
            shutil.move(path, "train" + path[path.find("/"):end])

# Removes temporary directory
regions = glob.glob("tempTrain/*")
for region in regions:
    os.rmdir(region)
os.rmdir('tempTrain')

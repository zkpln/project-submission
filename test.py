"""
 Testing audio samples should be placed in the "test" directory as such:
 > test
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
import numpy as np
from pyAudioAnalysis import audioTrainTest as aT

# Moves all files to temporary directory
os.mkdir("tempTest")
regions = glob.glob("test/*")
for region in regions:
    states = glob.glob(region + "/*")
    os.mkdir("tempTest" + region[region.find("/"):])
    for state in states:
        paths = glob.glob(state + "/*")
        for path in paths:
            shutil.move(path, "tempTest" + region[region.find("/"):])

# Classifies all testing data and gets accuracy rate
correct = 0
totalTested = 0
regions = glob.glob("tempTest/*")
for region in regions:
    currentRegion = region[region.find("/")+1:]
    paths = glob.glob(region + "/*")
    for path in paths:
        Result, P, classNames = aT.file_classification(path, "model", "randomforest")
        if classNames[np.argmax(P)] == currentRegion:
            correct = correct + 1
        totalTested = totalTested + 1

# Moves all files back to original directory
regions = glob.glob("tempTest/*")
for region in regions:
    paths = glob.glob(region + "/*")
    for path in paths:
        end = path.find(" ")
        if not path[end+1:end+2].isdigit():
            end = path[end+1:].find(" ") + end + 1
        if end != -1:
            shutil.move(path, "test" + path[path.find("/"):end])

# Removes temporary directory
regions = glob.glob("tempTest/*")
for region in regions:
    os.rmdir(region)
os.rmdir('tempTest')

# Prints resulting accuracy rate
print("Accuracy: " + str(correct/totalTested * 100) + "%")

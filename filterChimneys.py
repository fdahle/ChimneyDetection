//Created by Felix Dahle

import os
import csv
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.stats.stats import pearsonr


inputFolder = "wholearea_output"
outputCSV = "final_chimneys.csv"

with open(outputCSV, 'w', newline='') as writeFile:

    filewriter = csv.writer(writeFile, delimiter=';')
    filewriter.writerow(
        ["filename", "segmentID", "numberOfPoints", "centroidX", "centroidY", "centroidZ", "maxZ", "avgHeight",
         "minHeight", "maxHeight", "divHeight", "spanX", "spanY", "spanZ", "ratioXHeight",
         "ratioYHeight", "ratioXY", "Correl", "filterType"])

    counter = 0
    for filename in os.listdir(inputFolder):
        counter = counter + 1
        print(counter)
        file = inputFolder + "/" + filename

        plydata = PlyData.read(file)

        x = plydata.elements[0].data['x']
        y = plydata.elements[0].data['y']
        z = plydata.elements[0].data['z']
        segments = plydata.elements[0].data["segment_id"]
        height = plydata.elements[0].data["height"]
        face_id = plydata.elements[0].data["face_id"]

        data = np.transpose(np.vstack([x, y, z, segments, height, face_id]))

        uniqueSegments = np.unique(segments)

        for segment_id in uniqueSegments:

            filterType = 1

            subset = data[data[:, 3] == segment_id]

            numberPoints = subset.shape[0]

            centroidX = round(np.average(subset[:, 0]), 3)
            centroidY = round(np.average(subset[:, 1]), 3)
            centroidZ = round(np.average(subset[:, 2]), 3)
            maxZ = round(np.amax(subset[:,2]), 3)

            avgHeight = round(np.average(subset[:, 4]), 3)
            minHeight = round(np.amin(subset[:, 4]), 3)
            maxHeight = round(np.amax(subset[:, 4]), 3)
            difHeight = round(maxHeight - minHeight, 3)

            spanX = round(np.amax(subset[:, 0]) - np.amin(subset[:, 0]), 3)
            spanY = round(np.amax(subset[:, 1]) - np.amin(subset[:, 1]), 3)
            spanZ = round(np.amax(subset[:, 2]) - np.amin(subset[:, 2]), 3)


            #filter non square stuff
            ratioXHeight = round(spanX / avgHeight, 3)
            ratioYHeight = round(spanY / avgHeight, 3)
            ratioXY = round(spanX / spanY, 3)
            if filterType == 1 and (ratioXY < 0.8 or ratioXY > 1.2):
                #print("ratio")
                filterType = 4
                data = data[data[:, 3] != segment_id]
                continue

            #filter non square stuff part 2
            correl = round(pearsonr(subset[:, 0], subset[:, 1])[0], 3)
            if filterType == 1 and (correl > 0.9 or correl < -0.9):
                #print("correl")
                filterType = 5
                data = data[data[:, 3] != segment_id]
                continue

            #filter small segments
            if filterType == 1 and (spanX < 0.2 or spanY < 0.2):
                #print("span too small")
                filterType = 2
                data = data[data[:, 3] != segment_id]

            if filterType == 1 and (spanX > 1.5 or spanY > 1.5):
                #print("span too high")
                filterType = 3
                data = data[data[:, 3] != segment_id]

            #filter too small or too tall stuff
            if filterType == 1 and avgHeight > 2:
                #print("height")
                filterType = 6
                data = data[data[:, 3] != segment_id]

            if (spanX == 0 or spanY == 0):
                ratioXY = -9999
                ratioXHeight = -9999
                ratioYHeight = -9999
                correl = -9999

            filewriter.writerow([filename, segment_id, numberPoints, centroidX, centroidY, centroidZ, maxZ, avgHeight,
                                 minHeight, maxHeight, difHeight, spanX, spanY, spanZ, ratioXHeight, ratioYHeight,
                                 ratioXY, correl, filterType])

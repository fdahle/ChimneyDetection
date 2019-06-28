//Created by Felix Dahle

import random
import os
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import KDTree

def cluster(startNr, endNr):
    inputFolder = "segmented"
    outputFolder = "clustered"
    searchRange = 0.5

    badFiles = []
    counter = 0
    for filename in os.listdir(inputFolder):
        counter = counter + 1


        print("File: ", counter)

        if counter < startNr - 1:
            continue

        if counter > endNr + 1
            continue

        if filename.endswith(".ply"):

            file = inputFolder + "/" + filename

            plydata = PlyData.read(file)

            x = plydata.elements[0].data['x']
            y = plydata.elements[0].data['y']
            z = plydata.elements[0].data['z']
            seg = plydata.elements[0].data['segment_id']

            points = np.vstack([x, y, z, seg]).transpose()


            #remove classified points
            subset = points[points[:,3] == 0]

            try:
               #create kdTree
                kdTree = KDTree(subset[:, 0:3])
            except ValueError:
                badFiles.append(counter)
                continue

            segmentId = 1
            waitingList = []

            while np.count_nonzero(subset[:,3]==0) > 0:

                #get unclassified points
                indices = np.where(subset[:, 3] == 0)[0]

                print(len(indices))
                if len(indices) > 0:
                    #append random unclassified point
                    randomElem = random.choice(indices)
                    waitingList.append(randomElem)

                while len(waitingList) > 0:

                    #remove elem
                    elem = waitingList.pop(0)

                    #set id
                    subset[elem, 3] = segmentId

                    #get nearest neighbours
                    nnPoints = kdTree.query_ball_point([subset[elem, 0], subset[elem, 1], subset[elem, 2]], searchRange)

                    #check if already classified
                    for _elem in nnPoints:
                        if subset[_elem, 3] == 0 and elem not in waitingList:
                            waitingList.append(_elem)

                segmentId = segmentId + 1

        # save np.array to file
        vertex = len(subset)
        output = outputFolder + "/" + filename
        lisr = ("ply", "format ascii 1.0", "element vertex {}".format(vertex), "property float x", "property float y",
                "property float z", "property uint segment_id", "end_header")
        np.savetxt(output, subset, fmt='%f %f %f %i', delimiter=' ', newline='\n', header="\n".join(lisr), comments='')

    print("Bad files: ")
    print(badFiles)

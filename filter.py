//Created by Felix Dahle

import os
import numpy as np
from plyfile import PlyData, PlyElement


inputFolder = "clustered"
outputFolder = "filtered"

counter = 0
for filename in os.listdir(inputFolder):
    counter = counter + 1
    print(counter)
    file = inputFolder + "/" + filename

    plydata = PlyData.read(file)

    x = plydata.elements[0].data['x']
    y = plydata.elements[0].data['y']
    z = plydata.elements[0].data['z']
    seg = plydata.elements[0].data['segment_id']

    points = np.vstack([x, y, z, seg]).transpose()

    unique, counts = np.unique(points[:, 3], return_counts=True)

    for i, elem in enumerate(counts):
        if elem < 25:
            points=points[points[:,3]!=i+1]

    # save np.array to file
    vertex = len(points)
    output = outputFolder + "/" + filename
    lisr = (
    "ply", "format ascii 1.0", "element vertex {}".format(vertex), "property float x", "property float y",
    "property float z", "property uint segment_id", "end_header")
    np.savetxt(output, points, fmt='%f %f %f %i', delimiter=' ', newline='\n', header="\n".join(lisr),
                comments='')

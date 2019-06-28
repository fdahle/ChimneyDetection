//Created by amber mulder

import numpy as np
from concave_hull import concave_hull
import os
from plyfile import PlyData, PlyElement
from laspy.file import File
from scipy.spatial import ConvexHull
from shapely.geometry import mapping, Polygon
import fiona
import matplotlib.pyplot as plt

inputFolder = "all_output"
inputFormat = "ply"
inputFileName = "" #if empty the folder will be searched for files
outputFolder = "polygons"

# Define schema for output shp
schema = {
    'geometry': 'Polygon',
    'properties': {'id': 'str','seg': 'int'},
}
poly = []
attr = []

def create_polygons(file):
    
    #print("Detecting convex hull")
    try:
        plydata = PlyData.read(file)
        
        x = plydata.elements[0].data['x']
        y = plydata.elements[0].data['y']
        z = plydata.elements[0].data['z']
        seg = plydata.elements[0].data['segment_id']
        
        #import las file
        # data = File(file, mode = "r")
        # x = data.x
        # y = data.y
        # z = data.z
        # seg = data.segment_id
    
    
        points = np.vstack([x, y, z, seg]).transpose()
    
        xy = points[:,0:2]
    
        if len(points) != 0:
            un_seg = np.unique(points[:,3])
    
            for i in un_seg:
                seg_p_i = np.where(points[:,3]==i) # indexes of points belonging to this segment
                xy_seg = xy[seg_p_i]
                    #print(xy_seg)
    
                if len(xy_seg) < 3 or len(np.unique(xy_seg[:,0])) == 1 or len(np.unique(xy_seg[:,1])) == 1:
                    pass

                else:     
                    #create a concave hull based on points and alpha shape parameter
                    concave = concave_hull(xy_seg,0.5)
                    
                    longest_idx = 0
                    for loop_idx in range(len(concave)):
                        if len(concave[longest_idx]) < len(concave[loop_idx]):
                            longest_idx = loop_idx
                    longest_loop = concave[longest_idx]
                    longest_loop.pop()
                    
                    poly.append(Polygon(xy_seg[longest_loop]))
                    attr.append([inputFileName, i])
    except Exception as e:
        print("Error message: ", e)

            
    return poly, attr
 


if inputFileName != "":
        inputFile = inputFolder + "/" + inputFileName + "." + inputFormat
else:
        counter = 0
        poly_all = []
        attr_all = []
        for filename in os.listdir(inputFolder):
            if filename.endswith(".ply"):
                inputFileName = filename.split(".")[0]                
                inputFile = inputFolder + "/" + inputFileName + "." + inputFormat
                poly, attr = create_polygons(inputFile)
                counter = counter + 1

                for polygon in poly:
                    poly_all.append(polygon)
                for attribute in attr:
                    attr_all.append(attribute)
       
        # write to shp file    
        with fiona.open('concave_results.shp', 'w', 'ESRI Shapefile', schema) as c:
            for j in range(len(poly_all)):
                c.write({
                        'geometry': mapping(poly_all[j]),
                        'properties': {'id': attr_all[j][0] , 'seg': attr_all[j][1]},
                        })

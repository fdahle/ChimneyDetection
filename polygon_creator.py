//created by Amber MUlder

import numpy as np
import collections
import os
from plyfile import PlyData, PlyElement
import laspy
from laspy.file import File
from scipy.spatial import ConvexHull
from shapely.geometry import mapping, Polygon
import fiona
import matplotlib.pyplot as plt #deze kan mis later weg

inputFolder = "cropped"
inputFormat = "ply"
inputFileName = "" #if empty the folder will be searched for files
outputFolder = "polygons"

## Define schema for output shp
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
        #data = File(file, mode = "r")
        #x = data.x
        #y = data.y
        #z = data.z
        #seg = data.segment_id


        points = np.vstack([x, y, z, seg]).transpose()
        #print(points)
        xy = points[:,0:2]
        #shift_value = np.mean(xy, axis = 0)
        #xy_shift = xy - shift_value
        

        #print(points)  
        if len(points) != 0:
            un_seg = np.unique(points[:,3])
            #print(un_seg)
            for i in un_seg:
                seg_p_i = np.where(points[:,3]==i) # indexes of points belonging to this segment
                xy_seg = xy[seg_p_i]
                    #print(xy_seg)
                #print('len unique x: ', len(np.unique(xy_seg[:,0])))
                #print('len unique y: ', len(np.unique(xy_seg[:,1])))

                if len(xy_seg) < 3 or len(np.unique(xy_seg[:,0])) == 1 or len(np.unique(xy_seg[:,1])) == 1:
                    
                    pass
                    #hull_pts = xy_seg # or should we then skip the segment?

                else:                  
                    hull = ConvexHull(xy_seg)
                    hull_indices = hull.vertices
                    #print(hull_indices)

                    '''
                    print(len(points))
                    print(file, i)
                    ## plot them
                    plt.plot(points[:,0], points[:,1], 'o')
                    for simplex in hull.simplices:
                        plt.plot(xy_seg[simplex, 0], xy_seg[simplex, 1], 'k-')
                    plt.plot(xy_seg[hull.vertices,0], xy_seg[hull.vertices,1], 'r--', lw=2)
                    plt.plot(xy_seg[hull.vertices[0],0], xy_seg[hull.vertices[0],1], 'ro')
                    plt.show()
                    '''
                        
                    hull_pts = xy_seg[hull_indices]
                    #hull_pts = hull_pts_shifted + shift_value
                    poly.append(Polygon(hull_pts))
                    attr.append([inputFileName, i])


    except Exception as e:
        print(e)


    # write to file        
    with fiona.open('objects_1906.shp', 'w', 'ESRI Shapefile', schema) as c:
        for j in range(len(poly)):
            c.write({
                'geometry': mapping(poly[j]),
                'properties': {'id': attr[j][0] , 'seg': attr[j][1]},
                })
 

objects_id = []
seg_chim = []
with open("test.csv", "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    for lines in csv_reader:
      objects_id.append(lines[0])
      seg_obj.append(lines[1])
      
objects_id_np = np.array(objects_id)
seg_obj_np = np.array(seg_chim)



if inputFileName != "":
        inputFile = inputFolder + "/" + inputFileName + "." + inputFormat
else:
        #counter = 0
        for filename in os.listdir(inputFolder):
            if filename.endswith(".ply"):
                if filename[0:-4] in objects_id:
                    which_seg_id = np.where(objects_id_np == filename[0:-4])
                    which_seg_str = seg_obj_np[which_seg_id]
                    which_seg = []
                    inputFileName = filename.split(".")[0]
                    inputFile = inputFolder + "/" + inputFileName + "." + inputFormat
                    create_polygons(inputFile)
                #counter = counter + 1
                #if counter == 100:
                    #print("FINISHED")
                    #break

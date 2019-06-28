//created by Amber Mulder

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
import csv

## check if I need al these modules
inputCSV = "".csv
inputFolder = "cropped"
inputFormat = "ply"
inputFileName = "" #if empty the folder will be searched for files
outputFolder = "polygons"


## Define schema for output shp
schema = {
    'geometry': 'Polygon',
    'properties': {'id': 'str','seg': 'int', 'type' : 'int'},
}
poly = []
attr = []

def create_polygons(file, seg_id, which_type):

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
                #print('hoi')
            for i in range(len(un_seg)):

                if un_seg[i] in seg_id:
                        #print('checkkk')
                    seg_p_i = np.where(points[:,3]==un_seg[i]) # indexes of points belonging to this segment
                    xy_seg = xy[seg_p_i]
                            #print(xy_seg)
                        #print('len unique x: ', len(np.unique(xy_seg[:,0])))
                        #print('len unique y: ', len(np.unique(xy_seg[:,1])))
                        #print('check')
                    if len(xy_seg) < 3 or len(np.unique(xy_seg[:,0])) == 1 or len(np.unique(xy_seg[:,1])) == 1:
                            
                        pass
                            #hull_pts = xy_seg # or should we then skip the segment?

                    else:                  
                        hull = ConvexHull(xy_seg)
                        hull_indices = hull.vertices
                            #print(hull_indices)
                            #print('check2')
                        '''
                        ## plot them
                        plt.plot(points[:,0], points[:,1], 'o')
                        for simplex in hull.simplices:
                            plt.plot(xy_seg[simplex, 0], xy_seg[simplex, 1], 'k-')
                        plt.plot(xy_seg[hull.vertices,0], xy_seg[hull.vertices,1], 'r--', lw=2)
                        plt.plot(xy_seg[hull.vertices[0],0], xy_seg[hull.vertices[0],1], 'ro')
                        plt.show()
                        '''   
                        hull_pts = xy_seg[hull_indices]
                            #print(hull_pts)
                            #hull_pts = hull_pts_shifted + shift_value
                        poly.append(Polygon(hull_pts))

                            #make_np = np.array(seg_id)
                            #make_np_t = np.array(which_type)
                        loct = seg_id.index(un_seg[i])
                        #print('test seg id = ', seg_id[loct])
                        #print('test un_seg[i]= ', un_seg[i])
                            #loct = np.where(make_np == un_seg[i])

                        t = which_type[loct]
                        #print(t)
                            #print('loc t = ', loct)
                            #t = make_np_t[loct]
                            #tt = t.tolist()
                            #t = tt[0]
                            #print(t)

                        attr.append([inputFileName, un_seg[i], t])
                        


    except Exception as e:
        print(e)


    # write to file        
    with fiona.open('objects_1906.shp', 'w', 'ESRI Shapefile', schema) as c:
        for j in range(len(poly)):
            c.write({
                'geometry': mapping(poly[j]),
                'properties': {'id': attr[j][0] , 'seg': attr[j][1], 'type' : attr[j][2]}})
 

chimney_id = []
seg_chim = []
type_chim = []
with open(inputCSV, "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    for lines in csv_reader:
      chimney_id.append(lines[0])
      seg_chim.append(lines[1])
      type_chim.append(lines[18])
      
      
chimney_id_np = np.array(chimney_id)
seg_chim_np = np.array(seg_chim)
type_chim_np = np.array(type_chim)

#print(chimney_id)

if inputFileName != "":
        inputFile = inputFolder + "/" + inputFileName + "." + inputFormat
else:
        #counter = 0
        for filename in os.listdir(inputFolder):
            #print(filename)
            if filename.endswith(".ply"):
                if filename in chimney_id:
                    which_seg_id = np.where(chimney_id_np == filename)
                    which_seg_str = seg_chim_np[which_seg_id]
                    which_type_str = type_chim_np[which_seg_id]
                    which_seg = []
                    which_type = []
                    for n in which_seg_str:
                        which_seg.append(int(n))

                    for k in which_type_str:
                        which_type.append(int(k))

                    #print('len1 =', len(which_seg), 'len2 =',len(which_type))

                    inputFileName = filename.split(".")[0]
                    inputFile = inputFolder + "/" + inputFileName + "." + inputFormat
                    create_polygons(inputFile, which_seg, which_type)
                #counter = counter + 1
                #if counter == 100:
                    #print("FINISHED")
                    #break

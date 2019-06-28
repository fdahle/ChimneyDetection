@author: Vasileios Alexandridis NEW VERSION

import time
import numpy as np
from shapely.geometry import Point
from shapely.geometry import Polygon
import os
from scipy.linalg import eigh
import plyfile

#read obj input file
def read_obj(bag_id):
    
    //Path to obj file with bagId as name
    pathnew = ""
    
    #open obj file
    with open(pathnew, 'r') as obj:
        data= obj.read()
        
        #read the lines
        lines = data.splitlines()
        
        #read every line and save the vertices and faces in different lists
        vertices = []
        faces = []
        normal = []
        lista=[]
        
        for line in lines:
            element = line.split()
            if element[0] == 'v':
                vertices.append([float(element[1]),float(element[2]),float(element[3])])
            elif element[0] == 'vn':
                normal.append([float(element[1]),float(element[2]),float(element[3])])
            elif element[0] == 'f':
                t=[]
                for i in range(1,len(element)):
                    t.append([int(e) for e in element[i].split('/') if e])
                lista.append(t)
        for i in range(len(lista)):
            lista2=[]
            for j in lista[i]:
                lista2.append(j[0])
            faces.append(lista2)
            
    return vertices,faces

#read ply input file
def read_ply(bag_id):
    
    
    pathnew = "C:\\Users\\user\\Desktop\\detection\\example_segments\\plyfiles\\" + str(bag_id) +".ply"
    
    #read ply file
    plydata = plyfile.PlyData.read(pathnew);
    
    #store the coordinates
    x = plydata.elements[0].data['x']
    y = plydata.elements[0].data['y']
    z = plydata.elements[0].data['z']
    seg = plydata.elements[0].data['segment_id']
    
    points = np.vstack([x, y, z, seg]).transpose()
    
    return points


#write ply output file with x,y,z,segment_id, height difference, face_id
def write_ply(file,outputName):

    //Path to ply file with bagId as name
    pathnew = ""
    
    #check if there are points to write in the output ply file
    if len(file)>0:
        vertex = len(file)
        lisr = ("ply", "format ascii 1.0", "element vertex {}".format(vertex), "property float x", "property float y", "property float z", "property uint segment_id","property float height","property float face_id","end_header")
        np.savetxt(pathnew, file, fmt='%f %f %f %i %f %f', delimiter=' ', newline='\n', header="\n".join(lisr), comments='')
    else:
        print("No points to write")
  
#fit plane function      
def fit_plane(pts):

    # shift points to mean
    mean = np.mean(pts, axis = 0)  
    pts -= mean
    
    # compute covariance matrix and eigenvalues and eignevectors
    cov = np.cov(pts, rowvar = False)
    evals, evecs = eigh(cov)
    
    # find smallest eigenvalue, the corresponging eigenvector is our normal n
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]

    n = evecs[:,-1]
    c = mean

    return c, n

#retrieve the coordinates of every face
def getcoordinates (faces,vertices):  
    
    points2d = []
    points3d = []
    for f in faces:
        p=[]
        for index in f:
            x,y,z = vertices[index-1]
            p.append([x,y,z])
        p_array = np.array(p)
        points2d.append(p_array[:,:2])
        points3d.append(p_array)
        
    return points2d, points3d


#return pointcloud points that lies on the 2d polygon of every roof surface (face)
def check_point(points2d, points3d, dataset):
    
    #2d points of the poincloud dataset
    dataset2d = dataset[:,:2]
       
    final=[]
    boolvalues=[] #to test the overlap
    
    #point2d are the points of every face - roof part
    for i in range(len(points2d)):
        poly = Polygon(points2d[i])
        
        boolvalue = [0,0]
        
        inside=[]
        for p in range(len(dataset2d)):  #input pointcloud
            point = Point(dataset2d[p])
            boolean = poly.contains(point)  
            
            zvalue_face = np.mean(points3d[i][:,2])
            
            if  boolean is True:
                if zvalue_face > boolvalue[1]:
                    boolvalue =[i,zvalue_face]
                inside.append(dataset[p])
                
        boolvalues.append(boolvalue)       
        final.append(inside)     
        
    return final, boolvalues

def points_above_roof(inside_points,points3d):
    
    #dictionary with planes
    planes = {}
    for i in range(len(points3d)):
        planepoints = points3d[i]
        centroid, normal = fit_plane(planepoints) #fit the plane
        d_parameter =  -(normal[0] * centroid[0] + normal[1] * centroid[1] + normal[2] * centroid[2]) #calculate the d parameter of the plane
        planes[i] = [centroid, normal, d_parameter] #store the 3 values in the dictionary
    
    #intersection points between the ply points and the plane
    intersections = []
    for i in range(len(inside_points)): #iterate through the corresponding poincloud points of every face
        points = inside_points[i]
        centroid , normal, d_parameter = planes[i] #retrieve the plane parameters
        
        for p in points:  #calculate the intersection points with the plane
            x,y,z,seg = p
            point = [x,y,z,seg]
            
            intersect = point
            
            #z value from the plane
            z = - (normal[0] * intersect[0] + normal[1] * intersect[1] + d_parameter)/normal[2]
            
            #height difference among the original z value and the plane z value on every point
            diff = intersect[2]-z
            
            if diff > 0:
                intersections.append([intersect,diff,i])         
                       
    return intersections


#----MAIN CODE----#

//Path to the folder with ply files
path_plyfiles = ""

//Path to the folder with obj files
path1_objfiles = "" 

for plyf in os.listdir(path_plyfiles):
    
    #start execution time
    start_time = time.time()
    
    #read the ply-pointcloud files
    file_ply = plyf.split('.ply')
    input_ply = file_ply[0]
    
    objfile = path1_objfiles + input_ply + '.obj'

    if os.path.exists(objfile):

        input_obj = input_ply
        
        #get the vertices and faces from the obj file
        vertices, faces = read_obj(input_obj)

        #read the pointcloud
        dataset = read_ply (input_ply)

        #store 2d,3d coordinates of the vertices of each face
        points2d, points3d = getcoordinates (faces, vertices)

        #check if the point is inside or outside of a polygon
        inside_points, boolvalues = check_point(points2d, points3d,dataset)
        
        #points above roofs
        above_roofs = points_above_roof(inside_points,points3d)
        
        #write them in a nice table
        output= []
        for i in range(len(above_roofs)):
            x,y,z,seg = above_roofs[i][0]
            diff = above_roofs[i][1]
            face_id = above_roofs[i][2]
            output.append([x,y,z,seg,diff,face_id])
        output = np.array(output)
        
        #print statements to check the output file
        print("Number of points: ",len(output))

        
        #write ply file
        outputName = input_ply + "_above_roofs" + ".ply"
        write_ply(output,outputName)
        
        #execution time
        print("--- %s seconds ---" % (time.time() - start_time))

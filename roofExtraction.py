//Created by Yifang Zhao

import os
from laspy.file import File
import numpy as np
import csv
from scipy import spatial as sp
from scipy.linalg import eigh
from plyfile import PlyData
from scipy.spatial import Delaunay


    
# check if candidate is valid (in a certain distance to a plane)
# this function is written by Felix Dahle
def valid_candidate(test_point, seed_point, normal, maxdist):
    diff = test_point - seed_point
    dist = np.dot(diff, normal)
    if np.abs(dist) <= maxdist:
        return True
    return False

# this function is written by Felix Dahle
def segment(file, output, minimal_segment_count = 200, neighbourhood_type = "knn", epsilon = 0.3, k = 20, r = 1.0):
    """
    Segment a set of 3D points and assignment each a segment id.
    
    Input:
        file: point cloud file (las) that is a building.
        minimal_segment_count: the minimal required number of a segment.
        neighbourhood_type: 'knn' or 'radius'.
        epsilon: minimum required distance when assigning one point to a segment.
        k: specified when 'knn' is used. the number of neighbours to be returned.
        r: specified when 'radius' is used. The search radius.
    Output:
        an array of 3D points plus a column indicating the segment id.
    """
#    print("    Start Segmentation")

    # get File
    infile = File(file, mode='r')

    # get points
    x, y, z = infile.x, infile.y, infile.z
    buildingPoints = np.vstack((x, y, z)).transpose()
#    print(len(buildingPoints))
#    return

    # thinning (nth point)
    # no_ground_xyz = no_ground_xyz[::jparams['thinning-factor']]

    # make new list with segment_id column
    zeros = np.full((len(buildingPoints), 1), 0)
    nogr_xyzid = np.hstack((buildingPoints, zeros))

    # create kdTree for points
    tree = sp.cKDTree(buildingPoints)

    # set parameters for later use
    region_id = 0
    idx = -1
    loop_incr = 100
    ransac_iter = 10

    while idx < len(nogr_xyzid)-loop_incr:

        # set loop parameter
        stack = []
        stack_counts = []
        seedsurface_list = []

        # increment counter
        idx += loop_incr

        # Already classified this point
        if nogr_xyzid[idx,3] != 0:
            continue

        # get point and list of its neighbours
        pt = nogr_xyzid[idx,0:3]
        if neighbourhood_type == 'knn':
            point_dist, seedsurface_list = tree.query(pt, k)
        elif neighbourhood_type == 'radius':
            seedsurface_list = tree.query_ball_point(pt, r)

        # if list of neighbours < 3, cannot fit plane
        if len(seedsurface_list) < 3:
            continue

        # RANSAC - tested number of subsets to create seed surface can be varied
        seedsurface_list = np.squeeze(seedsurface_list)
        for iteration in range(ransac_iter):
            tmp_stack = []
            tmp_stack_counts = []
            random_idxs = np.random.choice(len(seedsurface_list), 3, replace=False)
            seed_point, normal = fit_plane(np.squeeze(nogr_xyzid[seedsurface_list[random_idxs],0:3]))
            # determine size of consensus
            for elem in seedsurface_list:
                if nogr_xyzid[elem,3] == 0 and valid_candidate(nogr_xyzid[elem, 0:3], seed_point, normal, epsilon):
                    tmp_stack.append(nogr_xyzid[elem, 0:3])
                    tmp_stack_counts.append(nogr_xyzid[elem])
            # store only the largest consensus
            if len(tmp_stack) > len(stack):
                stack = tmp_stack
                stack_counts = tmp_stack_counts

        # The candidate seed surface did not have enough members - minimum 60% of neighbours list should be valid
        if len(stack) < 0.6*len(seedsurface_list) or len(stack) < 3:
            continue
        else:
            region_id+=1
            for elem in stack_counts:
                elem[3] = region_id

        # get fitted plane for seed surface
        seed_point, normal = fit_plane(stack)#

        # initial count of region points (in seed surface)
        prev_region_count = len(stack_counts)
        curr_region_count = prev_region_count

        # Region growing
        while len(stack) > 0:
            # Re-estimate plane after stack has grown by at least 30%
            if curr_region_count > 1.3*prev_region_count:
                seed_point, normal = fit_plane(np.array(stack_counts)[:,0:3])
                # update count of region points after growing a percentage
                prev_region_count = curr_region_count

            x = stack.pop()

            # two possible methods
            if neighbourhood_type == 'knn':
                point_dist, neighbours_list = tree.query(x, k)
            elif neighbourhood_type == 'radius':
                neighbours_list = tree.query_ball_point(x, r)

            # look for valid candidates, assign region_id and keep track of region count
            for elem in neighbours_list:
                if nogr_xyzid[elem,3] == 0 and valid_candidate(nogr_xyzid[elem, 0:3], seed_point, normal, epsilon):
                    nogr_xyzid[elem,3] = region_id
                    # append x,y,z of point to be used in stack
                    stack.append(nogr_xyzid[elem, 0:3])
                    # append x,y,z of all points thusfar to be used in fitting new plane
                    stack_counts.append(nogr_xyzid[elem])
                    # update count of growing region
                    curr_region_count += 1


        else:
            # check if fully grown segment contains enough points
            if len(stack_counts) < minimal_segment_count:
                for elem in stack_counts:
                    elem[3] = 0
                region_id -= 1
    
#    return nogr_xyzid
    print("    1. Write segmented ply file")
    # save np.array to file
    # print(nogr_xyzid)
    vertex = len(nogr_xyzid)
    lisr = ("ply", "format ascii 1.0", "element vertex {}".format(vertex), "property float x", "property float y",
            "property float z", "property uint segment_id", "end_header")
    np.savetxt(output, nogr_xyzid, fmt='%f %f %f %i', delimiter=' ', newline='\n', header="\n".join(lisr), comments='')
#    print("    Write segmented las file.")
#    write_las(nogr_xyzid, infile, output)


def extract_roof(file, outputObj, alpha):
    """
    Input:
        file: .ply, segmented point cloud per building
        alpha: threshold for alpha shape
    Output: .obj, roof surfaces 
    """
#    print("    Start Extraction")
    
    plydata = PlyData.read(file)

    x = plydata.elements[0].data['x']
    y = plydata.elements[0].data['y']
    z = plydata.elements[0].data['z']
    seg = plydata.elements[0].data['segment_id']

    points = np.vstack([x, y, z, seg]).transpose()
    

#    # get File
#    infile = File(file, mode='r')
#
#    # get points
#    x, y, z, segment_ids = infile.x, infile.y, infile.z, infile.segment_id
#    points = np.vstack((x, y, z, segment_ids)).transpose()
    
    # all vertices and faces written to  .ply file
    vertices = []
    faces = []
    
    # vertex index
    i = 0
    #segment id
    segment_id = 1
    
    
    # points that are going to be outputed
    # without roof points and wall points
#    points_output = np.empty((0, 4), dtype = float, order = 'C')
    
    
    # select points according to segment_id
    # ignore value 0, namely, the unsegmented
    pts = points[points[:,3]==segment_id]
    while len(pts) > 2:
#        print ("    Processing segment {}".format(segment_id))
        # one face consists of indices of vertices
        face = []
        
        c, n = fit_plane(pts[:, :3])
        if (abs(n[2]) > 0.34):
#            plot_delaunay_alphaShape_concaveHull(pts[:, :2], alpha, "segment{}".format(segment_id))
            hull = concave_hull(pts[:, :2], alpha)
#            print("    contains {} loops".format(len(hull)))
            
            # find the index of the longest loop
            # in this way, the inner rings are neglected as well as some possible small outer rings
            longest_idx = 0
            for loop_idx in range(len(hull)):
                if len(hull[longest_idx]) < len(hull[loop_idx]):
                    longest_idx = loop_idx
            longest_loop = hull[longest_idx]
            
            # remove small&vertical surfaces
            # thresholds:
            #     minimum area: 5 m^2
            #     minimum angle between face normal and horizontal plane: 20 (sin20 = 0.34)
            
#            if get_area_single_polygon(pts, longest_loop) > 5:
            longest_loop.pop() # the last vertex is the same as the first one
            for vertex in longest_loop:
                vertex_x = pts[vertex, 0]
                vertex_y = pts[vertex, 1]
                vertex_z = (c[2]*n[2] - n[0]*(vertex_x-c[0]) - n[1]*(vertex_y-c[1])) / n[2]
                vertices.append([vertex_x, vertex_y, vertex_z])
                i += 1
                face.append(i)
            faces.append(face)
#            else:
#                # if the area of a segment is too small, it is regarded as a small structure, probably a chimney
#                # and should be outputed
#                points_output = np.append(points_output, pts, axis = 0)
#                print("        small area!")
#        else:
#            print("        steep segment!")
        
        segment_id += 1
        pts = points[points[:,3]==segment_id]
    
#    print("    2. Write segmented ply file without W&R.")
#    # save np.array to file
#    # print(nogr_xyzid)
#    vertex = len(points_output)
#    lisr = ("ply", "format ascii 1.0", "element vertex {}".format(vertex), "property float x", "property float y",
#            "property float z", "property uint segment_id", "end_header")
#    np.savetxt(outputPly, points_output, fmt='%f %f %f %i', delimiter=' ', newline='\n', header="\n".join(lisr), comments='')
#    
    print("    2. Write obj file.")
    write_faces_to_obj(vertices, faces, outputObj)
    
def get_area_single_polygon(points, single_polygon):          
    """
    Function that calculates the area of the input single polygon
    
    Input:
        single_polygon:   a ring of the polygon represented by a list of point indices.
    Output:
        returns the value of the area
    """
    x,y = [points[c][0] for c in single_polygon], [points[c][1] for c in single_polygon]
    area = 0
    for i in range(len(single_polygon) - 1):
        area += 0.5 * (x[i] * y[i + 1] - x[i + 1] * y[i])
    return abs(area)


# create plane out of points
def fit_plane(points):

    # shift points to mean
    mean = np.mean(points, axis = 0)
    pts = points - mean
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


def write_faces_to_obj(pts, faces, obj_filename):
#    print("=== Writing {} ===".format(obj_filename))
    f_out = open(obj_filename, 'w')
    for p in pts:
        f_out.write("v {:.3f} {:.3f} {:.3f}\n".format(p[0], p[1], p[2]))
    for face in faces:
        f_out.write("f")
        for idx in face:
            f_out.write(" {}".format(idx))
        f_out.write("\n")
    f_out.close()



########################Concave hull*************************************************
def sq_norm(v): #squared norm 
    return np.linalg.norm(v)**2

def circumcircle(points,simplex):
    A=[points[simplex[k]] for k in range(3)]
    M=[[1.0]*4]
    M+=[[sq_norm(A[k]), A[k][0], A[k][1], 1.0 ] for k in range(3)]
    M=np.asarray(M, dtype=np.float32)
    S=np.array([0.5*np.linalg.det(M[1:,[0,2,3]]), -0.5*np.linalg.det(M[1:,[0,1,3]])])
    a=np.linalg.det(M[1:, 1:])
    b=np.linalg.det(M[1:, [0,1,2]])
    return S/a,  np.sqrt(b/a+sq_norm(S)/a**2) #center=S/a, radius=np.sqrt(b/a+sq_norm(S)/a**2)

def get_alpha_complex(alpha, points, simplexes):
    #alpha is the parameter for the alpha shape
    #points are given data points 
    #simplexes is the  list of indices in the array of points 
    #that define 2-simplexes in the Delaunay triangulation
    return filter(lambda simplex: circumcircle(points,simplex)[1]<alpha, simplexes)

def concave_hull(pts, alpha):
    """
    Compute the concave hull of a set of 2D points.
    Input:
        pts: a set of 2D points.
        alpha: threshold for alpha shape generation.
    Output:
        Concave hull of the given points which is represented by a list of several lists of the point indices.
        It may contain inside rings as well as islands.
    """
    points = pts - np.mean(pts, axis = 0)
    tri = Delaunay(points)
#    plt.triplot(points[:, 0], points[:, 1], tri.simplices)
#    print("input pts number: {}, after triangulation: {}".format(len(pts), len(tri.simplices)))
    alpha_complex=list(get_alpha_complex(alpha, points, tri.simplices))
    #record frequency of edges
    f_edge = {}
    for simplex in alpha_complex:
        for j in range(-1, 2):
            vertex1 = simplex[j]
            vertex2 = simplex[j + 1]
            
            # if this edge is found in f_edge,
            # i.e., its duplicate is found in f_edge,
            # then, count + 1
            found = False
            for edge in f_edge.keys():
                v1 = edge[0]
                v2 = edge[1]
                if (vertex1 == v1 and vertex2 == v2) or (vertex1 == v2 and vertex2 == v1):
                    found = True
                    f_edge[edge] += 1
                    break
                    
            # if this edge never appear in f_edge
            if found == False:
                f_edge[(vertex1, vertex2)] = 1
            
#    print(f_edge)
    # remove duplicate edges
    boundary_unordered = [e for e in f_edge.keys() if f_edge[e] == 1]
#    print(boundary_unordered)
    boundary_ordered = []
    oneLoop = []
    v1, v2 = boundary_unordered.pop(0)
    oneLoop.append(v1)
    oneLoop.append(v2)
    while len(boundary_unordered) > 0:
        flag = False
        for e in boundary_unordered:
            # next edge found
            if e[0] == oneLoop[-1]:
                flag = True
                oneLoop.append(e[1])
                boundary_unordered.remove(e)
                break
            
        # next edge not found in left vertices, i.e., one loop formed
        if flag == False:
            boundary_ordered.append(oneLoop)
            oneLoop = []
            v1, v2 = boundary_unordered.pop(0)
            oneLoop.append(v1)
            oneLoop.append(v2)
        elif len(boundary_unordered) == 0:
            boundary_ordered.append(oneLoop)
#    print(boundary_ordered)
    return boundary_ordered
########################Concave hull*************************************************



Already_done = 0
Done_ahn3 = 0
Done_dim = 0
Fail = 0
NotFound = 0
counter = 0
# for those not available in AHN3, DIM has to be used

###################################################################
########################This is all you need to modify######################
BAG_divided = """.csv" # your bag_id file
inputFolder1 = "" # AHN3 path
inputFolder2 = "" # segmented DIM from Felix
interFolder = "" # segmented AHN3 going to be generated by this script
outputFolder = "" # obj roofs going to be generated by this script
###################################################################
###################################################################

inputFormat = "las"
interFormat = "ply"
outputFormat = "obj"


bag_ids = []
with open(BAG_divided) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        bag_ids.append(row[0])

print("Number of bag buildings found in tiles: {}".format(len(bag_ids)))


with open("log.txt", "w") as errors:
    for bag_id in bag_ids:
        counter += 1
    #    print(inputFileName)
        inputFile1 = inputFolder1 + "/" + bag_id + "." + inputFormat
        inputFile2 = inputFolder2 + "/" + bag_id + "." + interFormat
        interFile = interFolder + "/" + bag_id + "." + interFormat
    #    interFile2 = interFolder2 + "/" + inputFileName + "." + interFormat
        outputFile = outputFolder + "/" + bag_id + "." + outputFormat
    
        print("\nWith {}. {} ({:.2%} finished): ".format(counter, bag_id, counter / len(bag_ids)))
        try:
            if os.path.exists(outputFile):
                print("    Already exists!")
                Already_done +=1
            elif os.path.exists(inputFile1):
                print("    Found in AHN3...")
                segment(inputFile1, interFile, minimal_segment_count = 100, neighbourhood_type = "knn", epsilon = 0.1, k = 20, r = 0.7)
                extract_roof(interFile, outputFile, 0.7)
                Done_ahn3+=1
            elif os.path.exists(inputFile2):
                print("    Found in DIM...")
                extract_roof(inputFile2, outputFile, 0.2)
                Done_dim+=1
            else:
                print("    Not found!")
                errors.write("{}:\nNot found in both BAG and DIM.\n\n".format(bag_id))
                NotFound +=1
        except Exception as e:
            Fail+=1
            errors.write("{}:\n{}\n\n".format(bag_id, str(e)))
        
        
#    else:
#        print("    Not found.")
#        missing +=1
print("Done_ahn3: {}, Done_dim: {}, NotFound: {}, Fail: {}".format(Done_ahn3, Done_dim, NotFound, Fail))

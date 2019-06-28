//Created by Felix Dahle

import laspy
import os
import numpy as np

from scipy import spatial as sp
from scipy.linalg import eigh

minimal_segment_count = 150
neighbourhood_type = "knn"
epsilon = 0.4
k = 20
r = 1.0

inputFolder = "DIM"
inputFormat = "las"
inputFileName = ""
outputFolder = "segmented"

# check if candidate is valid (in a certain distance to a plane)
def valid_candidate(test_point, seed_point, normal, maxdist):
    diff = test_point - seed_point
    dist = np.dot(diff, normal)
    if np.abs(dist) <= maxdist:
        return True
    return False


# create plane out of points
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


def detect_planes(file):

    print("Start Segmentation")

    # get File
    infile = laspy.file.File(file, mode='r')

    # filter out non-ground points
    no_ground = np.where(np.logical_not(infile.raw_classification ==2))

    # get filtered points and create xyz array (scaled)
    x, y, z = infile.x[no_ground], infile.y[no_ground], infile.z[no_ground]
    no_ground_xyz = np.vstack((x, y, z)).transpose()

    # thinning (nth point)
    # no_ground_xyz = no_ground_xyz[::jparams['thinning-factor']]

    # make new list with segment_id column
    zeros = np.full((len(no_ground_xyz), 1), 0)
    nogr_xyzid = np.hstack((no_ground_xyz, zeros))

    # create kdTree for points
    tree = sp.cKDTree(no_ground_xyz)

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

    # save np.array to file
    vertex = len(nogr_xyzid)
    output = outputFolder + "/" + inputFileName + ".ply"
    lisr = ("ply", "format ascii 1.0", "element vertex {}".format(vertex), "property float x", "property float y",
            "property float z", "property uint segment_id", "end_header")
    np.savetxt(output, nogr_xyzid, fmt='%f %f %f %i', delimiter=' ', newline='\n', header="\n".join(lisr), comments='')

if inputFileName != "":
        inputFile = inputFolder + "/" + inputFileName + "." + inputFormat
else:
        counter = 0
        for filename in os.listdir(inputFolder):
            if filename.endswith(".las"):
                if counter < 0:
                    counter = counter + 1
                else:
                    inputFileName = filename.split(".")[0]
                    inputFile = inputFolder + "/" + inputFileName + "." + inputFormat
                    detect_planes(inputFile)
                    counter = counter + 1
                    print(counter)
                    print("FINISHED")

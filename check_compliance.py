"""
@author: Yifag Zhao
"""

import csv
import json
import fiona
import shapely.geometry as sg
from shapely.geometry import Point


def find_buildings_nearby(point, bagFile, radius):
    """
    Find all the buildings in bagfile that fall inside the circle.
    Input:
        point: a 2D point
        radius: 
        bagfile: shape file containing bag buildings
    Output:
        bag_ids: all bag building ids falling inside the circle
    """
    bag_ids = []
    bag_file = fiona.open(bagFile)
    pt = Point(point[0], point[1])
    for each in bag_file:
        if pt.distance(sg.shape(each["geometry"])) < radius:
            bag_ids.append(str(each["properties"]["pandid"]).split('.')[0])
    return bag_ids

def check_compliance(objectFile, bagFile, jsonFile, outputFile, radius = 25):
    """
    Check for every object in objectFile if it complies with the safety rules.
    Input:
        objectFile: all objects in the 5 tiles. a csv file
        bagFile: 
        jsonFile: heights of all buildings BAG
        outputFile: csv file used for implicit geometry
        radius:
    """
    heights = json.load(open(jsonFile))
    with open(objectFile, mode="r") as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=';')
        with open(outputFile, 'w', newline='') as writeFile:
            filewriter = csv.writer(writeFile, delimiter=';')
            filewriter.writerow(["pandid", "segmentID", "centroidX", "centroidY", "centroidZ", "spanX", "spanY", "maxHeight", "compliance", "filterType"])

            for row in csv_reader:
                x = float(row["centroidX"])
                y = float(row["centroidY"])
                z = float(row["centroidZ"])
                maximumZ = float(row["maxZ"])
                flag = "Y"
                bag_ids = find_buildings_nearby((x, y), bagFile, radius)
                for bag_id in bag_ids:
                    if bag_id in heights.keys() and heights[bag_id] > maximumZ - 2:  # compliance check
                        flag = "N"
                        break
                filewriter.writerow([row["filename"][:15], row["segmentID"], x, y, z, row["spanX"], row["spanY"], row["maxHeight"], flag, row["filterType"]])

            
        
        
if __name__ == "__main__":

    //Path to csv file
    objectFile = r""
    
    //Path to shp file
    bagFile = r""
    
    //path to json file
    jsonFile = ""
    
    outputFile = r"compliance.csv"
    
    check_compliance(objectFile, bagFile, jsonFile, outputFile)

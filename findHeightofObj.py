"""
@author: Yifang Zhao
"""

import os
import json

if __name__ == "__main__":

    //path to folder with obj files
    basepath = ""
    
    //path to json file
    jsonFile = ''
    
    heights = {}
    for entry in os.listdir(basepath):
        with open(os.path.join(basepath, entry), 'r') as obj:
            z = []
            for line in obj.readlines():
                if line.split(' ')[0] == 'v':
                    z.append(float(line.split(' ')[-1]))
            heights[entry.split('.')[0]] = max(z)
    
    print("Find heights for {} buildings.\nSave them in {}.".format(len(heights), jsonFile))
    with open(jsonFile, 'w') as f:
        json.dump(heights, f)

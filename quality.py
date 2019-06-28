//created by Amber Mulder

from shapely.geometry import mapping, Polygon 
import fiona
import shapefile    # pip install pyshp==1.1.4
                    # https://pythonhosted.org/Python%20Shapefile%20Library
import csv
import numpy as np
import pandas as pd
#from pandas_ml import ConfusionMatrix

# USER INPUT:
print('PREPARING DATA / ')
dc_shp = shapefile.Reader("") # Import detected chimney data
gt_shp = shapefile.Reader("") # Import ground truth data. In schema: first property has to be the id!
thresh = 10 # minimum overlap percentage required 
chimney_csv = "test.csv"                    
# define schema shapefile output
schema = {
    'geometry': 'Polygon',
    'properties': {'id_dc': 'str', 'id_gt': 'str', 'perc': 'int', 'seg' : 'int', 'chim' :'str'}, 
}

# get shapes input files
dc_chim = dc_shp.shapes() 
gt_chim = gt_shp.shapes()
prop_dc = dc_shp.records()
prop_gt = gt_shp.records()
print('PREPARING DATA \ ')
intersections = []
per_int = 0
print('CHECK INTERSECTIONS / ')
for i in range(len(dc_chim)):
    for j in range(len(gt_chim)):
        pol1 = Polygon(dc_chim[i].points)   # a detected potential chimny by algorithm
        pol2= Polygon(gt_chim[j].points)    # a ground truth chimney
        if pol1.intersects(pol2):
            inter = pol1.intersection(pol2)
            overlap = (inter.area/pol2.area)*100 # percentage of overlap
            if overlap >= thresh:
                rel_area = pol1.area / pol2.area
                if rel_area >= 0.4 and rel_area <= 2.5:
                    intersections.append([dc_chim[i], gt_chim[j], inter, overlap, i, j, prop_gt[j][1], prop_dc[i][0], prop_dc[i][1]])
                    per_int += overlap # at the end this will be used to calculate the mean overlap 

print('CHECK INTERSECTIONS \ ')

#### True and false positives and negatives
# True positive: Algorithm detected object at location where there is also an object in ground truth
TP = len(intersections)
print("true positive  ", TP)

# False positive: Algorithm detected object while there is no object in ground truth
FP= len(dc_chim) - TP
print("false positive ", FP)

# False negative: Algorithm did not detect object at location where there is an object in ground truth
FN = len(gt_chim) - TP
print("false negative ", FN)

### Confusion matrix
# GT 
matches = np.array(intersections)
bag_seg_gt = matches[:, [7, 8]]

un_id = []
for x in bag_seg_gt:
    un_id.append([x[0] + '_' + str(x[1])])

print(un_id[0])

un_id = np.array(un_id)    
gt_seg = matches[:,6]
gt_matches = np.column_stack((un_id, gt_seg))


# DC               
chimney_uniqueid_dc = []
seg_chim = []

# import the chimney data 
with open(chimney_csv, "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    for lines in csv_reader:
        l = lines[0]
        chimney_uniqueid_dc.append(l[0:-4] + '_' + str(lines[1]))

chimney_uniqueid_dc = np.array(chimney_uniqueid_dc)
print(chimney_uniqueid_dc[5])


in_dc = []
for a in gt_matches[:, 0]:
    if np.isin(a, chimney_uniqueid_dc):
        in_dc.append('y')
    else:
        in_dc.append('n')

# confusion matrix 
in_dc = np.array(in_dc)
result = np.column_stack((gt_matches, in_dc))

gt_numpy = result[:,1]
dt_numpy = result[:,2]
print(dt_numpy)
print(gt_numpy)
groundTruth = gt_numpy.tolist()
detected = dt_numpy.tolist()

y_actu = pd.Series(groundTruth, name='Ground Truth')
y_pred = pd.Series(detected, name='Detected')

df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Ground Truth'], colnames=['Detected'], margins=True)
print(df_confusion)


## Overall accuaracy
conf1 = df_confusion['n']['n']
conf2 = df_confusion['y']['y']
conf_tot = df_confusion['All']['All']
print(conf1)
print(conf2)
print(conf_tot)
ov_ac = (conf1 + conf2) / conf_tot * 100

print('overall accuracy is ', ov_ac)

## Kappa coefficient
p0 = ov_ac / 100
pno = ((df_confusion['n']['n'] + df_confusion['n']['y'])/ conf_tot) * ((df_confusion['n']['n'] + df_confusion['y']['n'])/ conf_tot)
pyes = ((df_confusion['y']['n'] + df_confusion['y']['y'])/ conf_tot) * ((df_confusion['n']['y'] + df_confusion['y']['y'])/ conf_tot)
pe = pno + pyes
k = ((p0 - pe) / (1 - pe))
print('kappa coefficient is ', k)


#Cm = Cm = ConfusionMatrix(groundTruth,detected)
#Cm.print_stats()
#print(Cm)

### Write to shapefile
print('WRITE FILE / ')
with fiona.open('test.shp', 'w', 'ESRI Shapefile', schema) as c:
    for k in range(len(intersections)):
        try:
            c.write({
                'geometry': mapping(Polygon(intersections[k][0].points)),
                'properties': {'id_dc': prop_dc[intersections[k][4]][0] , 'id_gt': prop_gt[intersections[k][5]][0],'perc': intersections[k][3], 'seg': prop_dc[intersections[k][4]][1], 'chim': prop_gt[intersections[k][5]][1]},

                })
        except Exception as e:
            print(e)
           

print('WRITE FILE \ ')

perc_det = (len(intersections))/len(gt_chim)*100 # percentage of ground truth chimneys detected
mean_ove = per_int / len(intersections)
print("The overlap percentage = ", thresh)
print("Of the " + str(len(gt_chim)) + " objects in the ground truth, the algorithm has detected " + str(perc_det) + "% (= " + str(len(intersections)) + " objects)")
print("The mean overlap is " + str(mean_ove) + "%")

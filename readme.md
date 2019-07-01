Small collection of code snippets for chimney detection as part of the synthesis project for the master Geomatics at the TU Delft. 
Goal of this project is to detect chimneys in urban pointcloud scenes.

Following snippets can found:

<ul>
  <li>check_compliance.py: detect the buildings that fall under the compliance rules for chimneys</li>
  <li>cluster.pyx: cluster a pointcloud into segments</li>
  <li>detect.pyx: detect planar segments in a pointcloud</li>
  <li>detection_above_roofs.py: detect which points of a pointcloud are above a roof</li>
  <li>filter.py: filter outliers of a pointcloud after plane detection</li>
  <li>filterChimneys.py: filter chimney out of a segmented pointcloud</li>
  <li>findHeightofObj.py: get the height of the building from a json file (BAG)</li>
  <li>polygon_creator.py: create a polygon (shp) out of a pointcloud </li>
  <li>polygon_creator_chimneys.py: same as before but more suitable to chimneys</li>
  <li>polygon_creator_concave.py: same as before but as a concave hull</li>
  <li>quality.py: checks the quality of the chimney detection</li>
  <li>roofExtraction.py: Extract roofs out of the pointcloud</li>
</ul>

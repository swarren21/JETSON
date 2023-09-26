'''
Sample Command:-
python detect_aruco_images.py --image Images/test_image_1.png --type DICT_5X5_100
'''
import numpy as np
from utils import ARUCO_DICT, aruco_display
import argparse
import cv2
import sys


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image containing ArUCo tag")
ap.add_argument("-t", "--type", type=str, default="DICT_5X5_100", help="type of ArUCo tag to detect")
args = vars(ap.parse_args())



print("Loading image...")
image = cv2.imread(args["image"])
h,w,_ = image.shape
width=600
height = int(width*(h/w))
image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)


# verify that the supplied ArUCo tag exists and is supported by OpenCV
if ARUCO_DICT.get(args["type"], None) is None:
	print(f"ArUCo tag type '{args['type']}' is not supported")
	sys.exit(0)

# load the ArUCo dictionary, grab the ArUCo parameters, and detect
# the markers
print("Detecting '{}' tags....".format(args["type"]))
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
arucoParams = cv2.aruco.DetectorParameters() 

arucoParams.adaptiveThreshWinSizeMin = 23
arucoParams.adaptiveThreshWinSizeMax = 23
arucoParams.adaptiveThreshWinSizeStep = 10

arucoParams.adaptiveThreshConstant = 30

arucoParams.minMarkerPerimeterRate = 0.05 # Minimum Size of Marker
arucoParams.maxMarkerPerimeterRate = 4.0 # Maximum size of Marker

arucoParams.polygonalApproxAccuracyRate = 0.05

arucoParams.minCornerDistanceRate = 0.05

arucoParams.minMarkerDistanceRate = 0.05

arucoParams.minDistanceToBorder = 3

arucoParams.markerBorderBits = 1

arucoParams.minOtsuStdDev = 5.0

arucoParams.perspectiveRemovePixelPerCell = 4

arucoParams.perspectiveRemoveIgnoredMarginPerCell = 0.13

arucoParams.maxErroneousBitsInBorderRate = 0.35

arucoParams.errorCorrectionRate = 0.6

arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE

arucoParams.cornerRefinementWinSize = 5

arucoParams.cornerRefinementMaxIterations = 30

arucoParams.cornerRefinementMinAccuracy = 0.1

corners, ids, rejected = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

(detected_markers, center) = aruco_display(corners, ids, rejected, image)
print(center[0:])
print(width, height)
cv2.line(image, center, (300, 225), (0, 255, 0), 2)
cv2.imshow("Image", detected_markers)

# # Uncomment to save
# cv2.imwrite("output_sample.png",detected_markers)

cv2.waitKey(0)

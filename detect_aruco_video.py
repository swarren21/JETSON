'''
Sample Usage:-
python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100
'''


import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time
import smbus
import os
import subprocess

bus = smbus.SMBus(0)
address = 0x6a

I2C_Flag = False
OFFSET_HORIZONTAL = 0 
OFFSET_VERTICAL = 20
OFFSET_EXTEND = 9 
ROT_GAIN = 10

def gstreamer_pipeline(
    capture_width=640,
    capture_height=480,
    display_width=640,
    display_height=480,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


# Main Loop
if __name__ == '__main__':

# Get argument inputs from the user
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", default="/home/mule/JETSON/calibration_matrix.npy",  help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", default="/home/mule/JETSON/distortion_coefficients.npy", help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_5X5_100", help="Type of ArUCo tag to detect")
    args = vars(ap.parse_args())
    print(args["K_Matrix"])
    print(args["D_Coeff"])

# Verify that the selected tags are valid    
    if ARUCO_DICT.get(args["type"], None) is None:
        print("ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

# Define variables 
    aruco_dict_type = "DICT_5X5_100"
    calibration_matrix_path = "/home/mule/JETSON/calibration_matrix.npy"
    distortion_coefficients_path = "/home/mule/JETSON/distortion_coefficients.npy"
    
# Load the calibration and distortion coefficients matrices
    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)

# Main while loop that will never be broken
##### Start of Main Loop #####
    while True:
        video = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        time.sleep(2.0)

        while True:
            ret, frame = video.read()

            if not ret:
                break
###################################################        

            # cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
            parameters = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(dictionary, parameters)


            corners, ids, rejected_img_points = detector.detectMarkers(frame)

            if len(corners) > 0:
                for i in range(0, len(ids)):
                    rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners, 0.02, k, d)
                    # Estimate the attitude of each marker and return the values rvet and tvec --- different

                    cv2.aruco.drawDetectedMarkers(frame, corners)
                    # from camera coeficcients
                    (rvec-tvec).any() # get rid of that nasty numpy value array error
                    # Data variables: [X Rot, Y Rot, Z Rot, X, Y, Z]
###################################################
            cv2.imshow('Estimated Pose', frame)

            key = cv2.waitKey(1) & 0xFF
            if I2C_Flag == False:
                I2C_Flag = True
                break
            if key == ord('q'):
                break
##### End of Main Loop #####

    video.release()
    cv2.destroyAllWindows()

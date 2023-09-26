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
OFFSET_VERTICAL = 0
OFFSET_EXTEND = 0 
ROT_GAIN = 10
HORIZONTAL_RESOLUTION = 640
VERTICAL_RESOLUTION = 480

def gstreamer_pipeline(
    capture_width=HORIZONTAL_RESOLUTION,
    capture_height=VERTICAL_RESOLUTION,
    display_width=HORIZONTAL_RESOLUTION,
    display_height=VERTICAL_RESOLUTION,
    framerate=60,
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


def writeData(data):
    try:
        bus.write_i2c_block_data(address, data[0], [data[1],data[2],data[4],data[3],data[5]])
        # print(data[0:])
        return True
    except:
        print("Send Data Failed")
        return False



def pose_esitmation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera
    return:-
    frame - The frame with the axis drawn on it
    '''
    # cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    parameters = cv2.aruco.DetectorParameters()
    parameters.adaptiveThreshWinSizeMin = 23
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 10

    parameters.adaptiveThreshConstant = 25

    detector = cv2.aruco.ArucoDetector(dictionary, parameters)


    corners, ids, rejected_img_points = detector.detectMarkers(frame)

    if len(corners) > 0:
        for i in range(0, len(ids)):
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners, 0.02, matrix_coefficients, distortion_coefficients)
            # Estimate the attitude of each marker and return the values rvet and tvec --- different

            cv2.aruco.drawDetectedMarkers(frame, corners)
            # from camera coeficcients
            (rvec-tvec).any() # get rid of that nasty numpy value array error
            ROT = (rvec)
            TRAN = [(tvec[0][0][0]*100+0.1216*tvec[0][0][2]*100-0.2062, tvec[0][0][1]*1000-0.3674*tvec[0][0][2]*100+4.8585, tvec[0][0][2]*100]
            print(TRAN[0:])

            # Data variables: [X Rot, Y Rot, Z Rot, X, Y, Z]
            data = [int(ROT[0][0][0]),int(ROT[0][0][1]),int(ROT[0][0][2]),int(TRAN[0]),int(TRAN[1]),int(TRAN[2])]
            I2C_Flag = writeData(data)
            # print(data[3:])
    else:
        data = [0,0,0,0,0,0]
        I2C_Flag = writeData(data)
 

    return True

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
    aruco_dict_type = ARUCO_DICT[args["type"]]
    calibration_matrix_path = args["K_Matrix"]
    distortion_coefficients_path = args["D_Coeff"]
    
# Load the calibration and distortion coefficients matrices
    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)

# Main while loop that will never be broken
##### Start of Main Loop #####
    while True:

# Wait for auto hitch function to be enabled on the Mule
##### Start of I2C Loop #####
        while True:
            p = subprocess.Popen(['i2cdetect','-y','-r','0'],stdout=subprocess.PIPE,)
    
            line = ""
            for i in range(0,9):
                line = line + str(p.stdout.readline()) + '\r'

# If an I2C device with address 0x6a becomes available, continue to aruco detection
            if "6a" in line:
                print("6a")
                break
            else:
                print("No I2C Devices Found")
                time.sleep(3)
##### End of I2C Loop #####

        video = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        time.sleep(2.0)

        while True:
            ret, frame = video.read()

            if not ret:
                break
        
            I2C_Flag = pose_esitmation(frame, aruco_dict_type, k, d)

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

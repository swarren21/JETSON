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
    capture_width=1280,
    capture_height=720,
    display_width=720,
    display_height=1280,
    framerate=120,
    flip_method=3,
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
        print(data[0:])
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

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)


    corners, ids, rejected_img_points = detector.detectMarkers(gray)

    if len(corners) > 0:
        for i in range(0, len(ids)):
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners, 0.02, matrix_coefficients, distortion_coefficients)
            # Estimate the attitude of each marker and return the values rvet and tvec --- different

            cv2.aruco.drawDetectedMarkers(frame, corners)
            # from camera coeficcients
            (rvec-tvec).any() # get rid of that nasty numpy value array error
            ROT = (rvec)
            TRAN = (tvec * 39.3701 * 10)
            # Data variables: [X Rot, Y Rot, Z Rot, X, Y, Z]
            print(ROT[0][0][0]*ROT_GAIN)
            print(ROT[0][0][1]*ROT_GAIN)
            print(ROT[0][0][2]*ROT_GAIN)
            data = [int(ROT[0][0][0] * ROT_GAIN),int(ROT[0][0][1] * ROT_GAIN),int(ROT[0][0][2] * ROT_GAIN * 180/31.41592),int(TRAN[0][0][0]-TRAN[0][0][2]/5+OFFSET_VERTICAL),int(TRAN[0][0][1]-TRAN[0][0][2]+OFFSET_HORIZONTAL) * 3,int(TRAN[0][0][2])-OFFSET_EXTEND]

            I2C_Flag = writeData(data)
    else:
        data = [0,0,0,0,0,0]
        I2C_Flag = writeData(data)
 

    return True

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", default="/home/mule/JETSON/calibration_matrix.npy",  help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", default="/home/mule/JETSON/distortion_coefficients.npy", help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_5X5_100", help="Type of ArUCo tag to detect")
    args = vars(ap.parse_args())
    print(args["K_Matrix"])
    print(args["D_Coeff"])

    
    if ARUCO_DICT.get(args["type"], None) is None:
        print("ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]
    calibration_matrix_path = args["K_Matrix"]
    distortion_coefficients_path = args["D_Coeff"]
    
    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)
    while True:
        while True:
            p = subprocess.Popen(['i2cdetect','-y','-r','0'],stdout=subprocess.PIPE,)
    
            line = ""
            for i in range(0,9):
                line = line + str(p.stdout.readline()) + '\r'
            if "6a" in line:
                print("6a")
                break
            else:
                print("No I2C Devices Found")
                time.sleep(3)

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

    video.release()
    cv2.destroyAllWindows()

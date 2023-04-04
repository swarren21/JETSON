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

bus = smbus.SMBus(0)
address = 0x6a

def writeData(data):
    try:
        bus.write_i2c_block_data(address, data[0], [data[1],data[2],data[3],data[4],data[5]])
    except:
        return -1



def pose_esitmation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    '''

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()


    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict,parameters=parameters)

    if ids is not None:

        rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners, 0.02, matrix_coefficients, distortion_coefficients)
        # Estimate the attitude of each marker and return the values rvet and tvec --- different
        
        ROT = (rvec * 100)
        TRAN = (tvec * 39.3701 * 10)
        data = [int(ROT[0][0][0]),int(ROT[0][0][1]),int(ROT[0][0][2]),int(TRAN[0][0][0]),int(TRAN[0][0][1] + 1),int(TRAN[0][0][2]/10)]
#        for n in range( len(data)):
#            if data[n] > 127:
#                data[n] = 127
#            if data[n] < -128:
#                data[n] = -128
        print(data)

        # from camera coeficcients
        (rvec-tvec).any() # get rid of that nasty numpy value array error

#        aruco.drawAxis(frame, mtx, dist, rvec, tvec, 0.1) #Draw axis
#        aruco.drawDetectedMarkers(frame, corners) #Draw a square around the mark
        writeData(data)

        for i in range(rvec.shape[0]):
            frame = cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec[i, :, :], tvec[i, :, :],0.01)
            cv2.aruco.drawDetectedMarkers(frame, corners)
        ###### DRAW ID #####
        # cv2.putText(frame, "Id: " + str(ids), (0,64), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)


        
        ##### DRAW "NO IDS" #####
#        cv2.putText(frame, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)

#        # If markers are detected
#    if len(corners) > 0:
#        for i in range(0, len(ids)):
#            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
#            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
#                                                                       distortion_coefficients)
#            # Draw a square around the markers
#            cv2.aruco.drawDetectedMarkers(frame, corners) 
#
#            # Draw Axis
#            try:
#                frame = cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec[i, :, :], tvec[i, :, :],0.02)
#            except:
#                continue  

    return frame

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", default="calibration_matrix.npy",  help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", default="distortion_coefficients.npy", help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
    args = vars(ap.parse_args())
    print(args["K_Matrix"])
    print(args["D_Coeff"])

    
    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]
    calibration_matrix_path = args["K_Matrix"]
    distortion_coefficients_path = args["D_Coeff"]
    
    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)

    video = cv2.VideoCapture(0)
    time.sleep(2.0)

    while True:
        ret, frame = video.read()

        if not ret:
            break
        
        output = pose_esitmation(frame, aruco_dict_type, k, d)

        cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

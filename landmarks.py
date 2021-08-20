import sys


from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX 
from TDDFA_ONNX import TDDFA_ONNX 
import os
from pathlib import Path
from glob import glob
import numpy as np
import json
import cv2
import pprint
from tqdm import tqdm
#from .utils import load_frame
import yaml





cfg = yaml.load(open("/home/ubuntu/AVSpeech-RFD/face_detection/configs/mb1_120x120.yml"), Loader=yaml.SafeLoader)


face_boxes = FaceBoxes_ONNX()
tddfa = TDDFA_ONNX(**cfg)


def load_frame(frame_path, resize_factor=1, rotate=False, crop = [0, -1, 0, -1]):
    #print("reading original frames .....")
    
    frame = cv2.imread(frame_path, 1)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if resize_factor > 1:
        frame = cv2.resize(frame, (frame.shape[1]//resize_factor, frame.shape[0]//resize_factor))
            
    if rotate:
        frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
            
    y1, y2, x1, x2 = crop
    if x2 == -1: x2 = frame.shape[1]
    if y2 == -1: y2 = frame.shape[0]
        
    frame = frame[y1:y2, x1:x2]
        
    return frame

def detect(frames, landmarks_save_string):
    landmarks_data = {}
    if os.path.exists(landmarks_save_string):
        
        print("(landmarks.json exists) Skipping .....") 
        return 
    else:
        for i, obj in enumerate(frames):
            final_landmarks = []
            landmarks = []
            import time
            start_time = time.time()
            print('calculating FL {}/{}'.format(i, len(frames)))
            frame_bgr = load_frame(obj)
            boxes = face_boxes(frame_bgr)
            param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
            vers = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)
            
            for ver in vers:
                param_lst, roi_box_lst = tddfa(frame_bgr, [ver], crop_policy='landmark')
                pred = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)
                landmarks.append(pred)
        
            for face in landmarks:
                for pts in face:
                    n = pts.shape[1]
                    co_ordinates = []
                    for i in range(n):
                        co_ordinates.append([float(pts[0, i]), float(pts[1, i])])
                final_landmarks.append(co_ordinates)
                
                print("predection ...{}".format(os.path.basename(obj)))
                print(time.time() - start_time)
            #pprint.pprint(final_landmarks)
        
            landmarks_data[os.path.basename(obj)] = final_landmarks
        
        with open(landmarks_save_string, 'w') as outfile:
            json.dump(landmarks_data, outfile)
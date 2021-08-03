import os
import cv2
from pathlib import Path
from glob import glob
import numpy as np
import json
from tqdm import tqdm
import imageio
import copy
#check wether GPU is available or not 


#face Alignment library initialization
import face_alignment
import time

fa = [face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, 
									device='cuda:{}'.format(id)) for id in range(1)]





video_fp = "/home/ubuntu/3DDFA_V2/exp/moneyheist_6_3.mp4"
save_fp = "/home/ubuntu/3DDFA_V2/examples/results/videos//moneyheist_6_3_face_alignment.mp4"


reader = imageio.get_reader(video_fp)

fps = reader.get_meta_data()['fps']

writer = imageio.get_writer(save_fp, fps=fps)

print("total_frames:{}".format(len(reader)))

for i, frame in enumerate(reader):
    print('calculating FL pred for frame IDX: {}'.format(i))
    
    start_time = time.time()
    img = copy.deepcopy(frame)
    pred = fa[0].get_landmarks(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    print(time.time() - start_time)
    if pred == None:
        pred = []
    else:
        pred = [p.tolist() for p in pred]
        
    for face in pred:
        for co_ordinates in face:
            cv2.circle(img, (int(co_ordinates[0]), int(co_ordinates[1])), 1, (255, 0, 0), -1)
            
    writer.append_data(img[..., ::1])  

writer.close()
print(f'Dump to {save_fp}')



                

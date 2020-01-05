import sys
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("done importing")
sys.stdout.flush()

df = pd.read_csv('/green-projects/project-sonyc_redhook/workspace/share/redhook-analysis/output/max_dataframe.csv')
print("read csv")
sys.stdout.flush()

def show_frame(video_file, frame_number):
    vid_obj = cv2.VideoCapture(video_file)
    vid_obj.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    res, frame = vid_obj.read()
    if res:
        cropped_frame = frame[200:850, 1100:1750]
        return cropped_frame
    else:
        return False
    
print("defined function")
sys.stdout.flush()

img_arr = [show_frame('/green-projects/project-sonyc_redhook/workspace/share/original_video/' \
               + str(int(df.iloc[x]['start_timestamp'])) + '.ts', df.iloc[x]['frame']) for x in df.index]

print("calculated img_arr")
sys.stdout.flush()

df['image'] = img_arr

df.to_hdf('max_dataframe_img.hdf5', key='df')

print("saved to hdf5")
sys.stdout.flush()
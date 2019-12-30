import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

df = pd.read_csv('max_dataframe.csv')

def show_frame(video_file, frame_number):
    vid_obj = cv2.VideoCapture(video_file)
    vid_obj.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    res, frame = vid_obj.read()
    if res:
        cropped_frame = frame[200:850, 1100:1750]
        return cropped_frame
    else:
        return False

img_arr = [show_frame('/green-projects/project-sonyc_redhook/workspace/share/original_video/' \
               + str(int(df.iloc[x]['start_timestamp'])) + '.ts', df.iloc[x]['frame']) for x in df.index]

df['image'] = img_arr

df.to_hdf('max_dataframe_img.hdf5')
import sys
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h5py

print("done importing")
sys.stdout.flush()

df = pd.read_csv('/green-projects/project-sonyc_redhook/workspace/share/redhook-analysis/output/max_dataframe_cut.csv')
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

with h5py.File('max_img.hdf5', 'w') as h5:
    d = h5.create_dataset('max_img',
                          (len(df),),
                          dtype=[('start_timestamp', 'f8'),
                                 ('frame', 'i8'),
                                 ('actual_timestamp', 'f8'),
                                 ('area', 'f8', ),
                                 ('probability', 'f8'),
                                 ('img', 'i8', (650, 650, 3))
                                ],
                            chunks=True,
                            maxshape=(len(df) * 650 * 650 * 3 * 6,))
#     print(d[0])
    for idx in df.index:
        d[idx] = (df.iloc[idx]['start_timestamp'], df.iloc[idx]['frame'], df.iloc[idx]['actual_timestamp'], \
                  df.iloc[idx]['area'], df.iloc[idx]['probability'], img_arr[idx])

print("saved to hdf5")
sys.stdout.flush()
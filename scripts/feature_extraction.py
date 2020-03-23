import openl3
import soundfile as sf
import pickle
import os
import sys

with open('../output/annotation_list.pickle', "rb") as f:
        annotation_list = pickle.load(f)
print('done loading annotation list')
sys.stdout.flush()

embedding_list = []
print('done making empty embedding list')
sys.stdout.flush()

total = 0
for folder in \
    os.listdir('/green-projects/project-sonyc_redhook/workspace/share/truck_audio/redhook_truck_audio/10s'):
        for file in os.listdir\
        ('/green-projects/project-sonyc_redhook/workspace/share/truck_audio/redhook_truck_audio/10s/' + folder):
            total += 1
            
count = 0
for folder in \
    os.listdir('/green-projects/project-sonyc_redhook/workspace/share/truck_audio/redhook_truck_audio/10s'):
        for file in os.listdir\
        ('/green-projects/project-sonyc_redhook/workspace/share/truck_audio/redhook_truck_audio/10s/' + folder):
            audio_timestamp = int(file.split(".")[0])
            audio, sr = sf.read('/green-projects/project-sonyc_redhook/workspace/share/truck_audio/' +
                                'redhook_truck_audio/10s/' + folder + '/' + file)
            emb, ts = openl3.get_audio_embedding(audio, sr, content_type='env', embedding_size=512)
            embedding = [emb, ts]
            embedding_list.append((audio_timestamp, embedding))
            count += 1
            print('done with ' + str(count) + ' out of ' + str(total))
            sys.stdout.flush()

print('done with all embeddings')
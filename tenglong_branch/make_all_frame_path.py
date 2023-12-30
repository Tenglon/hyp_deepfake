import os
from pathlib import Path

# root_dir = '/home/longteng/datasets/ff++23'
root_dir = Path('/ssd/ff++23/all_frames')
dest_dir = Path('/ssd/ff++23/tlong_all_frames/')
dest_dir.mkdir(parents=True, exist_ok=True)

dest_dir_neuraltextures = dest_dir / 'neuraltextures'
dest_dir_deepfakes = dest_dir / 'deepfakes'
dest_dir_face2face = dest_dir / 'face2face'
dest_dir_faceswap = dest_dir / 'faceswap'
dest_dir_original = dest_dir / 'original'

# clean the destination directory
for dir in [dest_dir_neuraltextures, dest_dir_deepfakes, dest_dir_face2face, dest_dir_faceswap, dest_dir_original]:
    if dir.exists():
        os.system('rm -rf {}'.format(dir))
    dir.mkdir(parents=True, exist_ok=True)

for split in ['train', 'val', 'test']:
    for video_name in os.listdir(root_dir / split):

        if 'face2face' in video_name:
            dest_dir = dest_dir_face2face / video_name
        elif 'faceswap' in video_name:
            dest_dir = dest_dir_faceswap / video_name
        elif 'neuraltextures' in video_name:
            dest_dir = dest_dir_neuraltextures / video_name
        elif 'deepfakes' in video_name:
            dest_dir = dest_dir_deepfakes / video_name
        else:
            dest_split_dir = dest_dir_original / split
            dest_split_dir.mkdir(parents=True, exist_ok=True)
            dest_dir = dest_split_dir / video_name

        source_dir = root_dir / split / video_name
        # create a soft link to the original video
        os.symlink(source_dir, dest_dir)


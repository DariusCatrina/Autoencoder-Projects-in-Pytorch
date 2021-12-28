# The dataset used for this project is the FFHQ DataSet from NVidia LABS
# You can download it via Github link: !git clone https://github.com/NVlabs/ffhq-dataset
# !cd ffhq-dataset && python download_ffhq.py -t 
# Or from google colab

import os
import shutil
from tqdm import tqdm

def download_files(limit):
    source_dir = 'drive/MyDrive/thumbnails128x128/'
    target_dir = 'FFHQ-DATA'
        
    folder_names = os.listdir(source_dir)
    folder_names.remove('LICENSE.txt')

    i = 0

    for folder in tqdm(folder_names):

        files = os.listdir(source_dir + folder)
        for file_ in files:
          if limit and i == limit:
            break
          else:
            shutil.copy(os.path.join(source_dir + '/' + folder, file_), target_dir)
            i+=1
        if limit and i == limit:
            break

def index_files(source_dir):
    idx = 0
    photos = sorted(os.listdir(source_dir))

    for file_ in photos:
      os.rename(source_dir+'/'+file_, source_dir + '/' + str(idx) + '.png')
      #print(file_, source_dir + '/' + str(idx) + '.png')
      idx+=1

    # print(f'FILES IN {source_dir}: {len(os.listdir(source_dir))}')



def train_test_split():
    source_dir = 'FFHQ-DATA'
    samples = sorted(os.listdir(source_dir))


    target_train_dir = 'FFHQ-TRAIN/'
    target_test_dir = 'FFHQ-TEST/'

    for idx in range(int(len(samples) * 9.5/10)):
        shutil.copy(os.path.join(source_dir, samples[idx]), target_train_dir)
    
    for idx in range(int(len(samples) * 9.5/10), len(samples)):
        shutil.copy(os.path.join(source_dir, samples[idx]), target_test_dir)

target_train_dir = 'FFHQ-TRAIN'
target_test_dir = 'FFHQ-TEST'
original_dir = 'FFHQ-DATA'


download_files(limit=45100)
train_test_split()
index_files(target_train_dir)
index_files(target_test_dir)

print('\n\n')
print(f'FILES IN {target_train_dir}: {len(os.listdir(target_train_dir))}')
print(f'FILES IN {target_test_dir}: {len(os.listdir(target_test_dir))}')

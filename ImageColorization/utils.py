import os
import shutil

def move_files():
    source_dir = 'ffhq-dataset/thumbnails128x128/'
    target_dir = 'FFHQ-ORIGINAL'
        
    folder_names = os.listdir(source_dir)
        
    for folder in folder_names:
        files = os.listdir(source_dir + folder)
        for file in files:
            shutil.move(os.path.join(source_dir + '/' + folder, file), target_dir)

def index_files(source_dir):
    idx = 0
    photos = sorted(os.listdir(source_dir))
    for file_ in photos:
      os.rename(source_dir+'/'+file_, source_dir + '/' + str(idx) + '.png')
      idx+=1




def train_test_split():
    source_dir = 'FFHQ-DATA'
    samples = os.listdir(source_dir)


    target_train_dir = 'FFHQ-TRAIN/'
    target_test_dir = 'FFHQ-TEST/'

    for idx in range(int(len(samples) * 9/10)):
        shutil.move(os.path.join(source_dir, samples[idx]), target_train_dir)
    
    for idx in range(int(len(samples) * 9/10), len(samples)):
        shutil.move(os.path.join(source_dir, samples[idx]), target_test_dir)

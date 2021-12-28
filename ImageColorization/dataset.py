import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np
from skimage.color import rgb2lab, rgb2gray, lab2rgb

transforms = T.Compose([
      T.ToTensor(),
      T.ConvertImageDtype(torch.float)
    ])

to_pil_transform = T.Compose([
    T.ToPILImage()
])


def transform(rgb_img, transforms=None):
  lab_image = (rgb2lab(np.array(rgb_img)) + 128) / 255 # normalization

  ab_img = lab_image[:,:, 1:3] # A,B channels  --- (128,128,2)
  gray_img = np.expand_dims(rgb2gray(np.array(rgb_img)), 2) #(L)uminosity channel  --- (128,128,1)

  if transforms:
    return transforms(gray_img), transforms(rgb_img), transforms(ab_img)#input, final_target, ab_target
  return gray_img, rgb_img, ab_img 

def reconstruct_img(ab_img, gray_img):

  recon_img = torch.cat((gray_img, ab_img), 0).detach().cpu().numpy()

  recon_img[0:1,:,:] = recon_img[0:1,:,:] * 100
  recon_img[1:3:,:,] = recon_img[1:3,:,:] * 255 - 128
  recon_img = np.swapaxes(recon_img, 0, -1)

  reconstruct_rgb = torch.swapaxes(torch.from_numpy(lab2rgb(recon_img)), 0, -1)

  return to_pil_transform(reconstruct_rgb)


class FFHQ_Dataset(Dataset):
    def __init__(self, IMG_DIR='FFHQ-DATA', transform=None) -> None:
        super().__init__()
        self.IMG_DIR = IMG_DIR
        self.transform = transform

    def __len__(self) -> int:
        import os
        return len(os.listdir(self.IMG_DIR))

    def __getitem__(self, idx) -> Image:
        import os
        found = 0
        for photo in os.listdir(self.IMG_DIR):
            if photo == (str(idx) + '.png'):
                img = Image.open(self.IMG_DIR + '/' + photo)
                found = 1
                break
        if not found:
            raise ValueError(f"The image with the index {idx} not found in the dataset")
              
        return self.transform(img, transforms=transforms)
    
    def __show__(self, idx, transform):
      if transform:
        img = transform(self.__getitem__(idx))
      else:
        img = self.__getitem__(idx)
      img

train_dataset = FFHQ_Dataset(IMG_DIR='FFHQ-TRAIN', transform=transform)
test_dataset = FFHQ_Dataset(IMG_DIR='FFHQ-TEST', transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

transforms = T.Compose([
    T.ToTensor()
    ])


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

        for photo in os.listdir(self.IMG_DIR):
            if photo == (str(idx) + '.png'):
                image = Image.open(self.IMG_DIR + '/' + photo)
                break
        if self.transform:
            return self.transform(image)

        return image
    
    def show(self, idx):
        image = self.__getitem__(idx)
        image.show()

dataset = FFHQ_Dataset()





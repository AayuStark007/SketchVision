import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import cv2
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train', name='pic2sketch'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        if name == 'pic2sketch':
            a_path = os.path.join(root, '256x256', 'photo/tx_000000000000/*')
            b_path = os.path.join(root, '256x256', 'sketch/tx_000000000000/*')
            self.files_A = sorted(glob.glob(a_path + '/*.*'))
            self.files_B = sorted(glob.glob(b_path + '/*.*'))
        else:
            self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
            self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class VideoDataset(Dataset):
    def __init__(self, capture, transforms_=None, stream=False):
        self.cap = capture
        self.transform = transforms.Compose(transforms_)
    
    def __getitem__(self, index):
        data = self.transform(cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB))

        return data

    def __len__(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))        

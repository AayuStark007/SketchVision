import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from models.base_model import Generator
from datasets import VideoDataset

import cv2
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
#parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
#parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='output/base_model_25_h2z/netG_A2B.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='output/base_model_25_h2z/netG_B2A.pth', help='B2A generator checkpoint file')
opt = parser.parse_args()
print(opt)

device = torch.device("cuda")

cap = cv2.VideoCapture(0)

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc).to(device)
netG_B2A = Generator(opt.output_nc, opt.input_nc).to(device)

# Load state dicts
netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
netG_B2A.load_state_dict(torch.load(opt.generator_B2A))

# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size).to(device)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size).to(device)

transforms_ = [ transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
transform = transforms.Compose(transforms_)
###################################
i = 0
###### Testing######

while cap.isOpened():
    img = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
    
    if img is None:
        break
    
    height, width, _ = img.shape

    w_crop_start = (width // 2) - (height // 2)
    w_crop_end   = (width // 2) + (height // 2)

    img = img[:, w_crop_start:w_crop_end, :]
    img = cv2.resize(img, (opt.size, opt.size))

    data = Image.fromarray(img)
    data_tensor = transform(data)
    
    real_A = input_A.copy_(data_tensor)

    fake_B = 0.5*(netG_A2B(real_A).detach() + 1.0)

    save_image(fake_B, 'output/D/%04d.png' % (i+1))
    
    data_out = cv2.imread('output/D/%04d.png' % (i+1))
    
    if i == 3:
        i = 0
    else:
        i += 1

    cv2.imshow("Output", data_out)
    
    if cv2.waitKey(24) & 0xFF == ord('q'):
        break

    #real_A = input_A.copy_
cap.release()
cv2.destroyAllWindows()
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
import config
import cv2
import numpy as np

from transform_utils import convert_to_binary_inv,convert_to_binary,adapt_size
from torchvision import datasets, transforms

ToTensor=transforms.ToTensor()

def append_dir_bigset(package_x,package_y,basepath,item_y):
    path = os.listdir(basepath)
    for item in path:
        item_x=Image.open(basepath + item)
        item_x = np.array(item_x)
        # if True:
        #     print(item_x.shape)
        #
        #     plt.matshow(item_x)
        #     plt.show()
        package_x.append(torch.tensor(item_x,dtype=torch.uint8))
        package_y.append(torch.tensor(item_y,dtype=torch.uint8))
        # return

if __name__ =="__main__":
    
    package_x = list()
    package_y = list()
    base_dir = 'E:/bigset/Train/'
    if 1:
        for i in range(0,10):
            append_dir_bigset(package_x,package_y,base_dir+str(i)+"/",config.CLASS.index(str(i)))
    base_dir = 'E:/bigset/Validation/'
    if 1:
        for i in range(0,10):
            append_dir_bigset(package_x,package_y,base_dir+str(i)+"/",config.CLASS.index(str(i)))
    print(len(package_x),len(package_y))
    torch.save([package_x,package_y],'package_bigset.pth')
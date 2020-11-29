from find_box import opencv_find_box
from transform_utils import convert_to_binary_inv,adapt_size
import os 
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
base='D:\\dataset_math\\html_gen\\'
names=os.listdir(base)


names=['div2.png']
for name in names:
    if(name[-4:]!='.png'):
        continue
    test_image = cv2.imread(base +name , 0)
    equations = opencv_find_box(test_image)
    cnt_write = 0
    for equation in equations:
        print(name)
        print(len(equation))
        for rect in equation:
            if(name[0:3]=='div'):
                print(rect)
            cell = test_image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
            cell = convert_to_binary_inv(cell)
            # plt.matshow(cell)
            # plt.show()
            # cell=torch.tensor(cell).float().cuda()
            
            # cell-=cell.min()
            # cell/=cell.max()
            # cell*=255

            # print(cell.shape)
            # print(cell.int())
            cell=adapt_size(cell)
            cell=cell.cpu().int().numpy().astype(np.uint8)[0]*255
            # plt.matshow(cell)
            # plt.show()

            subdir=name.split('.')[0]
            if(not os.path.exists(base+subdir)):
                os.makedirs(base+subdir) 
            
            cv2.imwrite(base+subdir+'/'+str(cnt_write)+'p.png',cell)
            cnt_write+=1
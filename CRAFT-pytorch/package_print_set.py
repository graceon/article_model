import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
import config
import cv2
import numpy as np

from torchvision import datasets, transforms

ToTensor = transforms.ToTensor()

base_dir = 'E:/myops/'
res_dir = 'E:/myops/res'
char_hw = 32


def adapt_size(cell):
    origin_w , origin_h  = cell.shape
    if origin_h > origin_w:
        h2 = char_hw
        w2 = int((char_hw / origin_h) * origin_w)

        cell = cv2.resize(cell, (h2, w2), interpolation=cv2.INTER_CUBIC)

        pad = char_hw - w2
        if pad == 0:
            pass
        elif pad % 2 == 1:
            cell = cv2.copyMakeBorder(cell, pad // 2, pad // 2 + 1, 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            cell = cv2.copyMakeBorder(cell, pad // 2, pad // 2, 0, 0, cv2.BORDER_CONSTANT, value=0)
    else:
        w2 = char_hw
        h2 = int((char_hw / origin_w) * origin_h)


        cell = cv2.resize(cell, (h2, w2), interpolation=cv2.INTER_CUBIC)



        pad = char_hw - h2


        if pad == 0:
            pass
        elif pad % 2 == 1:
            cell = cv2.copyMakeBorder(cell, 0, 0, pad // 2, pad // 2 + 1, cv2.BORDER_CONSTANT, value=0)
        else:
            cell = cv2.copyMakeBorder(cell, 0, 0, pad // 2, pad // 2, cv2.BORDER_CONSTANT, value=0)
    return cell


def to_center(item_x):
    contour = []

    item_x = adapt_size(item_x)
    item_x = item_x - item_x.min()
    item_x = item_x / item_x.max()
    item_x = item_x * 255
    for hw in range(char_hw * char_hw):
        if item_x[hw % char_hw][hw // char_hw] > 50:
            contour.append([hw % char_hw, hw // char_hw])
    contour = np.array(contour, dtype=np.int8)
    item_x = item_x[contour[:, 0].min():contour[:, 0].max() + 1, contour[:, 1].min():contour[:, 1].max() + 1]

    # item_x = torch.tensor(item_x, dtype=torch.uint8)

    item_x = adapt_size(item_x)

    item_x = item_x - item_x.min()
    item_x = item_x / item_x.max()
    item_x = item_x * 255


    return item_x


def append_dir_bigset(package_x, package_y, basepath, item_y):
    path = os.listdir(basepath)
    for item in path:
        item_x = Image.open(basepath + item)
        item_x = np.array(item_x)
        if(len(item_x.shape)>=3):
            item_x = cv2.cvtColor(item_x, cv2.COLOR_BGR2GRAY)

        item_x = to_center(item_x)


        print(item_x.shape)
        package_x.append(torch.tensor(item_x, dtype=torch.uint8))
        package_y.append(torch.tensor(item_y, dtype=torch.uint8))
        # cv2.imwrite(res_dir + str(len(package_x)) + '.png', item_x)
        # return


if __name__ == "__main__":

    package_x = list()
    package_y = list()

    for i in config.CLASS:
        append_dir_bigset(package_x, package_y, base_dir + str(i) + "/", config.CLASS.index(str(i)))

    print(len(package_x), len(package_y))
    torch.save([package_x, package_y], 'package_print_set.pth')

    # 等号放到一个特殊的数据集中
    package_x = list()
    package_y = list()

    for i in ['equal']:
        append_dir_bigset(package_x, package_y, base_dir + str(i) + "/", config.CLASS.index(str(i)))

    print(len(package_x), len(package_y))
    torch.save([package_x, package_y], 'package_print_set_equal.pth')
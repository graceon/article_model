import os
import re

import torch

import cv2


def index_max(set_dir):
    arr = os.listdir(set_dir)
    max = 0
    for i in arr:
        str_index = re.findall(r"\d+", i)
        if len(str_index) < 1:
            continue
        now = int(str_index[0])
        if (now > max):
            max = now
    return max


def str_to_int(i):
    str_index = re.findall(r"\d+", i)
    return int(str_index[0])


write_cnt = index_max('res_debug') + 1
print("dir_utils.write_cnt=", write_cnt)


def debug_write(img_numpy, name_str):
    global write_cnt
    cv2.imwrite('./res_debug/' + "[%03d]" % write_cnt + name_str + '.png', img_numpy)
    write_cnt += 1

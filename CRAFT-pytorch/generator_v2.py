import matplotlib.pyplot as plt

import numpy as np

from torchvision.transforms import Resize, Pad, ToPILImage, ToTensor
import config
import random
import cv2
import os
import imutils

import custom_dataset

char_hw = 32


def get_random_char(set):
    random_index = random.randint(0, len(set) - 1)
    img_char = set[random_index][0][0]
    y = set[random_index][1]

    contour = []
    for hw in range(char_hw * char_hw):
        if img_char[hw % char_hw][hw // char_hw] > 50:
            contour.append([hw % char_hw, hw // char_hw])
    if len(contour)<4:
        return get_random_char(set)
    contour = np.array(contour, dtype=np.int8)
    # print(contour[:,0].min(),contour[:,0].max(),contour[:,1].min(),contour[:,1].max())
    img_char = img_char[contour[:, 0].min():contour[:, 0].max() + 1, contour[:, 1].min():contour[:, 1].max() + 1]

    # img_char=torch.tensor(img_char,dtype=torch.uint8)
    return img_char, y


big_set = custom_dataset.custom_dataset("package_bigset.pth", False)
print_set = custom_dataset.custom_dataset("package_print_set.pth", False)
# plt.matshow(get_random_char(train_set))
# plt.show()


generate_save_path = './generator'

generate_labels = []

max_width = 160
max_width_segment = 41
generate_count = 0

while generate_count < 20000:

    generate_count += 1
    expression_rect = 0
    expression_rect_init = True

    start = 0
    w = 0
    h = 0
    json_record_line = []

    rotate_angle = random.randint(-10, 10)
    # 主要生成在上还是在下
    top_or_bottom = random.randint(0, 1)
    # padding 是否偏离中间
    padding_percent = np.random.randn()


    while start <= max_width:

        threshold = max_width * random.uniform(0.5, 0.9)

        if start < threshold:
            img_char, y = get_random_char(print_set)

            # print(img_char.shape)
            if img_char.shape[1] < 4:
                resize_scale_x = 1
            else:
                resize_scale_x = random.uniform(0.3, 0.5)
            if img_char.shape[0] < 4:
                resize_scale_y = 1
            else:
                resize_scale_y = random.uniform(0.3, 0.5)
            # print(img_char)
            img_char = img_char.numpy()

            img_char = imutils.rotate_bound(img_char, rotate_angle)

            # img_char = cv2.resize(img_char, (0, 0), fx=min(resize_scale_x, 30 / img_char.shape[1]),
            #                       fy=min(resize_scale_y, 30 / img_char.shape[0]),
            #                       interpolation=cv2.INTER_CUBIC)
            img_char = cv2.resize(img_char, (0, 0), fx=min(resize_scale_x, 30 / img_char.shape[1]),
                                  fy=min(resize_scale_y, 30 / img_char.shape[0]),
                                  interpolation=cv2.INTER_CUBIC)
        else:
            img_char, y = get_random_char(big_set)
            resize_scale = random.uniform(0.5, 1)
            img_char = cv2.resize(img_char.numpy(), (0, 0), fx=resize_scale, fy=resize_scale,
                                  interpolation=cv2.INTER_CUBIC)
        # img_char = ToPILImage()(img_char*255)
        # img_char = Resize((random.randint(10, 28), random.randint(10, 28)))(img_char)
        # img_char = (ToTensor()(img_char)).numpy().astype(np.uint8)
        h = img_char.shape[0]
        w = img_char.shape[1]

        padding = 32 - h
        # if padding % 2 == 1:
        #     padding_top = padding // 2 + 1
        #     padding_bottom = padding // 2
        # else:
        #     padding_top = padding // 2
        #     padding_bottom = padding // 2

        # 正态分布： 上下padding应该是1/2左右

        padding_top = round(padding_percent* padding)
        # print(padding,padding_top)
        while padding_top < 0 or padding_top > padding:
            padding_top = round(np.random.randn()* padding)

        padding_bottom = 32 - h - padding_top

        if top_or_bottom==1:
            temp=padding_bottom
            padding_bottom=padding_top
            padding_top=temp

        img_char = ToPILImage()(img_char)
        img_char = Pad((0, padding_top, 0, padding_bottom), 0)(img_char)
        img_char = np.array(img_char)

        # 改变字符深浅
        img_char = img_char * np.random.uniform(0.5, 1)

        # np.concatenate(img_char)

        json_record_line.append({'rect_char': (
            start,
            0,
            w,
            h
        ), 'char': config.CLASS_toString[y]})
        if expression_rect_init:
            expression_rect = img_char
            expression_rect_init = False
        else:
            expression_rect = np.concatenate((expression_rect, img_char), axis=1)
        start += w
        if random.randint(0, 4) == 0 or config.CLASS_toString[y] == '1':
            while_padding = random.randint(4, 16)
            start += while_padding
            expression_rect = np.concatenate((expression_rect, np.zeros((32, while_padding))), axis=1)

    # 添加噪点
    expression_rect = expression_rect + np.random.randn(expression_rect.shape[0], expression_rect.shape[1]) * 10
    # 添加阴影
    if random.randint(0, 1) == 1:
        shadow = np.roll(np.linspace(20, 200, expression_rect.shape[1]).reshape((expression_rect.shape[1])), axis=0,
                         shift=random.randint(0, 400))
    else:
        shadow = np.roll(np.linspace(200, 20, expression_rect.shape[1]).reshape((expression_rect.shape[1])), axis=0,
                         shift=random.randint(0, 400))
    shadow = np.repeat(shadow,32).reshape(32,expression_rect.shape[1])
    shadow = imutils.rotate_bound(shadow,random.randint(0,180))
    shadow = cv2.resize(shadow,(expression_rect.shape[1],expression_rect.shape[0]))


    # cv2.imshow("",shadow)
    # cv2.waitKey()
    expression_rect = expression_rect + shadow

    expression_rect = expression_rect - expression_rect.min()
    expression_rect = 255 - (expression_rect / expression_rect.max() * 255)

    width = start
    half = (width / 81) * 0.5
    target_text_start = ['b'] * 81
    target_text_end = ['b'] * 81
    target_text_mix = ['b'] * 81
    for char in json_record_line:
        char_start = round(((char['rect_char'][0]) * 81) / width)
        char_end = round(((char['rect_char'][0] + char['rect_char'][2]) * 81) / width)
        # print(char_start, char_end, char_end - char_start)
        target_text_start[char_start] = 's'
        # print(target_text_mix)
        for i in range(char_start, char_end):
            target_text_end[i] = char['char']
        # target_text_center[(char_start+char_end)//2] = char['char']
    for i in range(0, 81):
        if target_text_start[i] == 's':
            target_text_mix[i] = 's'
        else:
            target_text_mix[i] = target_text_end[i]
    # print(''.join(target_text_start))
    # print(''.join(target_text_end))
    # print(''.join(target_text_mix))

    generate_name = "%d.png" % generate_count
    cv2.imwrite(os.path.join(generate_save_path, generate_name), expression_rect)
    # print(generate_name + ' ' + ''.join(target_text_mix))
    generate_labels.append(generate_name + ' ' + ''.join(target_text_mix))

annotation_path = os.path.join(generate_save_path, "anno.txt")
with open(annotation_path, encoding="utf-8", mode="w") as file:
    for line in generate_labels:
        file.write(line + "\n")

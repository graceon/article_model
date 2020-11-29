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
max_width = 100
label_len = max_width // 4 + 1
generate_count = 0
generate_count_max = 30000
generate_save_path = './generator'
generate_labels = []


def get_random_char(set):
    random_index = random.randint(0, len(set) - 1)
    img_char = set[random_index][0][0].numpy()
    y = set[random_index][1].item()

    rotate_angle = random.randint(-10, 10)
    img_char = imutils.rotate_bound(img_char, rotate_angle)
    contour = []

    for hw in range(char_hw * char_hw):
        if img_char[hw % char_hw][hw // char_hw] > 50:
            contour.append([hw % char_hw, hw // char_hw])
    if len(contour) < 4:
        return get_random_char(set)
    contour = np.array(contour, dtype=np.int8)
    img_char = img_char[
               contour[:, 0].min():contour[:, 0].max() + 1,
               contour[:, 1].min():contour[:, 1].max() + 1
               ]
    return img_char, y


big_set = custom_dataset.custom_dataset("package_bigset.pth", False)
print_set = custom_dataset.custom_dataset("package_print_set.pth", False)
# plt.matshow(get_random_char(train_set))
# plt.show()


while generate_count < 30000:
    generate_count += 1
    expression_rect = 0
    expression_rect_init = True

    start = 0
    w = 0
    h = 0
    json_record_line = []
    is_print = random.randint(0, 1) == 1

    while start <= max_width:
        if is_print:
            img_char, y = get_random_char(print_set)
            # 0-9表示数字 大于等于10是操作符
            if y >= 10:
                start += 1
                h = img_char.shape[0]

                if h < 4:
                    fy = 1
                else:
                    fy = random.uniform(0.2, 0.5)
                img_char = cv2.resize(img_char, (0, 0), fx=random.uniform(0.4, 0.9), fy=fy,
                                      interpolation=cv2.INTER_CUBIC)
                padding = 32 - h
                padding_top = round(random.uniform(0.4, 0.6) * padding)
                # print(padding,padding_top)
                while padding_top < 0 or padding_top > padding:
                    padding_top = round(np.random.randn() * padding)
                padding_bottom = 32 - h - padding_top
                img_char = ToPILImage()(img_char)
                img_char = Pad((0, padding_top, 0, padding_bottom), 0)(img_char)
                img_char = np.array(img_char)
        else:
            img_char, y = get_random_char(big_set)

        h = img_char.shape[0]
        w = img_char.shape[1]
        img_char = cv2.resize(img_char, (0, 0), fx=1, fy=32 / h,
                              interpolation=cv2.INTER_CUBIC)
        # 改变字符深浅

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
            while_padding = random.randint(1, 10)
            start += while_padding
            expression_rect = np.concatenate((expression_rect, np.zeros((32, while_padding))), axis=1)

    expression_rect = expression_rect + np.random.randn(expression_rect.shape[0], expression_rect.shape[1]) * 10

    expression_rect = expression_rect - expression_rect.min()


    expression_rect = 255 - (expression_rect / expression_rect.max() * 255)

    # print(expression_rect.shape)
    # expression_rect = cv2.cvtColor(expression_rect, cv2.COLOR_BGR2GRAY)
    # print(expression_rect.shape)
    # plt.matshow(expression_rect)
    # plt.show()
    # print(expression_rect.type())


    # expression_rect = cv2.adaptiveThreshold(expression_rect, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                       cv2.THRESH_BINARY,31,10)

    width = start

    half = (width / label_len) * 0.5
    target_text_start = ['b'] * label_len
    target_text_end = ['b'] * label_len
    target_text_mix = ['b'] * label_len
    for char in json_record_line:
        char_start = round(((char['rect_char'][0]) * label_len) / width)
        char_end = round(((char['rect_char'][0] + char['rect_char'][2]) * label_len) / width)
        target_text_start[char_start] = 's'
        for i in range(char_start, char_end):
            target_text_end[i] = char['char']
    for i in range(0, label_len):
        if target_text_start[i] == 's':
            target_text_mix[i] = 's'
        else:
            target_text_mix[i] = target_text_end[i]
    # print(''.join(target_text_start))
    # print(''.join(target_text_end))
    # print(''.join(target_text_mix))

    generate_name = "%d.png" % generate_count



    cv2.imwrite(os.path.join(generate_save_path, generate_name), expression_rect)
    generate_labels.append(generate_name + ' ' + ''.join(target_text_mix))

annotation_path = os.path.join(generate_save_path, "anno.txt")
val_annotation_path = os.path.join(generate_save_path, "val_anno.txt")
line_cnt = 0
with open(annotation_path, encoding="utf-8", mode="w") as file:
    while line_cnt < len(generate_labels) * 0.9:
        file.write(generate_labels[line_cnt] + "\n")
        line_cnt += 1
with open(val_annotation_path, encoding="utf-8", mode="w") as file:
    while line_cnt < len(generate_labels):
        file.write(generate_labels[line_cnt] + "\n")
        line_cnt += 1

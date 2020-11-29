import matplotlib.pyplot as plt

import numpy as np

from torchvision.transforms import Resize, Pad, ToPILImage, ToTensor
import config
import random
import cv2
import os
import imutils
import math
import custom_dataset

char_hw = 32
max_width = 100
label_len = max_width // 4 + 1
generate_count = 0
generate_save_path = './generator'
generate_labels = []


def json_to_label(json_record_line, width):
    target_text_start = ['b'] * label_len
    target_text_main = ['b'] * label_len
    target_text_mix = ['b'] * label_len
    for char in json_record_line:
        char_start = math.floor(((char['rect_char'][0]) * label_len) / width)
        char_end = math.floor(((char['rect_char'][0] + char['rect_char'][2]) * label_len) / width)
        target_text_start[char_start] = 's'
        if char_end == label_len:
            char_end -= 1
        for i in range(char_start, char_end + 1):
            target_text_main[i] = char['char']
    for i in range(0, label_len):
        if target_text_main[i] == '=':
            target_text_mix[i] = '='
        elif target_text_main[i] == '1':
            target_text_mix[i] = '1'
        elif target_text_start[i] == 's':
            target_text_mix[i] = 's'
        else:
            target_text_mix[i] = target_text_main[i]
    return target_text_mix


def save_generate_image(path, image, adaptive=True):
    cv2.imwrite(path, image)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 减号直接用adaptive存在问题
    if False:
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 10)
    else:
        _, image = cv2.threshold(image, 160, 255,
                                 cv2.THRESH_BINARY_INV)
    cv2.imwrite(path, image)


def get_random_char(set):
    random_index = random.randint(0, len(set) - 1)
    img_char = set[random_index][0][0].numpy()
    y = set[random_index][1].item()

    rotate_angle = random.randint(-10, 10)
    img_char = imutils.rotate_bound(img_char, rotate_angle)
    contour = []

    for hw in range(char_hw * char_hw):
        if img_char[hw % char_hw][hw // char_hw] > 60:
            contour.append([hw % char_hw, hw // char_hw])

    # if len(set) < 10000:
    # >50的点少于4个，未知原因
    if len(contour) < 4:
        return get_random_char(set)
    contour = np.array(contour, dtype=np.int8)
    img_char = img_char[
               contour[:, 0].min():contour[:, 0].max() + 1,
               contour[:, 1].min():contour[:, 1].max() + 1
               ]
    return img_char, y


def crop(img_char, rotate=True):
    img_char = img_char.numpy()
    rotate_angle = random.randint(-10, 10)
    if rotate:
        img_char = imutils.rotate_bound(img_char, rotate_angle)
    contour = []
    for hw in range(char_hw * char_hw):
        if img_char[hw % char_hw][hw // char_hw] > 60:
            contour.append([hw % char_hw, hw // char_hw])
    if len(contour) < 4:
        return img_char, False
    contour = np.array(contour, dtype=np.int8)
    img_char = img_char[
               contour[:, 0].min():contour[:, 0].max() + 1,
               contour[:, 1].min():contour[:, 1].max() + 1
               ]
    return img_char, True


print_set_equal = custom_dataset.custom_dataset("package_print_set_equal.pth", False)

print_set = custom_dataset.custom_dataset("package_print_set.pth", False)
big_set = custom_dataset.custom_dataset("package_bigset.pth", False)


def generate_single(set, rotate):
    global generate_count
    for item, y in set:
        json_record_line = []
        img_char, ok = crop(item[0], rotate)
        if not ok:
            continue
        h = img_char.shape[0]
        w = img_char.shape[1]
        fy = 32 / h
        fx = fy
        img_char = cv2.resize(img_char, (0, 0), fx=fx, fy=fy,
                              interpolation=cv2.INTER_CUBIC)

        if img_char.shape[1] < 100:
            expression_rect = np.concatenate((img_char, np.zeros((32, 100 - img_char.shape[1]))), axis=1)
        else:
            expression_rect = img_char[:, 0:100]
        expression_rect = 255 - expression_rect
        json_record_line.append({'rect_char': (
            0,
            0,
            w,
            h
        ), 'char': config.CLASS_toString[y]})

        width = 100
        target_text_mix = json_to_label(json_record_line, width)

        generate_name = "%d.png" % generate_count
        save_generate_image(os.path.join(generate_save_path, generate_name), expression_rect, adaptive=(y != 11))
        generate_labels.append(generate_name + ' ' + ''.join(target_text_mix))

        generate_count += 1


if True:
    for i in range(0, 10):
        generate_single(print_set, rotate=True)
        print("generate_count=", generate_count)
    generate_single(print_set, rotate=False)
    print("generate_count=", generate_count)
    generate_single(big_set, rotate=False)
    print("generate_count=", generate_count)

y_equal = config.CLASS.index('equal')

# plt.matshow(get_random_char(train_set))
# plt.show()

mul_end = generate_count + 90000
while generate_count < mul_end:
    if generate_count % 100 == 1:
        print("generate_count=", generate_count)

    expression_rect = 0
    expression_rect_init = True

    start = 0
    w = 0
    h = 0
    json_record_line = []
    is_print = random.randint(0, 1) == 1

    last_middle = 0
    if is_print:
        while start <= max_width:
            img_char, y = get_random_char(print_set)
            # 0-9表示数字 大于等于10是操作符
            if y >= 10:
                continue
            h = img_char.shape[0]
            w = img_char.shape[1]
            img_char = cv2.resize(img_char, (0, 0), fx=1, fy=32 / h * random.uniform(0.8, 1),
                                  interpolation=cv2.INTER_CUBIC)
            h = img_char.shape[0]
            w = img_char.shape[1]
            padding = 32 - h
            padding_top = math.floor(random.uniform(0.01, 0.99) * padding)
            while padding_top < 0 or padding_top > padding:
                padding_top = math.floor(np.random.randn() * padding)
            padding_bottom = 32 - h - padding_top
            img_char = ToPILImage()(img_char)
            img_char = Pad((0, padding_top, 0, padding_bottom), 0)(img_char)
            img_char = np.array(img_char)
            img_char = 255 * (img_char > 60)

            if img_char.shape[1] + start >= max_width:
                break

            roll_count = 0
            if expression_rect_init:
                expression_rect = np.concatenate(
                    (
                        np.zeros((32, start)),
                        img_char,
                        np.zeros((32, max_width - (w + start)))
                    ), axis=1)
                expression_rect_init = False
            else:
                w = img_char.shape[1]
                expression_rect_add = np.concatenate(
                    (
                        np.zeros((32, start)),
                        img_char,
                        np.zeros((32, max_width - (w + start)))
                    ), axis=1)
                expression_rect_add_try = np.roll(expression_rect_add, -1, axis=1)
                while start - roll_count > last_middle and (expression_rect > 20).sum() + (
                        expression_rect_add_try > 20).sum() == (
                        (expression_rect + expression_rect_add_try) > 20).sum():
                    expression_rect_add = expression_rect_add_try
                    roll_count += 1
                    expression_rect_add_try = np.roll(expression_rect_add, -1, axis=1)
                expression_rect = expression_rect + expression_rect_add
                last_middle = start + (w // 2)
            json_record_line.append({'rect_char': (
                start - roll_count,
                0,
                w,
                h
            ), 'char': config.CLASS_toString[y]})

            start = start + w - roll_count
            # if config.CLASS_toString[y] == '1':
            #     white_padding = random.randint(1, 3)
            #     start += white_padding
            #     expression_rect = np.concatenate((expression_rect, np.zeros((32, white_padding))), axis=1)

    else:
        while start <= max_width:

            if start == 0 and random.randint(0, 1) == 1:
                img_char, y = get_random_char(print_set_equal)
                img_char = img_char - img_char.min()
                img_char = img_char / img_char.max() * 255
            else:
                img_char, y = get_random_char(big_set)
            h = img_char.shape[0]
            if y == 0:
                # 生成的数字是0
                fx = (32 / h) * random.uniform(0.4, 0.9)
                fy = fx
            elif y == y_equal:
                fx = (32 / h) * random.uniform(0.2, 0.8)
                fy = (32 / h) * random.uniform(0.2, 0.8)
            else:
                fx = (32 / h) * random.uniform(0.6, 0.9)
                fy = fx
            img_char = cv2.resize(img_char, (0, 0), fx=fx, fy=fy,
                                  interpolation=cv2.INTER_CUBIC)
            h = img_char.shape[0]
            padding = 32 - h
            padding_top = math.floor(random.uniform(0.01, 0.99) * padding)
            while padding_top < 0 or padding_top > padding:
                padding_top = math.floor(np.random.randn() * padding)
            padding_bottom = 32 - h - padding_top
            img_char = ToPILImage()(img_char)
            img_char = Pad((0, padding_top, 0, padding_bottom), 0)(img_char)
            img_char = np.array(img_char)

            h = img_char.shape[0]
            w = img_char.shape[1]
            img_char = cv2.resize(img_char, (0, 0), fx=1, fy=32 / h,
                                  interpolation=cv2.INTER_CUBIC)
            img_char = 255 * (img_char > 60)

            if img_char.shape[1] + start >= max_width:
                break

            roll_count = 0
            if expression_rect_init:
                expression_rect = np.concatenate(
                    (
                        np.zeros((32, start)),
                        img_char,
                        np.zeros((32, max_width - (w + start)))
                    ), axis=1)
                expression_rect_init = False
            else:
                w = img_char.shape[1]
                expression_rect_add = np.concatenate(
                    (
                        np.zeros((32, start)),
                        img_char,
                        np.zeros((32, max_width - (w + start)))
                    ), axis=1)

                expression_rect_add_try = np.roll(expression_rect_add, -1, axis=1)

                while start - roll_count > last_middle and (expression_rect > 20).sum() + (
                        expression_rect_add_try > 20).sum() == (
                        (expression_rect + expression_rect_add_try) > 20).sum():
                    expression_rect_add = expression_rect_add_try
                    roll_count += 1
                    expression_rect_add_try = np.roll(expression_rect_add, -1, axis=1)
                expression_rect = expression_rect + expression_rect_add
            last_middle = start + (w // 2)
            json_record_line.append({'rect_char': (
                start - roll_count,
                0,
                w,
                h
            ), 'char': config.CLASS_toString[y]})

            start = start + w - roll_count
            # if config.CLASS_toString[y] == '1':
            #     white_padding = random.randint(1, 3)
            #     start += white_padding
            #     expression_rect = np.concatenate((expression_rect, np.zeros((32, white_padding))), axis=1)

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

    # expression_rect = np.concatenate((expression_rect, np.ones((32, 100-w))*255), axis=1)

    width = 100
    if expression_rect.shape[1] < 100:
        expression_rect = np.concatenate(
            (
                expression_rect,
                np.ones((32, max_width - expression_rect.shape[1])) * 255
            ), axis=1)

    # half = (width / label_len) * 0.5

    target_text_mix = json_to_label(json_record_line, width)

    generate_name = "%d.png" % generate_count
    # plt.matshow(expression_rect)
    # print(target_text_mix)
    # plt.show()
    save_generate_image(os.path.join(generate_save_path, generate_name), expression_rect)
    generate_labels.append(generate_name + ' ' + ''.join(target_text_mix))
    generate_count += 1

annotation_path = os.path.join(generate_save_path, "anno.txt")
val_annotation_path = os.path.join(generate_save_path, "val_anno.txt")
line_cnt = 0
with open(annotation_path, encoding="utf-8", mode="w") as file:
    while line_cnt < len(generate_labels) - 3000:
        file.write(generate_labels[line_cnt] + "\n")
        line_cnt += 1
with open(val_annotation_path, encoding="utf-8", mode="w") as file:
    while line_cnt < len(generate_labels):
        file.write(generate_labels[line_cnt] + "\n")
        line_cnt += 1

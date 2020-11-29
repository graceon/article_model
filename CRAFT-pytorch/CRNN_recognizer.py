import sys


path_to_CRNN_Chinese_Characters_Rec = "E:/article_model/CRNN_Chinese_Characters_Rec-stable"

sys.path.append(path_to_CRNN_Chinese_Characters_Rec)
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import lib.utils.utils as utils
import lib.models.crnn as crnn
import lib.config.alphabets as alphabets
import yaml
from easydict import EasyDict as edict
import argparse
from dir_utils import debug_write
# checkpoint_path= "output/OWN/crnn/"+"2020-11-05-22-16"+"/checkpoints/"+"checkpoint_16_acc_1.0000.pth"

CRNN_project_path = path_to_CRNN_Chinese_Characters_Rec+'/'

checkpoint_path = CRNN_project_path + "best.pth"


def parse_arg():
    parser = argparse.ArgumentParser(description="demo")

    parser.add_argument('--cfg', help='experiment configuration filename', type=str,
                        default=CRNN_project_path + 'lib/config/OWN_config.yaml')
    parser.add_argument('--image_path', type=str, default='images/test_2.png', help='the path to your image')
    parser.add_argument('--checkpoint', type=str, default=checkpoint_path,
                        help='the path to your checkpoints')

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config, args


def recognition(config, img, model, converter, device):
    # github issues: https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec/issues/211
    # _,img = cv2.threshold(img,175,255,cv2.THRESH_OTSU)
    h, w = img.shape

    # if img.mean() < 175:
    # img = 255 - img

    fy = config.MODEL.IMAGE_SIZE.H / h
    fx = fy*1.2
    # if img.shape[1] < img.shape[0]//2:
    #     fx = 4
    # else:
    #     fx = fy

    img = cv2.resize(img, (0, 0), fx=fx, fy=fy,
                     interpolation=cv2.INTER_CUBIC)

    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                             cv2.THRESH_BINARY, 31, 1)

    # pad = np.ones((img.shape[0], 2)) * 0
    # img = np.concatenate((pad,img), axis=1)
    if img.shape[0] < 80:
        pad = np.ones((img.shape[0], 100 - img.shape[0])) * 0
        img = np.concatenate((img, pad), axis=1)

    h, w = img.shape
    image_debug = img

    img = np.reshape(img, (1, 1, h, w))
    img = img.astype(np.float32)
    if img.min() != img.max():
        img = img - img.min()
    img = img / img.max()

    # img = img.transpose([2, 0, 1])

    preds = model(torch.Tensor(img).to(device))
    _, preds = preds.max(2)

    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))

    sim_pred = converter.decode(preds.data, preds_size.data, raw=True)


    debug_write(image_debug,"crnn"+sim_pred.replace('/','d'))


    # plt.matshow(img)
    # plt.show()

    # if img.shape[1]<160:
    #     img=np.concatenate(
    #         (img,np.ones((img.shape[0],160-img.shape[1]))*img.mean()),
    #         axis=1
    #     )

    # fisrt step: resize the height and width of image to (32, x)

    # print(config.MODEL.IMAGE_SIZE.H / h, config.MODEL.IMAGE_SIZE.W / w)
    # img = cv2.resize(img, (0, 0), fx=self.inp_w / img_w, fy=self.inp_h / h, interpolation=cv2.INTER_CUBIC)

    # 腐蚀图像

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # img = cv2.erode(img, kernel)

    # img = cv2.resize(img, (0, 0), fx=1, fy=config.MODEL.IMAGE_SIZE.H / h, interpolation=cv2.INTER_CUBIC)
    return sim_pred


config, args = parse_arg()
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

model = crnn.get_crnn(config).to(device)

print('loading pretrained model from {0}'.format(args.checkpoint))
checkpoint = torch.load(args.checkpoint)
if 'state_dict' in checkpoint.keys():
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()
converter = utils.strLabelConverter(config.DATASET.ALPHABETS)


def recognizer(img):
    return recognition(config, img, model, converter, device)


def compress(input):
    last_char = -1
    res = ''
    for char in input:
        if char == 'b':
            continue
        if char == 's':
            last_char = -1
            continue
        elif last_char != char:
            res += char
            last_char = char
    return res

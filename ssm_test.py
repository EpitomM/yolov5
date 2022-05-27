from card_ocr import card_ocr
import os
import cv2
import json
import random
import torchvision
import torch.nn.functional as F
from torch import nn
import numpy as np
import albumentations as A
import base64
from albumentations.augmentations.transforms import Resize,CenterCrop
from albumentations.pytorch import ToTensor
from io import BytesIO
from PIL import Image



def get_test_transform(image_size):
    longest_size = max(image_size[0], image_size[1])
    return A.Compose([
        #Resize(int(config.img_height*1.5),int(config.img_weight*1.5)),
        CenterCrop(300, 300),
        A.LongestMaxSize(longest_size, interpolation=cv2.INTER_CUBIC),

        A.PadIfNeeded(image_size[0], image_size[1],
                      border_mode=cv2.BORDER_CONSTANT, value=0),

        A.Normalize(),
        ToTensor()
    ])


if __name__ == '__main__':
    ## 0. 构造请求数据：卡片的 base64 字符串
    pic_name = 'card_detection_data_final_edition/jinritoutiao_feed_card/train/images/hupeipei_oppoa35_jinritoutiao_20211125_b46_141.jpg'
    with open(pic_name, 'rb') as f:
        image_data = f.read()
        base64_data = base64.b64encode(image_data)  # base64编码

        ## 1. 开始处理
        # base64 转为 ndarray
        img_data = base64.b64decode(base64_data)
        nparr = np.fromstring(img_data, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        print(type(img_np))

        img = get_test_transform(img_np.shape)(image=img_np)["image"]
        print(img)



import cv2
import numpy as np
import json
import os
import re
from PIL import Image
import random
import time

'''
数据集路径：card_detection_data_final_edition/jinritoutiao_feed
功能：构造卡片数据集。
'''

'''
方法说明：裁剪图片
    @param img 原图像
    @param obj，obj 数据举例如
    {
        "region_attributes": {
            "propertyName": "直播"
        },
        "shape_attributes": {
            "name": "card",
            "x": -3,
            "y": 275,
            "width": 718,
            "height": 650
    }
    返回值：裁剪后图像
'''
def crop_image(img, obj):
    shape_attributes = obj.get('shape_attributes')
    # 获得卡片的左上角、右下角坐标
    x_left_top = int(shape_attributes.get('x'))
    y_left_top = int(shape_attributes.get('y'))
    width = shape_attributes.get('width')
    height = shape_attributes.get('height')
    if x_left_top < 0:
        x_left_top = 0
    if y_left_top < 0:
        y_left_top = 0
    # 判定左上角坐标、宽、高是否都大于0
    if x_left_top >= 0 and y_left_top >= 0 and width > 0 and height > 0:
        x_right_bottom = int(x_left_top + width)
        y_right_bottom = int(y_left_top + height)
        print(x_left_top, y_left_top, x_right_bottom, y_right_bottom)
        # 裁剪图片中的卡片
        cropped = img[y_left_top: y_right_bottom, x_left_top: x_right_bottom]   # 裁剪坐标为[y0:y1, x0:x1]
        return cropped




def get_x_y(shape_attributes):
    x_left_top = int(shape_attributes.get('x'))
    y_left_top = int(shape_attributes.get('y'))
    width = shape_attributes.get('width')
    height = shape_attributes.get('height')
    if x_left_top < 0:
        x_left_top = 0
    if y_left_top < 0:
        y_left_top = 0
    # 判定左上角坐标、宽、高是否都大于0
    if x_left_top >= 0 and y_left_top >= 0 and width > 0 and height > 0:
        x_right_bottom = int(x_left_top + width)
        y_right_bottom = int(y_left_top + height)
        return x_left_top, y_left_top, x_right_bottom, y_right_bottom
    return None, None, None, None


def split_card(json_path, picture_path):
    start = time.time()
    # 获得所有图片名字
    picture_names = os.listdir(picture_path)
    # 排序
    picture_names.sort()
    # 共 96031 个数据，先处理前 30000 个
    picture_names = picture_names[: 30000]
    # 获得所有 json 文件名
    json_file_names = os.listdir(json_path)
    # 遍历所有 json 文件
    # json_file = 'abudukadierjiang_oppoa11_jinritoutiao_20211115_b31_12.json'
    for json_file in json_file_names:
        # 生成 [0,9] 的随机数，决定该图片放到训练集还是测试集
        rand_num = random.randint(0, 9)
        # json 文件全路径
        json_file_path = os.path.join(json_path, json_file)
        # 去掉 .json 后缀
        json_name = json_file[: -5]
        # 图片后缀
        suffix = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng', '.webp', '.mpo',
        '.BMP', '.JPG', '.JPEG', '.PNG', '.TIF', '.TIFF', '.DNG', '.WEBP', '.MPO']
        # 因为不知道图片后缀名，所以只能遍历所有后缀
        for su in suffix:
            # 带后缀的图片名
            pic_name = json_name + su
            # 找到该图片正确的后缀
            if pic_name in picture_names:
                # 标记 card 序号
                number = 0
                try:
                    # 图片全路径
                    picture_full_path = os.path.join(picture_path, pic_name)
                    print('图片全路径', picture_full_path)
                    # 读取图片
                    img = cv2.imread(picture_full_path)
                    if img is not None:
                        # 打开每一个 json 文件
                        with open(json_file_path, 'r', encoding='utf-8') as f_json:
                            # 加载 json
                            data = json.load(f_json)
                            objects = data.get('objects')
                            # 遍历所有框
                            for obj in objects:
                                shape_attributes = obj.get('shape_attributes')
                                region_attributes = obj.get('region_attributes')
                                # 获取卡片类型，如图文、广告、视频等
                                propertyName = region_attributes.get('propertyName')
                                if shape_attributes is not None:
                                    name = shape_attributes.get('name')
                                    # 找到卡片
                                    if name == 'card':
                                        print('this is a card...', shape_attributes)
                                        # 裁剪图片
                                        # cropped = crop_image(img, obj)
                                        # 获得卡片的左上角、右下角坐标
                                        x_left_top, y_left_top, x_right_bottom, y_right_bottom = get_x_y(shape_attributes)
                                        if x_left_top is None or y_left_top is None or x_right_bottom is None or y_right_bottom is None:
                                            print('坐标值不合法')
                                            continue
                                        print(x_left_top, y_left_top, x_right_bottom, y_right_bottom)
                                        # 裁剪图片中的卡片
                                        cropped = img[y_left_top: y_right_bottom,
                                                  x_left_top: x_right_bottom]  # 裁剪坐标为[y0:y1, x0:x1]
                                        new_picture_path = 'card_detection_data_final_edition/jinritoutiao_feed_card_yolo/picture/'
                                        label_file = 'card_detection_data_final_edition/jinritoutiao_feed_card_yolo/json/'
                                        # if rand_num > 8:
                                        #     new_picture_path = 'card_detection_data_final_edition/jinritoutiao_feed_card_yolo/val/picture/'
                                        #     label_file = 'card_detection_data_final_edition/jinritoutiao_feed_card_yolo/val/json/'
                                        new_path = new_picture_path + json_name + str(number) + su

                                        # 保存卡片
                                        cv2.imwrite(new_path, cropped)
                                        print('图片保存成功...', new_path)

                                        # 构建卡片的 json 标签文件

                                        # 找到这张卡片中的物体
                                        card_objects = []
                                        # 遍历所有物体
                                        for obj_in_card in objects:
                                            shape_attributes = obj_in_card.get('shape_attributes')
                                            if shape_attributes is not None:
                                                name = shape_attributes.get('name')
                                                if name == 'rect' or name == 'card':
                                                    # 获得物体的左上角、右下角 x y
                                                    in_card_x_left_top, in_card_y_left_top, in_card_x_right_bottom, in_card_y_right_bottom = get_x_y(shape_attributes)
                                                    # 如果值不合法
                                                    if in_card_x_left_top is None or in_card_y_left_top is None or in_card_x_right_bottom is None or in_card_y_right_bottom is None:
                                                        print('坐标值不合法')
                                                        continue
                                                    # 找到这张卡片中的物体
                                                    if in_card_x_left_top >= x_left_top and in_card_y_left_top >= y_left_top and in_card_x_right_bottom <= x_right_bottom and in_card_y_right_bottom <= y_right_bottom:
                                                        obj_in_card.get('shape_attributes')['x'] = in_card_x_left_top - x_left_top
                                                        obj_in_card.get('shape_attributes')['y'] = in_card_y_left_top - y_left_top
                                                        card_objects.append(obj_in_card)
                                        data = {"autoTransferIndex": 1,
                                                     "image_name": json_name + str(number) + su,
                                                     "objects": card_objects,
                                                     "is_card_added": "true"
                                                    }
                                        new_json_path = label_file + json_name + str(number) + '.json'
                                        # 写入 json 文件
                                        with open(new_json_path, 'w', encoding='utf-8') as write_f:
                                            write_f.write(json.dumps(data, indent=4, ensure_ascii=False))
                                        print('标签保存成功...', new_json_path)
                                        number += 1


                except Exception as e:
                    print(e)
                    print('错误图片：', picture_full_path)
                    with open('err_yolo.log', 'a', encoding='utf-8') as f_write:
                        f_write.write(picture_full_path + '\n')

    end = time.time()
    use = end-start
    print('总耗时(s): ', use)




from sklearn.metrics import classification_report
classification_report()

if __name__ == "__main__":
    base_path = 'card_detection_data_final_edition'
    dataset_name = 'jinritoutiao_feed/'
    json_path = os.path.join(base_path, dataset_name + 'json')
    picture_path = os.path.join(base_path, dataset_name + 'picture')
    split_card(json_path, picture_path)
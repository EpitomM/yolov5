from split_card import crop_image
from card_ocr import card_ocr
import os
import cv2
import json
import random

if __name__ == '__main__':
    # filePath = 'card_detection_data_final_edition/jinritoutiao_feed/picture/yanghan_huaweichangxiang20_jinritoutiao_20211118_b33_10.jpg'
    # with open(filePath, 'rb') as fp:
    #     f_res = fp.read()
    #     print(type(f_res))

    base_path = 'card_detection_data_final_edition'
    dataset_name = 'jinritoutiao_feed/'
    json_path = os.path.join(base_path, dataset_name + 'json')
    picture_path = os.path.join(base_path, dataset_name + 'picture')
    picture_names = os.listdir(picture_path)
    json_file = 'yanghan_huaweichangxiang20_jinritoutiao_20211118_b33_10.json'
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
            # try:
            # 图片全路径
            picture_full_path = os.path.join(picture_path, pic_name)
            print(picture_full_path)
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
                                cropped = crop_image(img, obj)
                                print('crop type', type(cropped))
                                rand_pic_name = str(random.randint(0, 9)) + '.jpg'
                                cv2.imwrite(rand_pic_name, cropped)
                                ocr_results = card_ocr(rand_pic_name)
                                print(ocr_results)

                                res_word = ''
                                for res in ocr_results:
                                    res_word += res.get('words')
                                print(res_word)
                                os.remove(rand_pic_name)
            # except Exception as e:
            #     print(e)
            #     print('错误图片：', picture_full_path)
            #     with open('err.log', 'a', encoding='utf-8') as f_write:
            #         f_write.write(picture_full_path + '\n')
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import random
import collections
import json
import yaml
import os
import re
import traceback
from tqdm import tqdm

def get_file_list(path):
    for _, _, files in os.walk(path):
        return files

def polyline_card_gen(objects):
    """建立一个数组用来给所有polyline排序"""
    polyline_list = []
    for ori_label in objects:
        if ori_label['shape_attributes']['name'] == 'polyline':
            # 类别，x左，x右，y
            polyline_list.append([ori_label['region_attributes']['propertyName'],
                                ori_label['shape_attributes']["all_points_x"][0],
                                ori_label['shape_attributes']["all_points_x"][1],
                                ori_label['shape_attributes']["all_points_y"][0]])
            sorted(polyline_list,key=(lambda x:x[3]))
    object_list = objects
    #print("card数：",len(polyline_list)-1)
    for i in range(len(polyline_list)-1):
        card_region_attributes = collections.OrderedDict()
        card_shape_attributes = collections.OrderedDict()
        card_region_attributes["propertyName"] = polyline_list[i][0]
        card_shape_attributes["name"] = "card"
        card_shape_attributes["x"] = polyline_list[i][1] + 5 #初始x左上角加5
        card_shape_attributes["y"] = polyline_list[i][3] + 5
        card_shape_attributes["width"] = polyline_list[i][2] - polyline_list[i][1] - 10 #宽度两侧各减5
        card_shape_attributes["height"] = polyline_list[i+1][3] - polyline_list[i][3] - 10
        card_dict= collections.OrderedDict([("region_attributes",card_region_attributes),
                                            ("shape_attributes",card_shape_attributes)])
        """将card标注添加到标注中"""
        object_list.append(card_dict)
    return False, object_list

def card_add(dataset_num,dataset_name, path="card_detection_data_final_edition"):
    """
    检测一个图片的json文件，所有polyline，上下都有的情况下，添加一个card
        card的属性以图片宽小一点，高为下减上，xy为左上角
    rect里面是中文的情况下，添加一个card
    """
    """地址初始化"""
    path = os.path.join(path, dataset_name)
    images_path = os.path.join(path, 'picture')
    jsons_path = os.path.join(path, 'json')
    """获取数据集文件列表"""
    files = get_file_list(jsons_path)
    data_item_num = 0
    """标签种类初始化
        下述五种出现在polyline中说明无card标注
        下述五种出现在rect上说明card标注在rect中需调整
    """
    object_type = ['图文', '广告', '直播', '视频', '动态']
    """遍历数据进行预处理"""
    for file_name in tqdm(files):
        if data_item_num == dataset_num:
            break
        file = open(os.path.join(jsons_path, file_name), 'r', encoding='utf-8')
        json_dict = json.load(file)
        if "is_card_added" in json_dict:
            print("is_card_added")
            data_item_num += 1
            continue
        if 'image_name' in json_dict:
            image_name = json_dict['image_name']
        elif 'filename' in json_dict:
            image_name = json_dict['filename']
        else:
            print("出错了")
            continue
        image_path = os.path.join(images_path,image_name)
        if not os.path.exists(os.path.join(image_path)):
            continue
        temp = None
        if 'objects' in json_dict:
            temp = json_dict['objects']
        elif 'regions' in json_dict: # 剔除键名 regions
            json_dict["objects"] = json_dict.pop("regions")
            temp = json_dict["objects"]
        else:
            print(json_dict)
            continue
        polyline_flag = True
        object_list = json_dict['objects']
        for ori_label in temp:
            """
            检测一个图片的json文件，所有polyline，上下都有的情况下，添加一个card
                card的属性以图片宽小一点，高为下减上，xy为左上角
            rect里面是中文的情况下，添加一个card
            """
            try:
                shape_type = ori_label['shape_attributes']['name']
                property = ori_label['region_attributes']['propertyName']
                if shape_type == 'polyline' and property in object_type and polyline_flag:
                    polyline_flag, object_list = polyline_card_gen(object_list)
                elif shape_type == 'rect' and property in object_type:
                    card_region_attributes = collections.OrderedDict()
                    card_shape_attributes = collections.OrderedDict()
                    card_region_attributes["propertyName"] = ori_label['region_attributes']['propertyName']
                    card_shape_attributes["name"] = "card"
                    card_shape_attributes["x"] = ori_label['shape_attributes']["x"]
                    card_shape_attributes["y"] = ori_label['shape_attributes']["y"]
                    card_shape_attributes["width"] = ori_label['shape_attributes']["width"]
                    card_shape_attributes["height"] = ori_label['shape_attributes']["height"]
                    card_dict= collections.OrderedDict([("region_attributes",card_region_attributes),
                                                        ("shape_attributes",card_shape_attributes)])
                    """将card标注添加到标注中"""
                    object_list.append(card_dict)
            except Exception as e:
                """错误情况应该写入错误日志，不应在控制台直接输出"""
                print("错误类型：", e.__class__.__name__)
                print("错误明细：", e)
                traceback.print_exc()
                print("异常标签：", ori_label)
                print("异常文件路径：", jsons_path, file_name)
        """保存json文件"""
        json_dict["objects"] = object_list
        json_dict["is_card_added"] = True
        json_str = json.dumps(json_dict, ensure_ascii=False, indent=4)
        #print(json_str)
        with open(os.path.join(jsons_path, file_name), 'w', encoding='utf-8') as f:
            f.write(json_str)
        data_item_num += 1
    print("处理样本总数：", data_item_num)

def review(dataset_num,dataset_name, path="card_detection_data_final_edition"):
    """检查是否card已添加"""
    """地址初始化"""
    path=os.path.join(path,dataset_name)
    images_path=os.path.join(path,'picture')
    jsons_path=os.path.join(path,'json')
    """获取数据集文件列表"""
    files=get_file_list(jsons_path)
    data_item_num = 0
    object_type = ['图文', '广告', '直播', '视频', '动态']
    for file_name in files:
        if data_item_num == dataset_num:
            break
        if data_item_num % 100 == 0:
            print("已检查样本数：",data_item_num)
        file = open(os.path.join(jsons_path,file_name), 'r', encoding='utf-8')
        json_dict = json.load(file)
        temp = None
        if 'objects' in json_dict:
            temp = json_dict['objects']
        elif 'regions' in json_dict: # 剔除键名 regions
            json_dict["objects"] = json_dict.pop("regions")
            temp = json_dict["objects"]
        else:
            print(json_dict)
            continue
        for ori_label in temp:
            try:
                shape_type = ori_label['shape_attributes']['name']
                property = ori_label['region_attributes']['propertyName']
                if shape_type == 'card' and property in object_type:
                    print("111")
            except Exception as e:
                """错误情况应该写入错误日志，不应在控制台直接输出"""
                print("错误类型：", e.__class__.__name__)
                print("错误明细：", e)
                traceback.print_exc()
                print("异常标签：", ori_label)
                print("异常文件路径：", jsons_path, file_name)
        data_item_num += 1
    print("检查样本总数：", data_item_num)

if __name__ == "__main__":
    card_add(2, "baidu_feed")
    #review(2000, "tengxunxinwen_feed")
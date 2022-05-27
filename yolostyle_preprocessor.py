#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import random
import json
import yaml
import os
import re
import traceback
from tqdm import tqdm

def get_file_list(path):
    for _, _, files in os.walk(path):
        return files

def preprocess(dataset_num, dataset_name, path="card_detection_data_final_edition",
            res_path="datasets"):
    """判断是否存在数据集的yaml文件，若没有则生成一个"""
    # if not os.path.exists(os.path.join("/home/work/bianyuan/feed/yolov5/data",""+dataset_name+".yaml")):
    #     yaml_dict = {
    #         "path":"/home/work/DATA_DIR/repository/FeedImg/feed/datasets/"+dataset_name,
    #         "train":"images/",
    #         "val":"images/",
    #         "nc":11,
    #         "names":['Title', 'Image', 'Author', 'Comment', 'Time', 'Detail', 'Image_Text', 'Ad', 'Live', 'Video', 'Moment']
    #     }
    #     with open(os.path.join("/home/work/bianyuan/feed/yolov5/data",""+dataset_name+".yaml"), 'w', encoding='utf-8') as f:
    #         yaml.dump(data=yaml_dict, stream=f, allow_unicode=True)
    """地址初始化"""
    path=os.path.join(path,dataset_name)
    res_path=os.path.join(res_path,dataset_name)
    images_path=os.path.join(path,'picture')
    jsons_path=os.path.join(path,'json')
    res_images_path=os.path.join(res_path,'images')
    res_labels_path=os.path.join(res_path,'labels')
    """获取文件列表"""
    files=get_file_list(jsons_path)
    data_item_num = 0
    """迭代生成yolo可识别的数据集样本"""
    for file_name in tqdm(files):
        if data_item_num == dataset_num:
            break
        file = open(os.path.join(jsons_path,file_name), 'r', encoding='utf-8')
        json_dict = json.load(file)
        if 'image_name' in json_dict:
            image_name = json_dict['image_name']
        elif 'filename' in json_dict:
            image_name = json_dict['filename']
        else:
            Exception_item = {
                "异常类型":"键名异常",
                "异常文件路径":os.path.join(jsons_path, file_name),
                "错误":"图像名的键名不是'image_name'或'filename'。",
            }
            # Exception_log_list.append(Exception_item)
            # Exception_num+=1
            continue
        image_path = os.path.join(images_path,image_name)
        if  not os.path.exists(image_path):
            continue
        """图片预处理
        图像原始尺寸 h * w  
        resize 将图片的高统一到同样的高度 hn=480 -> wn = w / h * hn
        padding 补充两侧 单侧宽度 wp = (640-wn)/2
        最终目标尺寸hf * wf = 480 * 640
        rename
        """
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        h=image.shape[0]
        w=image.shape[1]
        hf=480
        wf=640
        hn=hf
        wn=int(w/h*hn)
        wp=int((wf-wn)/2)
        wp_l=wp
        wp_r=wp
        if wp*2+wn<wf:
            wp_l=wp+1
        if wp_l+wp_r+wn!=wf:
            print(image.shape,image_name)
        image = cv.resize(image,(wn,hn))
        border = cv.copyMakeBorder(
            image,
            top=0,
            bottom=0,
            left=wp_l,
            right=wp_r,
            borderType=cv.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
        """生成画布&保存图像"""
        plt.figure(figsize=(wf/100.0,hf/100.0),facecolor='gray')
        plt.axis('off')
        plt.imshow(border, aspect='equal')
        plt.savefig(os.path.join(res_images_path,image_name),
            bbox_inches='tight',# 去除坐标轴占用的空白
            pad_inches=0.0#去除白边框
        )
        plt.close()
        """标签预处理
        根据json生成txt
        类别（polyline）：'Image_Txt', 'Ad', 'Video', 'Live', 'Moment'
        类别（rect）：'Title', 'Image', 'Author', 'Comment', 'Time', 'Detail'
        """
        object_type = ['标题', '图像', '作者', '评论', '时间', '详细内容', '图文', '广告', '直播', '视频', '动态']
        object_type_eng = ['Title', 'Image', 'Author', 'Comment', 'Time', 'Detail', 'Image_Txt', 'Ad', 'Live', 'Video', 'Moment']
        temp = None
        if 'objects' in json_dict:
            temp = json_dict['objects']
        elif 'regions' in json_dict:
            # 修改regions键名为objects
            json_dict['objects'] = json_dict.pop('regions')
            temp = json_dict['objects']
        else:
            # 其他情况不予执行
            print(json_dict)
            continue
        labels_array = []
        for ori_label in temp:
            try:
                """按类别提取"""
                shape_type = ori_label['shape_attributes']['name']
                property = ori_label['region_attributes']['propertyName']
                if shape_type == 'polyline':
                    continue
                rect_flag = False
                for label_i,label_type in enumerate(object_type[:6]):
                    if property == label_type or shape_type == object_type_eng[label_i]:
                        label_w = (ori_label['shape_attributes']["width"] / w * wn) / wf
                        label_h = (ori_label['shape_attributes']["height"] / h * hn) / hf
                        label_x = ((ori_label['shape_attributes']["x"] / w * wn) + wp_l) / wf + label_w / 2
                        label_y = (ori_label['shape_attributes']["y"] / h * hn) / hf + label_h / 2
                        labels_array.append([label_i,label_x,label_y,label_w,label_h])
                        rect_flag = True
                if rect_flag:
                    continue
                for label_i,label_type in enumerate(object_type[6:]):
                    if property == label_type:
                        label_w = (ori_label['shape_attributes']["width"] / w * wn) / wf
                        label_h = (ori_label['shape_attributes']["height"] / h * hn) / hf
                        label_x = ((ori_label['shape_attributes']["x"] / w * wn) + wp_l) / wf + label_w / 2
                        label_y = (ori_label['shape_attributes']["y"] / h * hn) / hf + label_h / 2
                        labels_array.append([label_i+6,label_x,label_y,label_w,label_h])
                        continue
            except Exception as e:
                """错误情况应该写入错误日志，不应在控制台直接输出"""
                print("错误类型：", e.__class__.__name__)
                print("错误明细：", e)
                traceback.print_exc()
                print("异常数据名：",image_name)
                print("标签信息：",ori_label)
            """保存txt文件"""
            txt_name = ""+image_name[:-4]+".txt"
            if labels_array == []:
                np.savetxt(os.path.join(res_labels_path,txt_name),labels_array)
            else:
                np.savetxt(os.path.join(res_labels_path,txt_name),
                    labels_array,
                    fmt="%d %.6f %.6f %.6f %.6f",
                    newline="\n"
                )
        data_item_num+=1
    print("处理样本总数：",data_item_num)

"""
    针对人工标注错误的修改函数
    data_type_list:标注规范指定的标签种类列表
    rect_type_list:待清洗的标签种类字典
"""
def clear_labels(json_path, json_dict, data_type_list, rect_type_list, Exception_log_list):
    sub_Exception_num = 0
    temp = None
    if 'objects' in json_dict:
        temp = json_dict['objects']
    elif 'regions' in json_dict:
        # 修改regions键名为objects
        json_dict['objects'] = json_dict.pop('regions')
        temp = json_dict['objects']
    else:
        # 其他情况不予执行
        Exception_item = {
            "异常类型":"标签列表键名异常",
            "异常文件路径":json_path,
            "错误":"读取标签失败，标签列表键名不是'objects' 或者 'regions'。",
        }
        Exception_log_list.append(Exception_item)
        sub_Exception_num+=1
        return sub_Exception_num
    cleared_labels = []
    for ori_label in temp:
        try:
            """按类别提取"""
            shape_type = ori_label['shape_attributes']['name']
            property = ori_label['region_attributes']['propertyName']
            if shape_type != 'rect':# 不是rect类别直接跳过
                continue
            if property not in data_type_list and property in rect_type_list and type(rect_type_list[property])!=int:
                ori_label['region_attributes']['propertyName'] = rect_type_list[property]
            cleared_labels.append(ori_label)
        except Exception as e:
            """添加异常记录项"""
            Exception_item = {
                "异常类型":"清洗标签过程异常",
                "异常文件路径":json_path,
                "错误":traceback.format_exc(),
                "异常标签":ori_label,
            }
            Exception_log_list.append(Exception_item)
            sub_Exception_num+=1
    """替换并保存清洗过的json文件"""
    json_dict['objects'] = cleared_labels
    json.dump(
        json_dict,open(json_path,'w',encoding='utf-8'),
        indent=4,ensure_ascii=False
    )
    return sub_Exception_num

"""第二批数据预处理成yolo模型需要的格式"""
def preprocess_2(dataset_num, dataset_name, data_type_list):
    """判断是否存在数据集的yaml文件，若没有则生成一个"""
    if not os.path.exists(os.path.join("/home/work/bianyuan/feed/yolov5/data",""+dataset_name+".yaml")):
        yaml_dict = {
            "path":"/home/work/DATA_DIR/repository/FeedImg/feed/datasets/"+dataset_name,
            "train":"images/",
            "val":"images/",
            "nc":len(data_type_list),
            "names":data_type_list
        }
        with open(os.path.join("/home/work/bianyuan/feed/yolov5/data",""+dataset_name+".yaml"), 'w', encoding='utf-8') as f:
            yaml.dump(data=yaml_dict, stream=f, allow_unicode=True)
    """地址初始化"""
    path="/home/work/DATA_DIR/repository/FeedImg/feed/card_detection_data_final_edition"
    res_path="/home/work/DATA_DIR/repository/FeedImg/feed/datasets"
    path=os.path.join(path,dataset_name)
    res_path=os.path.join(res_path,dataset_name)
    images_path=os.path.join(path,'picture')
    jsons_path=os.path.join(path,'json')
    res_images_path=os.path.join(res_path,'images')
    if not os.path.exists(res_images_path):
        os.makedirs(res_images_path)
    res_labels_path=os.path.join(res_path,'labels')
    if not os.path.exists(res_labels_path):
        os.makedirs(res_labels_path)
    Exception_log_list = []# 错误日志列表
    """获取文件列表"""
    files=get_file_list(jsons_path)
    data_item_num = 0
    Exception_num = 0
    """迭代生成yolo可识别的数据集样本"""
    for file_name in tqdm(files):
        try:
            if data_item_num == dataset_num:
                break
            file = open(os.path.join(jsons_path,file_name), 'r', encoding='utf-8')
            json_dict = json.load(file)
            if 'image_name' in json_dict:
                image_name = json_dict['image_name']
            elif 'filename' in json_dict:
                image_name = json_dict['filename']
            else:
                Exception_item = {
                    "异常类型":"键名异常",
                    "异常文件路径":os.path.join(jsons_path, file_name),
                    "错误":"图像名的键名不是'image_name'或'filename'。",
                }
                Exception_log_list.append(Exception_item)
                Exception_num+=1
                continue
            image_path = os.path.join(images_path,image_name)
            if  not os.path.exists(image_path):
                continue
            """图片预处理
            图像原始尺寸 h * w  
            resize 将图片的高统一到同样的高度 hn=480 -> wn = w / h * hn
            padding 补充两侧 单侧宽度 wp = (640-wn)/2
            最终目标尺寸hf * wf = 480 * 640
            rename
            """
            image = cv.imread(image_path)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            h=image.shape[0]
            w=image.shape[1]
            hf=480
            wf=640
            hn=hf
            wn=int(w/h*hn)
            wp=int((wf-wn)/2)
            wp_l=wp
            wp_r=wp
            if wp*2+wn<wf:
                wp_l=wp+1
            if wp_l+wp_r+wn!=wf:
                print(image.shape,image_name)
            image = cv.resize(image,(wn,hn))
            border = cv.copyMakeBorder(
                image,
                top=0,
                bottom=0,
                left=wp_l,
                right=wp_r,
                borderType=cv.BORDER_CONSTANT,
                value=[0, 0, 0]
            )
            """生成画布&保存图像"""
            plt.figure(figsize=(wf/100.0,hf/100.0),facecolor='gray')
            plt.axis('off')
            plt.imshow(border, aspect='equal')
            plt.savefig(os.path.join(res_images_path,image_name),
                bbox_inches='tight',# 去除坐标轴占用的空白
                pad_inches=0.0#去除白边框
            )
            plt.close()
            """标签预处理
            清洗标签
            根据json生成txt
            """
            Exception_num+=clear_labels(# 清洗标签函数
                os.path.join(jsons_path, file_name),
                json_dict,
                data_type_list,
                os.path.join(path, ""+dataset_name+"_rect_type_list.json"),
                Exception_log_list,
            )
            object_type = data_type_list
            temp = json_dict['objects']# 标签清洗时已经将所有标签列表键名修改为 objects
            labels_array = []
            for ori_label in temp:
                try:
                    """按类别提取"""
                    shape_type = ori_label['shape_attributes']['name']
                    property = ori_label['region_attributes']['propertyName']
                    if shape_type != 'rect':# 不是rect类别直接跳过
                        continue
                    for label_i,label_type in enumerate(object_type):
                        if property == label_type:
                            label_w = (ori_label['shape_attributes']["width"] / w * wn) / wf
                            label_h = (ori_label['shape_attributes']["height"] / h * hn) / hf
                            label_x = ((ori_label['shape_attributes']["x"] / w * wn) + wp_l) / wf + label_w / 2
                            label_y = (ori_label['shape_attributes']["y"] / h * hn) / hf + label_h / 2
                            labels_array.append([label_i,label_x,label_y,label_w,label_h])
                            continue
                except Exception as e:
                    """添加异常记录项"""
                    Exception_item = {
                        "异常类型":"标签异常",
                        "异常文件路径":os.path.join(jsons_path, file_name),
                        "错误":traceback.format_exc(),
                        "异常标签":ori_label
                    }
                    Exception_log_list.append(Exception_item)
                    Exception_num+=1
                """保存txt文件"""
                txt_name = ""+image_name[:-4]+".txt"
                if labels_array == []:
                    np.savetxt(os.path.join(res_labels_path,txt_name),labels_array)
                else:
                    np.savetxt(os.path.join(res_labels_path,txt_name),
                        labels_array,
                        fmt="%d %.6f %.6f %.6f %.6f",
                        newline="\n"
                    )
            data_item_num+=1
        except Exception:
            """添加异常记录项"""
            Exception_item = {
                "异常类型":"样本异常",
                "异常文件路径":os.path.join(jsons_path, file_name),
                "错误":traceback.format_exc()
            }
            Exception_log_list.append(Exception_item)
            Exception_num+=1
    """保存错误日志"""
    print("处理样本总数：",data_item_num)
    pre_res_dict = {
        "处理样本总数":data_item_num,
        "异常数量":Exception_num,
    }
    json.dump(
        pre_res_dict,open(os.path.join(res_path,""+dataset_name+"_preprocess_res.json"),'w',encoding='utf-8'),
        indent=4,ensure_ascii=False
    )
    Exception_log_dict = {"errors":Exception_log_list}
    json.dump(
        Exception_log_dict,open(os.path.join(res_path,""+dataset_name+"_Exception_log.json"),'w',encoding='utf-8'),
        indent=4,ensure_ascii=False
    )

def review(dataset_num, dataset_name):
    """检查图像标签是否正确"""
    pass

data_list = [
    "baidu_feed.yaml",
    "jinritoutiao_feed.yaml",
    "tengxunxinwen_feed.yaml",
    "wechat_search.yaml",
    "bilibili_search.yaml",
    "zhihu_search.yaml",
    "jinritoutiao_search.yaml",
    "shenma_search.yaml",
    "sogou_search.yaml",
    "kuake_search.yaml",
    "baidu_search.yaml",
    "redbook_search.yaml",
    "bilibili_feed.yaml"
]

data_type_dict = {
    "baidu_feed":['Title', 'Image', 'Author', 'Comment', 'Time', 'Detail', 'Image_Text', 'Ad', 'Live', 'Video', 'Moment'],
    "jinritoutiao_feed":['Title', 'Image', 'Author', 'Comment', 'Time', 'Detail', 'Image_Text', 'Ad', 'Live', 'Video', 'Moment'],
    "tengxunxinwen_feed":['Title', 'Image', 'Author', 'Comment', 'Time', 'Detail', 'Image_Text', 'Ad', 'Live', 'Video', 'Moment'],
    "wechat_search":["搜索框","自然结果","卡片","广告","标题","摘要","图片"],
    "bilibili_search":["搜索框","自然结果","卡片","广告","标题","作者","图片","播放量","评论数"],
    "zhihu_search":["搜索框","自然结果","卡片","广告","标题","摘要","图片","点赞数","评论数"],
    "jinritoutiao_search":["搜索框","自然结果","卡片","广告","标题","摘要","图片"],
    "shenma_search":["搜索框","自然结果","卡片","广告","标题","摘要","图片"],
    "sogou_search":["搜索框","自然结果","卡片","广告","标题","摘要","图片"],
    "kuake_search":["搜索框","自然结果","卡片","广告","标题","摘要","图片"],
    "baidu_search":["搜索框","自然结果","卡片","广告","标题","摘要","图片"],
    "redbook_search":["搜索框","自然结果","卡片","广告","标题","作者","图片","点赞数"],
    "bilibili_feed":["自然结果","卡片","广告","标题","作者","图片"]
}

if __name__ == "__main__":
    # for data in data_list[:3]:
    #     preprocess_2(-1, data[:-5], data_type_dict[data[:-5]])
    dataset_name = "jinritoutiao_feed"
    data_path = "card_detection_data_final_edition"
    json_path = os.path.join(data_path, dataset_name + '/json')
    json_files = os.listdir(json_path)
    dataset_num = len(json_files)
    # for json_file in json_files:
    #     json_file_path = os.path.join(json_path, json_file)
    #     with open(json_file_path, 'rb') as fp:
    #         data = json.load(fp)
    #         print(data)
    preprocess(dataset_num, dataset_name)
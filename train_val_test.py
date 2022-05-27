import os
import shutil
import random

# 保证随机可复现
random.seed(0)


# def mk_dir(file_path):
#     if os.path.exists(file_path):
#         # 如果文件夹存在，则先删除原文件夹在重新创建
#         shutil.rmtree(file_path)
#     os.makedirs(file_path)

def split_data(file_path, new_file_path, train_rate, val_rate, test_rate):
    # yolov5训练自己数据集时 准备了images图片文件夹和txt标签文件夹；但是
    # 需要分割训练集、验证集、测试集3个文件夹，每个文件夹有images和labels
    # 2个文件夹;此方法可以把imags和labels总文件夹，分割成3个文件夹;
    # file_path ='images 文件夹'
    # xmlpath= 'txt文件夹'
    # new_file_path='保存的新地址'

    eachclass_image = []
    for image in os.listdir(file_path):
        eachclass_image.append(image)
    total = len(eachclass_image)
    random.shuffle(eachclass_image)
    train_images = eachclass_image[0:int(train_rate * total)]  # 注意左闭右开
    val_images = eachclass_image[int(train_rate * total):int((train_rate + val_rate) * total)]  # 注意左闭右开
    test_images = eachclass_image[int((train_rate + val_rate) * total):]

    for image in train_images:
        print(image)
        old_path = file_path + '/' + image
        new_path1 = new_file_path + '/' + 'train' + '/' + 'images'
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path = new_path1 + '/' + image
        # print(new_path)
        shutil.copy(old_path, new_path)
    new_name = os.listdir(new_file_path + '/' + 'train' + '/' + 'images')
    # print(new_name[1][:-4])
    for im in new_name:
        old_xmlpath = xmlpath + '/' + im[:-3] + 'txt'
        new_xmlpath1 = new_file_path + '/' + 'train' + '/' + 'labels'
        if not os.path.exists(new_xmlpath1):
            os.makedirs(new_xmlpath1)
        new_xmlpath = new_xmlpath1 + '/' + im[:-3] + 'txt'
        shutil.copy(old_xmlpath, new_xmlpath)

    for image in val_images:
        old_path = file_path + '/' + image
        new_path1 = new_file_path + '/' + 'val' + '/' + 'images'
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path = new_path1 + '/' + image
        shutil.copy(old_path, new_path)
    new_name = os.listdir(new_file_path + '/' + 'val' + '/' + 'images')
    for im in new_name:
        old_xmlpath = xmlpath + '/' + im[:-3] + 'txt'
        new_xmlpath1 = new_file_path + '/' + 'val' + '/' + 'labels'
        if not os.path.exists(new_xmlpath1):
            os.makedirs(new_xmlpath1)
        new_xmlpath = new_xmlpath1 + '/' + im[:-3] + 'txt'
        shutil.copy(old_xmlpath, new_xmlpath)

    for image in test_images:
        old_path = file_path + '/' + image
        new_path1 = new_file_path + '/' + 'test' + '/' + 'images'
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path = new_path1 + '/' + image
        shutil.copy(old_path, new_path)
    new_name = os.listdir(new_file_path + '/' + 'test' + '/' + 'images')
    for im in new_name:
        old_xmlpath = xmlpath + '/' + im[:-3] + 'txt'
        new_xmlpath1 = new_file_path + '/' + 'test' + '/' + 'labels'
        if not os.path.exists(new_xmlpath1):
            os.makedirs(new_xmlpath1)
        new_xmlpath = new_xmlpath1 + '/' + im[:-3] + 'txt'
        shutil.copy(old_xmlpath, new_xmlpath)


if __name__ == '__main__':
    file_path = "../coco128/images"
    xmlpath = '../coco128/labels'
    new_file_path = "../coco128_split"
    split_data(file_path, new_file_path, train_rate=0.78, val_rate=0.2, test_rate=0.02)


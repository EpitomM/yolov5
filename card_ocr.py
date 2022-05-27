import cv2
from aip import AipOcr

""" 你的 APPID AK SK  图2的内容"""
APP_ID = '26263169'
API_KEY = 'umpullFjLG6hUv04A0kywBD2'
SECRET_KEY = 'B5lvMkE3U4luSFvCA5kVgPLbNhngjm9G'

client = AipOcr(APP_ID, API_KEY, SECRET_KEY)

# fname = '/home/work/DATA_DIR2/disk2/repository/FeedImg/feed/card_detection_data_final_edition/jinritoutiao_feed_card/train/images/abasi_oppoa57_jinritoutiao_20211119_b37_411.jpg'

""" 读取图片 """
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

# image = get_file_content(fname)

# """ 调用通用文字识别, 图片参数为本地图片 """
# results = client.general(image)["words_result"]  # 还可以使用身份证驾驶证模板，直接得到字典对应所需字段

# img = cv2.imread(fname)
# for result in results:
#     text = result["words"]
#     location = result["location"]

#     print(text)
#     # 画矩形框
#     cv2.rectangle(img, (location["left"],location["top"]), (location["left"]+location["width"],location["top"]+location["height"]), (0,255,0), 2)

# cv2.imwrite("ocr_result.jpg", img)


def card_ocr(img_name):
    image = get_file_content(img_name)
    results = client.general(image)["words_result"]  # 还可以使用身份证驾驶证模板，直接得到字典对应所需字段
    return results

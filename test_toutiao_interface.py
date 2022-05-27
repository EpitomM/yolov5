import cv2

if __name__ == '__main__':
    image_path = 'card_detection_data_final_edition/jinritoutiao_feed/picture/22_jinritoutiao_20211110_b27_1.jpg'
    image = cv2.imread(image_path)
    print('before image:', image.shape)
    out = image.transpose((2, 1, 0))[::-1]
    print('after image:', out.shape)
    # BGR
    # RGB
    # img_array = np.fromstring(img_b64decode,np.uint8) # 转换np序列
    # image=cv2.imdecode(img_array,cv2.COLOR_BGR2RGB)  # 转换Opencv格式
    # print('before image.shape:', image.shape)   # (1520, 720, 3)
    ori_image = image
    h = image.shape[0]
    w = image.shape[1]
    hf = 480
    wf = 640
    hn = hf
    wn = int(w / h * hn)

    wp = int((wf - wn) / 2)
    wp_l = wp
    wp_r = wp
    if wp * 2 + wn < wf:
        wp_l = wp + 1
    if wp_l + wp_r + wn != wf:
        print(image.shape, image_path)
    image = cv2.resize(image, (wn, hn))
    print('image.shape:', image.shape)  # (480, 227, 3)
    print('left=' + str(wp_l) + 'right=' + str(wp_r))
    border = cv2.copyMakeBorder(
        image,
        top=0,
        bottom=0,
        left=wp_l,
        right=wp_r,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )
    # return ori_image, border, wp_l, h / hn
import cv2
import os
import numpy as np

img_path = r'G:\Dataset\MoNuSeg\MoNuSeg/img/'
mask_path = r'G:\Dataset\MoNuSeg\MoNuSeg/mask/'
edge_path = r'G:\Dataset\MoNuSeg\MoNuSeg/edge/'
save_img = r'G:\Dataset\MoNuSeg\MoNuSeg\dataAug/img/'
save_mask = r'G:\Dataset\MoNuSeg\MoNuSeg\dataAug/mask/'
save_edge = r'G:\Dataset\MoNuSeg\MoNuSeg\dataAug/edge/'
num = 0


def save_IM(img, mask, edge, name):
    global num
    for i in range(4):
        cv2.imwrite(save_img + str(num) + name, img)
        cv2.imwrite(save_mask + str(num) + name, mask, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        cv2.imwrite(save_edge + str(num) + name, edge, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        num += 1
        img = np.rot90(img, -1)
        mask = np.rot90(mask, -1)
        edge = np.rot90(edge, -1)


if __name__ == '__main__':
    filelist = os.listdir(mask_path)
    # 设置卷积核
    kernel = np.ones((3, 3), np.uint8)
    for file in filelist:
        print(file)
        img = cv2.imread(img_path + file)
        mask = cv2.imread(mask_path + file, cv2.IMREAD_GRAYSCALE)
        edge = cv2.imread(edge_path + file, cv2.IMREAD_GRAYSCALE)

        # 图像膨胀处理,两种都尝试，哪种好用哪种
        edge = cv2.dilate(edge, kernel)

        img_1 = cv2.flip(img, 1)
        mask_1 = cv2.flip(mask, 1)
        edge_1 = cv2.flip(edge, 1)
        save_IM(img_1, mask_1, edge_1, file)
        save_IM(img, mask, edge, file)
        num = 0
        # img90 = cv2.flip(trans_img, 0)

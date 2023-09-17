import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image


def pl_show(img):
    plt.imshow(img)
    plt.show()


def dilate_mask(mask_in, kernel_size):
    # 定义膨胀的核（可以是任何形状，这里使用正方形核）
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 进行膨胀操作
    dilated_mask = cv2.dilate(mask_in, kernel, iterations=1)
    dilated_mask[dilated_mask != 0] = 255
    return dilated_mask


mask_file = r'G:\Dataset\MoNuSeg\MoNuSeg\test\pre_mask/TCGA-44-2665-01B-06-BS6.png'
edge_file = r'G:\Dataset\MoNuSeg\MoNuSeg\test\pre_edge/TCGA-44-2665-01B-06-BS6.png'
img_file = r'G:\Dataset\MoNuSeg\MoNuSeg\test\img/TCGA-44-2665-01B-06-BS6.png'
if __name__ == '__main__':
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    edge = cv2.imread(edge_file, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(img_file)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask_ins = mask.copy()
    mask_ins[edge != 0] = 0
    # pl_show(mask)
    # pl_show(edge)
    # pl_show(mask_ins)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_ins, connectivity=8)

    # pl_show(labels)
    print(num_labels)
    for i in range(10):
        for mask_id in range(num_labels):
            id_mask = np.zeros((labels.shape[0], labels.shape[1]))
            id_mask[labels == mask_id] = mask_id
            id_mask = dilate_mask(id_mask, 3)
            labels[(id_mask != 0) & (mask != 0) & (labels == 0)] = mask_id
    # pl_show(labels)

    # 创建一个彩色图像以显示标记的连通区域
    colored_labels = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)

    # 随机生成颜色
    colors = []
    for i in range(num_labels):
        colors.append((np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))

    # 绘制标记的连通区域
    for row in range(labels.shape[0]):
        for col in range(labels.shape[1]):
            if labels[row, col] != 0:
                colored_labels[row, col] = colors[labels[row, col]]
    pl_show(mask)
    cv2.imwrite('mask.png', colored_labels)
    pl_show(colored_labels)



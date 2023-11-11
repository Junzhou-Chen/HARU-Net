import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image


file_path = r'G:\Dataset\kumar'
save_file = os.path.join(file_path, 'test', 'ins_mask/')
mask_file = os.path.join(file_path, 'test', 'pre_mask/')
edge_file = os.path.join(file_path, 'test', 'pre_edge/')

if not os.path.exists(save_file):
    os.mkdir(save_file)

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




if __name__ == '__main__':
    filelist = os.listdir(mask_file)
    for file in filelist:
        print(file)
        mask = cv2.imread(mask_file + file, cv2.IMREAD_GRAYSCALE)
        edge = cv2.imread(edge_file + file, cv2.IMREAD_GRAYSCALE)

        mask_ins = mask.copy()
        mask_ins[edge != 0] = 0
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_ins, connectivity=8)

        print('细胞核个数:', num_labels)
        for i in range(6):
            for mask_id in range(num_labels):
                id_mask = np.zeros((labels.shape[0], labels.shape[1]))
                id_mask[labels == mask_id] = mask_id
                id_mask = dilate_mask(id_mask, 3)
                labels[(id_mask != 0) & (mask != 0) & (labels == 0)] = mask_id
        # pl_show(labels)
        img = Image.fromarray(labels)

        # 保存图像
        img.save(save_file + file[:-4] + ".tiff")
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
        cv2.imwrite(file, colored_labels)
        pl_show(colored_labels)

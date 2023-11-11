import cv2
import os
import numpy as np

kumar_path = r'G:\Dataset\kumar'

def save_IM(img, mask, edge, name):
    global num
    for i in range(4):
        save_name = str(num) + name
        cv2.imwrite(os.path.join(aug_save_path, 'img', save_name), img)
        cv2.imwrite(os.path.join(aug_save_path, 'mask', save_name), mask, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        cv2.imwrite(os.path.join(aug_save_path, 'edge', save_name), edge, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        num += 1
        img = np.rot90(img, -1)
        mask = np.rot90(mask, -1)
        edge = np.rot90(edge, -1)


if __name__ == '__main__':
    train_path = os.path.join(kumar_path, 'train')
    test_path = os.path.join(kumar_path, 'test')
    kernel = np.ones((3, 3), np.uint8)
    num = 0
    for aug_path in [train_path, test_path]:
        aug_save_path = os.path.join(aug_path, 'aug')
        if not os.path.exists(aug_save_path):
            os.mkdir(aug_save_path)
            os.mkdir(os.path.join(aug_save_path, 'img'))
            os.mkdir(os.path.join(aug_save_path, 'edge'))
            os.mkdir(os.path.join(aug_save_path, 'mask'))
        files_path = os.listdir(os.path.join(aug_path, 'Images'))
        for file in files_path:
            print(file)
            img = cv2.imread(os.path.join(aug_path, 'Images', file))
            mask = cv2.imread(os.path.join(aug_path, 'mask', file), cv2.IMREAD_GRAYSCALE)
            edge = cv2.imread(os.path.join(aug_path, 'edge', file), cv2.IMREAD_GRAYSCALE)

            # 图像膨胀处理,两种都尝试，哪种好用哪种
            edge = cv2.dilate(edge, kernel)

            img_1 = cv2.flip(img, 1)
            mask_1 = cv2.flip(mask, 1)
            edge_1 = cv2.flip(edge, 1)

            save_IM(img_1, mask_1, edge_1, file)
            save_IM(img, mask, edge, file)
            num = 0

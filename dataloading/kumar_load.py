import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def pl_show(img):
    plt.imshow(img)
    plt.show()


def cv_show(img):
    cv2.imshow("image", img)
    cv2.waitKey(0)

kumar_path = r'G:\Dataset\kumar'

if __name__ == '__main__':
    train_path = os.path.join(kumar_path, 'train')
    test_path = os.path.join(kumar_path, 'test')

    child_path = [train_path, test_path]
    for trans_path in child_path:
        edge_path = os.path.join(trans_path, 'edge')
        mask_path = os.path.join(trans_path, 'mask')
        npy_path = os.path.join(trans_path, 'Masks')
        if not os.path.exists(edge_path):
            os.mkdir(edge_path)
        if not os.path.exists(mask_path):
            os.mkdir(mask_path)
        npys_path = os.listdir(npy_path)
        for pro_path in npys_path:
            npy = np.load(os.path.join(npy_path, pro_path))
            edge = cv2.Canny(np.uint8(npy), 0, 1)
            edge[edge > 0] = 255
            mask = npy.copy()
            mask[mask != 0] = 255
            cv2.imwrite(os.path.join(mask_path, pro_path[:-3]) + 'png', mask, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
            cv2.imwrite(os.path.join(edge_path, pro_path[:-3]) + 'png', edge, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
            print(pro_path)

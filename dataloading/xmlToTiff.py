# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from PIL import Image

def pl_show(img_s):
    plt.imshow(img_s)
    plt.show()


def xml2mask(filename, dir):
    mask = np.zeros([1000, 1000], dtype=np.uint16)

    xml = str(dir + "\\" + file)
    tree = ET.parse(xml)
    root = tree.getroot()
    regions = root.findall('Annotation/Regions/Region')
    color = 1
    for region in regions:
        mask_copy = np.zeros([1000, 1000], dtype=np.uint8)
        points = []
        for point in region.findall('Vertices/Vertex'):
            x = float(point.attrib['X'])
            y = float(point.attrib['Y'])
            points.append([x, y])

        pts = np.asarray([points], dtype=np.int32)
        cv2.fillPoly(img=mask_copy, pts=pts, color=255)
        mask[mask_copy != 0] = color
        color += 1
    print(color)
    pl_show(mask)
    img = Image.fromarray(mask)

    # 保存图像
    img.save(r"G:\Dataset\MoNuSeg\MoNuSegTrainingData\matt/" + os.path.splitext(filename)[0] + ".tiff")
    # cv2.imwrite(r"G:\Dataset\MoNuSeg\MoNuSegTrainingData\matt/" + os.path.splitext(filename)[0] + ".tiff", mask)


dir = r"G:\Dataset\MoNuSeg\MoNuSegTrainingData\label/"
files = os.listdir(dir)
i = 0
for file in files:
    xml2mask(file, dir)
    i += 1
    print('已完成{0}幅图像!'.format(i))
print("全部完成!")

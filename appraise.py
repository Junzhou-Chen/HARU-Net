import numpy as np
import cv2
import os
import argparse
# true dir
val_path = r'G:\Dataset\MoNuSeg\MoNuSeg\test\mask/'
# predict dir
out_path = r'G:\Dataset\MoNuSeg\MoNuSeg\test\pre_mask//'


# Dice
def Dice(y_true, y_pred):
    intersection = np.sum(np.logical_and(y_true, y_pred))
    all_sum = np.sum(np.array(y_true, dtype= bool)) + np.sum(np.array(y_pred, dtype= bool))
    return (2. * intersection) / all_sum if all_sum else 1


# IoU
def IoU(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou = intersection.sum() / union.sum()
    return iou if union.sum() else 1


# Pixel accuracy
def PA(y_true, y_pred):
    positive = np.sum(np.logical_and(y_true, y_pred))
    negative = np.sum(np.logical_and(np.logical_not(y_true), np.logical_not(y_pred)))
    return (positive + negative) / (y_true.shape[0] * y_true.shape[1])


# PQ
def PQ(y_true, y_pred):
    # True Positive(TP) and False Positive(FP)
    TP = np.logical_and(y_true, y_pred).sum()
    FP = np.array(y_pred, dtype= bool).sum() - TP
    # True Negative(TN)
    # TN = np.logical_and(np.logical_not(y_true), np.logical_not(y_pred)).sum()
    # False Negative(FN)
    FN = np.array(y_true, dtype= bool).sum() - TP
    # get PQ
    numerator = TP
    denominator = TP + FP / 2 + FN / 2
    return numerator / denominator if denominator else 1


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate the predicted masks')
    parser.add_argument('--input', '-i', default='', metavar='INPUT', help='Filenames of input images')
    parser.add_argument('--output', '-o', default='', metavar='OUTPUT', help='Filenames of output images')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.input != '':
        val_path = args.input
    if args.output != '':
        out_path = args.output
    file_list = os.listdir(out_path)
    dice, pq, iou, pa = [], [], [], []
    print("Begin appraise")
    for file in file_list:
        true_mask = cv2.imread(val_path + file, cv2.IMREAD_GRAYSCALE)
        # gt[gt != 0] = 1
        pred_mask = cv2.imread(out_path + file, cv2.IMREAD_GRAYSCALE)
        # pred[pred != 0] = 1
        dice.append(Dice(true_mask, pred_mask))
        pq.append(PQ(true_mask, pred_mask))
        iou.append(IoU(true_mask, pred_mask))
        pa.append(PA(true_mask, pred_mask))
    print(f"Dice: {np.mean(dice):.4f}")
    print(f"PQ: {np.mean(pq):.4f}")
    print(f"IoU: {np.mean(iou):.4f}")
    print(f"PA: {np.mean(pa):.4f}")

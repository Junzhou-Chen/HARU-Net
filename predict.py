import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from network import HARUNet
from utils.utils import plot_img_and_mask

file_path = r'G:\Dataset\kumar'
in_files = os.path.join(file_path, 'test', 'Images/')
out_masks = os.path.join(file_path, 'test', 'pre_mask/')
out_edges = os.path.join(file_path, 'test', 'pre_edge/')

if not os.path.exists(out_edges):
    os.mkdir(out_edges)
if not os.path.exists(out_masks):
    os.mkdir(out_masks)
# in_files = r'G:\Dataset\MoNuSeg\MoNuSeg\test\img/'
# out_masks = r'G:\Dataset\MoNuSeg\MoNuSeg\test\pre_mask/'
# out_edges = r'G:\Dataset\MoNuSeg\MoNuSeg\test\pre_edge/'


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    net.to(device=device)
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        mask, edge = net(img)
        mask = F.interpolate(mask, (full_img.size[1], full_img.size[0]), mode='bilinear')
        edge = F.interpolate(edge, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = mask.argmax(dim=1)
            edge = edge.argmax(dim=1)
        else:
            mask = torch.sigmoid(mask) > out_threshold
            edge = torch.sigmoid(edge) > out_threshold

    return mask[0].long().squeeze().cpu().numpy(), edge[0].long().squeeze().cpu().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default=r'./checkpoint_epoch63.pth',
                        metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', default='', metavar='INPUT', help='Filenames of input images')
    parser.add_argument('--output_mask', '-om', default='', metavar='OUTPUT', help='Filenames of output masks')
    parser.add_argument('--output_edge', '-oe', default='', metavar='OUTPUT', help='Filenames of output edges')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=True, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if args.input != '':
        in_files = args.input
    if args.output_mask != '':
        out_masks = args.output
    if args.output_edge != '':
        out_edges = args.output

    net = HARUNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')
    file_list = os.listdir(in_files)
    for filename in file_list:

        logging.info(f'Predicting image {filename} ...')
        img = Image.open(in_files + filename)
        mask, edge = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_filename = out_masks + filename
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

            out_filename = out_edges + filename
            result = mask_to_image(edge, mask_values)
            result.save(out_filename)
            logging.info(f'Edge saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)


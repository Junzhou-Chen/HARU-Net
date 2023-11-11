import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
# import wandb
from utils import evaluate
# import utils.evaluate as evaluate
from network import HARUNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

kumar_path = r'G:\Dataset\kumar'

# img path
dir_img = os.path.join(kumar_path, 'train', 'aug', 'img')
# mask path
dir_mask = os.path.join(kumar_path, 'train', 'aug', 'mask')
# edge path
dir_edge = os.path.join(kumar_path, 'train', 'aug', 'edge')
# img path
test_img = os.path.join(kumar_path, 'test', 'aug', 'img')
# mask path
test_mask = os.path.join(kumar_path, 'test', 'aug', 'mask')
# mask path
test_edge = os.path.join(kumar_path, 'test', 'aug', 'edge')
# .pth save path
dir_checkpoint = Path('./checkpoints/')


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    try:
        train_set = CarvanaDataset(dir_img, dir_mask, dir_edge, img_scale)
        val_set = CarvanaDataset(test_img, test_mask, test_edge, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        train_set = BasicDataset(dir_img, dir_mask, dir_edge, img_scale)
        val_set = BasicDataset(test_img, test_mask, test_edge, img_scale)

    # 2. Split into train / validation partitions
    n_val = len(val_set)
    n_train = len(train_set)
    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    # (Initialize logging)
    # experiment = wandb.init(project='HARUNet', resume='allow', anonymous='must')
    # experiment.config.update(
    #     dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
    #          val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    # )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=50) # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks, true_edges = batch['image'], batch['mask'], batch['edge']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                true_edges = true_edges.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred, edges_pred = model(images)
                    if model.n_classes == 1:
                        losses = [criterion(i(1), true_masks.float()) for i in masks_pred]
                        loss = sum(losses)
                        # loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        losses = [dice_loss(F.sigmoid(i.squeeze(1)), true_masks.float(), multiclass=False) for i in
                                  masks_pred]
                        loss = sum(losses)
                        # loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        losses1 = [criterion(i, true_masks) for i in masks_pred]
                        loss1 = sum(losses1)
                        # loss = criterion(masks_pred, true_masks)
                        losses1 = [dice_loss(
                            F.softmax(i, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        ) for i in masks_pred]
                        loss1 += sum(losses1)
                        losses2 = [criterion(i, true_edges) for i in edges_pred]
                        loss2 = sum(losses2)
                        # Dice loss
                        losses2 = [dice_loss(
                            F.softmax(i, dim=1).float(),
                            F.one_hot(true_edges, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        ) for i in edges_pred]
                        loss2 += sum(losses2)
                        loss = loss1 + loss2

                # images = images[0]
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                # change
                masks_pred = masks_pred[0]
                edges_pred = edges_pred[0]
                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                # experiment.log({
                #     'train loss': loss.item(),
                #     'step': global_step,
                #     'epoch': epoch
                # })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        # histograms = {}
                        # for tag, value in model.named_parameters():
                        #     tag = tag.replace('/', '.')
                        #     if not (torch.isinf(value) | torch.isnan(value)).any():
                        #         histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        #     if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                        #         histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score, edge_score = evaluate.evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)
                        # print("success")
                        logging.info('Validation Dice mask score: {}'.format(val_score))
                        logging.info('Validation Dice edge score: {}'.format(edge_score))
                        # try:
                        #     experiment.log({
                        #         'learning rate': optimizer.param_groups[0]['lr'],
                        #         'validation Dice': val_score,
                        #         'edge Dice': edge_score,
                        #         # 'images': wandb.Image(images[0].cpu()),
                        #         # 'masks': {
                        #         #     'true': wandb.Image(true_masks[0].float().cpu()),
                        #         #     'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                        #         # },
                        #         'step': global_step,
                        #         'epoch': epoch,
                        #         **histograms
                        #     })
                        # except:
                        #     pass

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = train_set.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the HARUNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=2, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.3, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = HARUNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )

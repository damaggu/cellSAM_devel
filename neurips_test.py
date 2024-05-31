import os
import cv2
import torch
import argparse

import numpy as np
import dask.array as da
import imageio.v3 as iio
import lightning.pytorch as pl
import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage.segmentation import relabel_sequential

from cellSAM.utils import relabel_mask
from cellSAM.wsi import segment_chunk, segment_wsi
from cellSAM.model import get_local_model, get_model



def mask_outline(mask):
    """got this from Ross. the boundaries end up being slightly thinner than skimage"""
    outline = np.zeros_like(mask, dtype=np.uint8)
    outline[:, 1:][mask[:, :-1] != mask[:, 1:]] = 1
    outline[:-1, :][mask[:-1, :] != mask[1:, :]] = 1
    return outline


def add_white_borders(img, mask, color=None):
    if color is None:
        color = [1.0, 1.0, 1.0]
    assert img.shape[:2] == mask.shape
    assert img.shape[2] == 3
    assert len(img.shape) == 3

    boundary = mask_outline(mask)
    img = np.array(img)  # copy
    r, c = np.where(np.isclose(1.0, boundary))
    img[r, c] = color
    return img


if __name__ == "__main__":
    # fix the seed
    pl.seed_everything(42)

    # argsparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_chunks", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--bbox_threshold", type=float, default=0.4)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    path_to_all_imgs = '/data/user-data/rdilip/cellSAM/dataset/val_tuning/neurips'
    all_images = os.listdir(path_to_all_imgs)
    all_images = [img for img in all_images if img.endswith('.X.npy')]

    if bool(args.debug):
        # all_images = all_images[:8]
        # all_images = ['cell_00041.b0.X.npy']
        # all_images = ['cell_00044.b0.X.npy']
        all_images = ['cell_00001.b0.X.npy']
    results_path = './results1024/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    chunks = args.num_chunks
    chunk_size = len(all_images) // chunks

    start = args.chunk * chunk_size
    end = (args.chunk + 1) * chunk_size

    if args.chunk == chunks - 1:
        end = len(all_images)

    if args.model_path is not None:
        modelpath = args.model_path
        model = get_local_model(modelpath)
        model.bbox_threshold = args.bbox_threshold
        if torch.cuda.is_available():
            model = model.to(device)
    else:
        model = None

    if start == end == 0:
        start = 0
        end = 1

    print(f"Processing images from {start} to {end}")
    for img in tqdm(all_images[start:end]):
        # try:
        #     wsi = iio.imread(os.path.join(path_to_all_imgs, img))[:, :, [0, 2, 1]]
        # except:
        #     wsi = cv2.imread(os.path.join(path_to_all_imgs, img))
        #     wsi = cv2.cvtColor(wsi, cv2.COLOR_BGR2RGB)
        wsi = np.load(os.path.join(path_to_all_imgs, img)).transpose((1,2,0))
        gt_mask = np.load(os.path.join(path_to_all_imgs, img.replace('.X.npy', '.y.npy')))

        # if args.debug:
        #     wsi = wsi[:512, :512]
        input = da.from_array(wsi, chunks=args.tile_size)
        # mask = segment_chunk(input, model=model, device='cuda:0', normalize=True)[0]
        # mask = segment_chunk(input, model=model, device=device, normalize=True)[0]
        labels = segment_wsi(input, 100, 100, 0.4, normalize=False, model=model, device=device).compute()
        labels = relabel_mask(relabel_sequential(labels)[0])
        labels = np.expand_dims(labels, axis=0)

        # save the results
        iio.imwrite(os.path.join(results_path, img.split('.')[0] + '.tiff'), labels)
        plt.imshow(labels[0])
        plt.title(img.split('.')[0])
        # save as
        plt.savefig(os.path.join(results_path, img.split('.')[0] + '_inspection.png'))
        plt.close()
        print(f"Processed {img}")

    print("Done.")

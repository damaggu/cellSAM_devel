import os
import cv2
import json
import torch
import joblib
import skimage
import argparse

import numpy as np
import pickle as pkl
import dask.array as da
import imageio.v3 as iio
import lightning.pytorch as pl
import matplotlib.pyplot as plt

from glob import glob

from skimage.exposure import adjust_gamma, equalize_adapthist, adjust_log
from tqdm import tqdm
from scipy.signal import find_peaks
from skimage.segmentation import relabel_sequential

from cellSAM.utils import relabel_mask, f1_score, fill_holes_and_remove_small_masks
from cellSAM.wsi import segment_chunk, segment_wsi
from cellSAM.model import get_local_model, get_model, segment_cellular_image


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


def contains_any_number(s, numbers):
    return any(num in s for num in numbers)


def featurizer(clf, img):
    """ img is H, W, C """
    data = img[..., 1:].ravel()
    counts, bins = np.histogram(data, bins=40, density=True)
    n_peaks = len(find_peaks(counts, height=0.5)[0])

    feature = np.array([np.mean(data), np.std(data), n_peaks, *img.shape[:-1]])
    is_big_bc = clf.predict(feature.reshape(1, -1))

    return is_big_bc


def is_low_contrast_clahe(image, lower_threshold=0.04, upper_threshold=0.05, lower_mean=0.15, upper_mean=0.25):
    # to greyscale
    # image = image.mean(axis=2)
    # min-max scaling
    cp = equalize_adapthist(image, kernel_size=256)
    diff = np.abs(image - cp)
    # diff>0
    diff = diff[diff > 0]
    mean_diff = np.median(diff)
    mean_std = np.std(diff)
    print(f"Mean diff: {mean_diff}")
    print(np.mean(cp))
    islowcontrast = lower_threshold < mean_diff < upper_threshold
    return [islowcontrast, mean_diff, mean_std]


def resize_results(save_dir, img_dir, results_dir, gt_dir):
    all_images = os.listdir(img_dir)
    all_images = sorted([img for img in all_images if img.endswith('.X.npy')])

    os.makedirs(save_dir, exist_ok=True)
    for img in all_images:
        gt_path = os.path.join(gt_dir, img.split('.')[0] + '_label.tiff')
        gt_mask = iio.imread(gt_path)

        label_path = os.path.join(results_dir, img.split('.')[0] + '.tiff')
        labels = iio.imread(label_path)

        if gt_mask.shape[0] < 512:
            # remove padding
            padding = 512 - gt_mask.shape[0]
            labels = labels[padding // 2:-padding // 2, :]
        if gt_mask.shape[1] < 512:
            padding = 512 - gt_mask.shape[1]
            labels = labels[:, padding // 2:-padding // 2]

        iio.imwrite(os.path.join(save_dir, img.split('.')[0] + '.tiff'), labels)


def get_median_size(labels):
    sizes = []
    sizes_abs = []
    for mask in np.unique(labels):
        if mask == 0:
            continue
        area = (labels == mask).sum().item()
        # normalizing by area
        sizes.append(area / (labels.shape[0] * labels.shape[1]))
        sizes_abs.append(area)
    sizes = np.array(sizes)
    sizes_abs = np.array(sizes_abs)
    # median size
    median_size = np.median(sizes)
    return median_size, sizes, sizes_abs


if __name__ == "__main__":
    # fix the seed
    pl.seed_everything(42)

    # argsparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_chunks", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--bbox_threshold", type=float, default=0.4)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--results_path", type=str, default='./results1024/')
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--iou_depth", type=int, default=100)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--img_path", type=str, default='./debugdata/images/cell_00849.png')
    parser.add_argument("--plt_gt", type=int, default=0)
    parser.add_argument("--preproc", type=int, default=0)
    parser.add_argument("--use_gt", type=int, default=0)
    parser.add_argument("--verbose", type=int, default=0)

    parser.add_argument("--lower_contrast_threshold", type=float, default=0.025)
    parser.add_argument("--upper_contrast_threshold", type=float, default=0.1)

    parser.add_argument("--medium_cell_threshold", type=float, default=0.002)
    parser.add_argument("--large_cell_threshold", type=float, default=0.015)
    parser.add_argument("--medium_cell_max", type=int, default=60)
    parser.add_argument("--medium_mean_diff_threshold", type=float, default=0.1)
    parser.add_argument("--cells_min_size", type=int, default=500)
    parser.add_argument("--border_size", type=int, default=5)

    # TODO: adaptive tiling, adaptive overlap, adaptive CLAHE

    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    if args.model_path is not None:
        modelpath = args.model_path
        model = get_local_model(modelpath)
        model.bbox_threshold = args.bbox_threshold
        # model.iou_threshold = 0.9
        if torch.cuda.is_available():
            model = model.to(device)
    else:
        model = None

    verbose = True

    img = iio.imread(args.img_path)
    # switch last 2 channels bc nuclear and whoelcell are switched, #TODO: autmatically detect or have input arg like cellpose
    img = img[..., [0, 2, 1]]
    # to float
    img = img.astype(np.float32) / 255.0

    ### some additional processing for low-contrast images -- not strictly necessary
    low_contrast, mean_diff, mean_std = is_low_contrast_clahe(img, lower_threshold=args.lower_contrast_threshold,
                                                              upper_threshold=args.upper_contrast_threshold)
    low_contrast = (low_contrast and img[..., 1].max() == 0) if mean_diff < 0.05 else low_contrast
    if low_contrast:
        clip_limit = 0.01
        kernel_size = 256
        gamma = 2
        if mean_diff > args.lower_contrast_threshold and mean_std < 0.05:
            clip_limit = 0.02
            kernel_size = 384
            gamma = 1.2
            model.bbox_threshold = 0.15
        if mean_diff > 0.065 and mean_std < 0.05:
            clip_limit = 0.05
            model.bbox_threshold = 0.15
        if mean_diff > 0.065 and (0.035 < mean_std < 0.04):
            clip_limit = 0.01
        img = equalize_adapthist(img, kernel_size=kernel_size, clip_limit=clip_limit)
        img = adjust_gamma(img, gamma=gamma)

    inp = da.from_array(img, chunks=256)
    labels = segment_wsi(inp, 200, 200, args.iou_threshold, normalize=False, model=model,
                         device=device, bbox_threshold=args.bbox_threshold).compute()

    median_size, sizes, sizes_abs = get_median_size(labels)

    print(f"Median size: {median_size:.4f}")

    # only if cells are small we to WSI inference
    if median_size < args.medium_cell_threshold:
        doing_wsi = True
        # cells are medium or small -> do WSI
        inp = da.from_array(img, chunks=args.tile_size)
        labels = segment_wsi(inp, args.overlap, args.iou_depth, args.iou_threshold, normalize=False, model=model,
                             device=device, bbox_threshold=args.bbox_threshold).compute()
    else:
        labels = segment_cellular_image(img, model=model, normalize=False, device=device)[0]

    # # labels to individual masks
    # # filter out masks smaller than min size
    masks = []
    for mask in np.unique(labels):
        m_array = (labels == mask).astype(np.int32)
        if mask == 0:
            continue
        # is m_array at the edge?
        if m_array.sum() < args.cells_min_size and m_array[args.border_size:-args.border_size,
                                                   args.border_size:-args.border_size].sum() == 0:
            continue
        masks.append(m_array * mask)
    labels = np.max(masks, axis=0)

    result = relabel_mask(relabel_sequential(labels)[0])

    plt.imshow(result)
    plt.show()

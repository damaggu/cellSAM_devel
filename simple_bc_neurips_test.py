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

from tqdm import tqdm
from scipy.signal import find_peaks
from skimage.segmentation import relabel_sequential

from cellSAM.utils import relabel_mask
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

# Function to check if a string contains any number from the list
def contains_any_number(s, numbers):
    return any(num in s for num in numbers)


def featurizer(clf,img):
    """ img is H, W, C """
    data = img[..., 1:].ravel()
    counts, bins = np.histogram(data, bins=40, density=True)
    n_peaks = len(find_peaks(counts, height=0.5)[0])    
    
    feature = np.array([np.mean(data), np.std(data), n_peaks])
    is_big_bc = clf.predict(feature.reshape(1, -1))

    return is_big_bc


def resize_results(save_dir, img_dir, results_dir, gt_dir):
    all_images = os.listdir(img_dir)
    all_images = sorted([img for img in all_images if img.endswith('.X.npy')])

    os.makedirs(save_dir, exist_ok=True)
    for img in all_images:
        gt_path =  os.path.join(gt_dir, img.split('.')[0] + '_label.tiff')
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
    parser.add_argument("--data_path", type=str, default='./neurips/')
    parser.add_argument("--plt_gt", type=int, default=0)
    parser.add_argument("--preproc", type=int, default=0)
    parser.add_argument("--use_gt", type=int, default=0)

    # TODO: adaptive tiling, adaptive overlap, adaptive CLAHE

    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    path_to_all_imgs = args.data_path
    all_images = os.listdir(path_to_all_imgs)
    all_images = sorted([img for img in all_images if img.endswith('.X.npy')])

    results_inferences_preproc_nowsi = os.path.join(args.results_path, 'preproc_nowsi_inferences')
    results_inferences_nopreproc_nowsi = os.path.join(args.results_path, 'nopreproc_nowsi_inferences')
    results_inferences_wsi = os.path.join(args.results_path, 'wsi_inferences')
    
    os.makedirs(results_inferences_preproc_nowsi, exist_ok=True)
    os.makedirs(results_inferences_nopreproc_nowsi, exist_ok=True)
    os.makedirs(results_inferences_wsi, exist_ok=True)
    

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

    verbose = True
    plt_gt = bool(args.plt_gt)
    use_preproc = bool(args.preproc)
    use_gt = bool(args.use_gt)

    if plt_gt:
        assert 'tuning' in path_to_all_imgs, "Ground truth only available for tuning set"

    print(f"Processing images from {start} to {end}")

    # load blood cell labels
    data = set(pkl.load(open('bloodcell_pths_tuning.pkl', 'rb')))
    data = {filename.split('.')[0] for filename in data}
    # filter images based on blood cell data
    all_images = [img for img in all_images if img.split('.')[0] in data]


    for img in tqdm(all_images[start:end]):
        print(f"Starting to process {img}")

        base = img.split('.')[0]
        # try:
        #     wsi = iio.imread(os.path.join(path_to_all_imgs, img))[:, :, [0, 2, 1]]
        # except:
        #     wsi = cv2.imread(os.path.join(path_to_all_imgs, img))
        #     wsi = cv2.cvtColor(wsi, cv2.COLOR_BGR2RGB)
        img_pth = os.path.join(path_to_all_imgs, img)
        wsi = np.load(img_pth).transpose((1, 2, 0))
        
        
        # loading gt mask
        gt_path = "./evals/tuning/labels/" + img.split('.')[0] + '_label.tiff'
        gt_mask = iio.imread(gt_path)



        ## compute various labels for image

        ## pre proc - no wsi
        pl, ph = np.percentile(wsi[...,1], (0, 30))
        inp_new = np.zeros_like(wsi)
        inp_new[...,-1] = skimage.exposure.rescale_intensity(wsi[...,1], in_range=(pl, ph))
        preproc_nowsi_labels = segment_cellular_image(inp_new, model=model, normalize=False, device=device)[0]
        preproc_nowsi_labels = relabel_mask(relabel_sequential(preproc_nowsi_labels)[0])


        ## NO pre proc - no wsi 
        nopreproc_nowsi_labels = segment_cellular_image(wsi, model=model, normalize=False, device=device)[0]
        nopreproc_nowsi_labels = relabel_mask(relabel_sequential(nopreproc_nowsi_labels)[0])

        ## WSI
        input = da.from_array(wsi, chunks=args.tile_size)
        try:
            wsi_labels = segment_wsi(input, args.overlap, args.iou_depth, args.iou_threshold, normalize=False, model=model,
                                device=device, bbox_threshold=args.bbox_threshold).compute()
            if len(np.unique(wsi_labels)) < 5:
                print('rerunning with normalize=True')
                wsi_labels = segment_wsi(input, args.overlap, args.iou_depth, args.iou_threshold, normalize=True, model=model,
                                    device=device, bbox_threshold=args.bbox_threshold).compute()
        except ZeroDivisionError:
            print('except')
            wsi_labels = segment_wsi(input, args.overlap, args.iou_depth, args.iou_threshold, normalize=True, model=model,
                                device=device, bbox_threshold=args.bbox_threshold).compute()
            
        wsi_labels = relabel_mask(relabel_sequential(wsi_labels)[0])

        # fixing/reversing padding
        if gt_mask.shape[0] < 512:
            # remove padding
            padding = 512 - gt_mask.shape[0]
            preproc_nowsi_labels = preproc_nowsi_labels[padding // 2:-padding // 2, :]
            nopreproc_nowsi_labels = nopreproc_nowsi_labels[padding // 2:-padding // 2, :]
            wsi_labels = wsi_labels[padding // 2:-padding // 2, :]
        if gt_mask.shape[1] < 512:
            padding = 512 - gt_mask.shape[1]
            preproc_nowsi_labels = preproc_nowsi_labels[:, padding // 2:-padding // 2]
            nopreproc_nowsi_labels = nopreproc_nowsi_labels[:, padding // 2:-padding // 2]
            wsi_labels = wsi_labels[:, padding // 2:-padding // 2]

        # save the results
        iio.imwrite(os.path.join(results_inferences_preproc_nowsi, img.split('.')[0] + '.tiff'), preproc_nowsi_labels)
        iio.imwrite(os.path.join(results_inferences_nopreproc_nowsi, img.split('.')[0] + '.tiff'), nopreproc_nowsi_labels)
        iio.imwrite(os.path.join(results_inferences_wsi,img.split('.')[0] + '.tiff'), wsi_labels)


    print("Done.")

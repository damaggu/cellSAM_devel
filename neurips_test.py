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


# def featurizer(clf,img):
#     """ img is H, W, C """
#     data = img[..., 1:].ravel()
#     counts, bins = np.histogram(data, bins=40, density=True)
#     n_peaks = len(find_peaks(counts, height=0.5)[0])    
    
#     feature = np.array([np.mean(data), np.std(data), n_peaks])
#     is_big_bc = clf.predict(feature.reshape(1, -1))

#     return is_big_bc

def featurizer(clf, img):
    """ img is H, W, C """
    data = img[..., 1:].ravel()
    counts, bins = np.histogram(data, bins=40, density=True)
    n_peaks = len(find_peaks(counts, height=0.5)[0])    
    
    feature = np.array([np.mean(data), np.std(data), n_peaks, *img.shape[:-1]])
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

    if bool(args.debug):
        all_images = all_images[:28]
        # all_images = ['cell_00041.b0.X.npy']
        # all_images = ['cell_00028.b0.X.npy']

        # all_images = ['cell_00044.b0.X.npy']
        # all_images = ['cell_00023.b0.X.npy']
        # all_images = ['cell_00027.b0.X.npy']
        # all_images = ['cell_00028.b0.X.npy']
        # 'TestHidden_043'
        # all_images = ['TestHidden_012.b0.X.npy']
        # all_images = ['TestHidden_020.b0.X.npy']
        # all_images = ['TestHidden_043.b0.X.npy']
        # all_images = ['cell_00032.b0.X.npy']
    else:
        import matplotlib

        matplotlib.use('Agg')

    results_inferences = os.path.join(args.results_path, 'inferences')
    results_inspections = os.path.join(args.results_path, 'inspections')
    os.makedirs(results_inferences, exist_ok=True)
    os.makedirs(results_inspections, exist_ok=True)
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

    # if plt_gt:
    #     assert 'tuning' in path_to_all_imgs, "Ground truth only available for tuning set"

    print(f"Processing images from {start} to {end}")

    if use_gt:
        if 'tuning' in path_to_all_imgs:
            data = set(pkl.load(open('bloodcell_pths_tuning.pkl', 'rb')))
            # remove extentions
            data = {filename.split('.')[0] for filename in data}
        elif 'hidden' in path_to_all_imgs:
            data = set(pkl.load(open('/home/rdilip/cellSAM_debug_space/bloodcell_pths.pkl', 'rb')))
            # remove extentions
            data = {filename.split('.')[0] for filename in data}
        else:
            raise ValueError("Unknown dataset")

    # import bloodcell classifier
    # clf = joblib.load('./saved_models/xgboost_classifier.pkl')
    clf = joblib.load('./saved_models/new_classifier.pkl')

    wsi_imgs_flagged = []
    wsi_imgs_reg = []
    no_wsi_imgs = []
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
        if plt_gt:
            # load from rohits path (.npy files)
            # gt_mask = np.load(os.path.join(path_to_all_imgs, img.replace('.X.npy', '.y.npy'))).transpose((1, 2, 0))
            
            # load from markus path (.tiff files)
            gt_path = "./evals/tuning/labels/" + img.split('.')[0] + '_label.tiff'
            gt_mask = iio.imread(gt_path)


        if use_gt:
            if base in data:
                use_wsi = False
            else:
                use_wsi = True
        else:
            # New Classiefer Stuff
            use_wsi = featurizer(clf, wsi)[0] == 0

        # if args.debug:
        #     wsi = wsi[:512, :512]
        if not use_wsi:
            if use_preproc:
                # now we do percentiling
                pl, ph = np.percentile(wsi[...,1], (0, 30))
                inp_new = np.zeros_like(wsi)
                inp_new[...,-1] = skimage.exposure.rescale_intensity(wsi[...,1], in_range=(pl, ph))
                labels = segment_cellular_image(inp_new, model=model, normalize=False, device=device)[0]
            else:
                labels = segment_cellular_image(wsi, model=model, normalize=False, device=device)[0]

            if len(np.unique(labels)) == 1:
                wsi_imgs_flagged.append(img)
                use_wsi = True
            else:
                no_wsi_imgs.append(img)
                
        if use_wsi:
            wsi_imgs_reg.append(img)
            input = da.from_array(wsi, chunks=args.tile_size)

            ### rerunning with different preprocessing if no cells are found
            try:
                labels = segment_wsi(input, args.overlap, args.iou_depth, args.iou_threshold, normalize=False, model=model,
                                    device=device, bbox_threshold=args.bbox_threshold).compute()
                if len(np.unique(labels)) < 5:
                    labels = segment_wsi(input, args.overlap, args.iou_depth, args.iou_threshold, normalize=True,
                                        model=model,
                                        device=device, bbox_threshold=args.bbox_threshold).compute()
            except ZeroDivisionError:
                print('except')
                labels = segment_wsi(input, args.overlap, args.iou_depth, args.iou_threshold, normalize=True, model=model,
                                    device=device, bbox_threshold=args.bbox_threshold).compute()
        labels = relabel_mask(relabel_sequential(labels)[0])

        ### reshaping based on gt label
        if plt_gt:
            # fixing/reversing padding
            if gt_mask.shape[0] < 512:
                # remove padding
                padding = 512 - gt_mask.shape[0]
                labels = labels[padding // 2:-padding // 2, :]
            if gt_mask.shape[1] < 512:
                padding = 512 - gt_mask.shape[1]
                labels = labels[:, padding // 2:-padding // 2]

        # save the results
        iio.imwrite(os.path.join(results_inferences, img.split('.')[0] + '.tiff'), labels)

        if plt_gt:
            # save plots of predicted mask with ground truth
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(labels)
            ax[0].set_title('Predicted')
            ax[1].imshow(gt_mask)
            ax[1].set_title('Ground Truth')
            plt.savefig(os.path.join(results_inspections, img.split('.')[0] + '.png'))
        else:
            # save plots of predicted mask
            plt.imshow(labels)
            plt.title(img.split('.')[0])
            # save as
            plt.savefig(os.path.join(results_inspections, img.split('.')[0] + '.png'))
        print(f"Processed {img}")

        verbose = False
        if verbose:
            plt.show()
            plt.imshow(wsi)
            plt.title(img.split('.')[0])
            plt.show()

            if plt_gt:
                plt.imshow(gt_mask)
                plt.title(img.split('.')[0])
                plt.show()

        plt.close()

    wsi_imgs_dict = {'flagged': wsi_imgs_flagged, 'regular': wsi_imgs_reg, 'no_wsi': no_wsi_imgs}
    # Save dictionary as a JSON file
    json_path = os.path.join(args.results_path, 'wsi_imgs_dict.json')
    with open(json_path, 'w') as json_file:
        json.dump(wsi_imgs_dict, json_file, indent=4)


    print() 
    print('wsi_imgs_dict saved')
    print('wsi img dict', wsi_imgs_dict)
    print()

    print("Done.")

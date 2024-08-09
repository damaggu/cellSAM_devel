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

    path_to_all_imgs = args.data_path
    all_images = os.listdir(path_to_all_imgs)
    all_images = sorted([img for img in all_images if img.endswith('.X.npy')])

    if bool(args.debug):
        # all_images = all_images[:28]
        # all_images = ['cell_00041.b0.X.npy']
        # all_images = ['cell_00012.b0.X.npy'] # blood
        # all_images = ['cell_00095.b0.X.npy']  # livecell but low contrast not working as expected
        # all_images = ['cell_00086.b0.X.npy']
        # all_images = ['cell_00045.b0.X.npy'] # TN
        # all_images = ['cell_00081.b0.X.npy']  # low contrast livecell
        # all_images = ['cell_00100.b0.X.npy'] # low contrast bacteria
        # all_images = ['cell_00086.b0.X.npy']
        # all_images = ['cell_00037.b0.X.npy']
        # all_images = ['cell_00038.b0.X.npy']

        # all_images = ['cell_00044.b0.X.npy']
        all_images = ['cell_00011.b0.X.npy']
        # all_images = ['cell_00027.b0.X.npy']
        # all_images = ['cell_00028.b0.X.npy']
        # 'TestHidden_043'
        # all_images = ['TestHidden_347.b0.X.npy']  # TODO:
        # all_images = ['TestHidden_205.b0.X.npy']  # TODO: needs medium -> median size 0.0049
        # all_images = ['TestHidden_179.b0.X.npy'] #TODO: double check
        # all_images = ['TestHidden_278.b0.X.npy'] #TODO: wrong medium
        # all_images = ['TestHidden_110.b0.X.npy'] #TODO: redo w. higher contrast
        # all_images = ['TestHidden_020.b0.X.npy']
        # all_images = ['TestHidden_397.b0.X.npy'] # large
        # all_images = ['TestHidden_399.b0.X.npy'] # tiny
        # all_images = ['TestHidden_073.b0.X.npy'] # tiny -> low contrast
        # all_images = ['TestHidden_398.b0.X.npy'] # medium
        # all_images = ['TestHidden_179.b0.X.npy']  #
        # all_images = ['TestHidden_393.b0.X.npy'] # small -> low contrast
        # all_images = ['TestHidden_005.b0.X.npy'] # large
        # all_images = ['TestHidden_047.b0.X.npy'] # medium
        # all_images = ['TestHidden_048.b0.X.npy'] # medium
        # all_images = ['TestHidden_060.b0.X.npy'] # medium
        # all_images = ['TestHidden_114.b0.X.npy']  # small

        # to check 027, 005, 179 --> too many?

        # new tests
        # all_images = ['TestHidden_027.b0.X.npy']
        # all_images = ['TestHidden_005.b0.X.npy']
        # all_images = ['TestHidden_179.b0.X.npy']
        # all_images = ['TestHidden_189.b0.X.npy']
        # all_images = ['TestHidden_002.b0.X.npy']
        # all_images = ['TestHidden_006.b0.X.npy']


        # all_images = ['TestHidden_001.b0.X.npy']
        # all_images = ['TestHidden_009.b0.X.npy']
        # all_images = ['TestHidden_011.b0.X.npy']
        # all_images = ['TestHidden_029.b0.X.npy']
        # all_images = ['TestHidden_054.b0.X.npy']
        # all_images = ['TestHidden_056.b0.X.npy']
        # all_images = ['TestHidden_334.b0.X.npy']
        # all_images = ['TestHidden_318.b0.X.npy']


        ### v7 was better
        # all_images = ['TestHidden_021.b0.X.npy'] # still bad
        # all_images = ['TestHidden_031.b0.X.npy']
        # all_images = ['TestHidden_035.b0.X.npy']


        # all_images = ['TestHidden_175.b0.X.npy']
        # all_images = ['TestHidden_092.b0.X.npy']
        # all_images = ['TestHidden_011.b0.X.npy']
        # all_images = ['TestHidden_177.b0.X.npy']
        # all_images = ['TestHidden_157.b0.X.npy'] # TODO: remove small cells, fill holes
        # all_images = ['TestHidden_098.b0.X.npy']
        # all_images = ['TestHidden_145.b0.X.npy']
        # all_images = ['TestHidden_001.b0.X.npy']
        # all_images = ['TestHidden_092.b0.X.npy']


        # all_images = ['TestHidden_027.b0.X.npy']
        # all_images = ['TestHidden_092.b0.X.npy']
        # all_images = ['TestHidden_001.b0.X.npy']
        # all_images = ['TestHidden_334.b0.X.npy']
        # all_images = ['TestHidden_304.b0.X.npy'] # needs lower threshold of 0.0025
        # all_images = ['TestHidden_248.b0.X.npy'] # adjusting median size to acc for this
        # all_images = ['TestHidden_316.b0.X.npy'] # adjusting median size to acc for this
        # all_images = ['TestHidden_331.b0.X.npy'] # adjusting large cell size
        # all_images = ['TestHidden_342.b0.X.npy'] # adjusting large cell size
        # all_images = ['TestHidden_083.b0.X.npy'] # adjusting large cell size, mean=0.066, std = 0.054
        # all_images = ['TestHidden_399.b0.X.npy'] # adjusting large cell size, mean=0.0699, std = 0.053
        # all_images = ['TestHidden_122.b0.X.npy'] # adjusting large cell size
        # all_images = ['TestHidden_124.b0.X.npy'] # adjusting large cell size
        # all_images = ['TestHidden_029.b0.X.npy'] # low contrast problem, mean=0.066, std = 0.048
        # all_images = ['TestHidden_203.b0.X.npy'] # low contrast problem, mean=0.066, std = 0.048

        # 83; adjusting edge to 5 or so; kick out v2 and v7; osilab overlap.... 263, 164

        # empty cells
        # 0.15

        # examples 001,
        # 122 v 10 vs. v14

        # double check boarders -> ignored? -> double check smlall cells on # 122

        ### probmel
        # all_images = ['TestHidden_347.b0.X.npy']


        # all_images = ['TestHidden_085.b0.X.npy']
        # all_images = ['cell_00032.b0.X.npy']
        pass
    else:
        import matplotlib

        matplotlib.use('Agg')

    hidden_raw_imgs = glob('/data/user-data/rdilip/cellSAM/raw/neurips/Testing/Hidden/images/*')
    tune_raw_imgs = glob('/data/user-data/rdilip/cellSAM/raw/neurips/Tuning/images/*')

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
        # model.iou_threshold = 0.9
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

    processing_dict = {}
    for img in tqdm(all_images[start:end]):
        if "Adenoid" in img or "Tonsil" in img:
            continue
        print(f"Starting to process {img}")
        processing_dict[img] = ""

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
            if 'val' in path_to_all_imgs:
                # /data/user-data/rdilip/cellSAM/raw/neurips/Tuning/images/
                gt_label_path = "./evals/tuning/labels/" + img.split('.')[0] + '_label.tiff'
                gt_mask = iio.imread(gt_label_path)

                gt_path = [s for s in tune_raw_imgs if base in s]
                if len(gt_path) > 1:
                    raise ValueError("More than one ground truth image found")
                gt_img = iio.imread(gt_path[0])
            elif 'hidden' or 'test' in path_to_all_imgs:
                # /data/user-data/rdilip/cellSAM/raw/neurips/Testing/Hidden/images/
                gt_path = [s for s in hidden_raw_imgs if base in s]
                if len(gt_path) > 1:
                    raise ValueError("More than one ground truth image found")
                gt_img = iio.imread(gt_path[0])
            else:
                raise ValueError("Unknown dataset")

        if use_gt:
            if base in data:
                use_wsi = False
            else:
                use_wsi = True
        else:
            # New Classiefer Stuff
            bloodcell = featurizer(clf, wsi)[0] == 1

        processing_dict[img] += "_bloodcell" if bloodcell else ""
        use_wsi = not bloodcell


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


        # new values
        # blood cell 12 -  0.121261
        # bacteria 0.05141763016581535
        # 1 - low contrast 0.03363665193319321
        # 2- low contrast 0.0307489
        # blood 0.07721655
        # TN 0.025151383

        ### check for 81 contrast values against TN
        # 81 low contrast 0.021225083

        # tissuenet 0.0162

        # 0.015 for low contrast images wihtout mean; with mean 0.37, std 0.095
        # 0.08 for blod cell
        # 0.04 for yeast
        # 0.02 for TN

        # TODO: add to ablation
        # TODO:preprocessing for some livecell images the nuclear channels is somehow not 0
        # if np.mean(abs(wsi[..., 1] - wsi[..., 2])) < 0.01:
        #     wsi[..., 1] = wsi[..., 0]
        #     processing_dict[img] += "_first_channel_removed"

        # wsi[..., 1] = wsi[..., 0]

        low_contrast, mean_diff, mean_std = is_low_contrast_clahe(wsi, lower_threshold=args.lower_contrast_threshold,
                                             upper_threshold=args.upper_contrast_threshold)
        low_contrast = (low_contrast and wsi[..., 1].max() == 0) if mean_diff < 0.05 else low_contrast
        low_contrast = low_contrast and not bloodcell
        # low_contrast = low_contrast and (0.4 > wsi[..., 2].mean() or wsi[..., 2].mean() > 0.5)
        bloodcell2 = 0.5 < wsi[..., 2].mean() < 0.75 and abs(wsi[..., 2].mean()- wsi[..., 1].mean()) > 0.0005 and wsi[..., 1].max() != 0
        low_contrast = low_contrast and not bloodcell2
        processing_dict[img] += "_low_contrast" if low_contrast else ""

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
            # wsi = equalize_adapthist(wsi, kernel_size=256, clip_limit=0.005)
            wsi = equalize_adapthist(wsi, kernel_size=kernel_size, clip_limit=clip_limit)
            # wsi = equalize_adapthist(wsi, kernel_size=256, clip_limit=0.03)
            wsi = adjust_gamma(wsi, gamma=gamma)
            # wsi = adjust_log(wsi, gain=1.2)


        # TODO: integrate this with the above
        # wsi = adjust_gamma(wsi, gamma=0.55)

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


        labels = segment_cellular_image(wsi, model=model, normalize=False, device=device)[0]

        inp = da.from_array(wsi, chunks=512)
        labels = segment_wsi(inp, 200, 200, args.iou_threshold, normalize=False, model=model,
                             device=device, bbox_threshold=args.bbox_threshold).compute()

        median_size, sizes, sizes_abs = get_median_size(labels)

        print(f"Median size: {median_size:.4f}")

        # img 27 -> 0.0026; should be bigger
        # where to have 0.0025 + sizes 56;
        # 0.0025 + sizes 41
        # 0.002 + 149

        processing_dict[img] += f"_median_{median_size:.4f}"
        processing_dict[img] += f"_num_{len(sizes)}"

        # plt.imshow(wsi)
        # plt.show()

        # middle sized cells -> median = 0.0017 -> 512x512 x 150-200; num 218
        # large, median size 0.007, num - 47;; median 0.008, num 52
        # small -> 0.0007795, num 156;;; median 0.00111
        # if len(sizes) < 5 -> do WSI
        #

        # if args.medium_cell_threshold <= median_size < args.large_cell_threshold and len(sizes) > 5 and mean_diff > args.medium_mean_diff_threshold:
        if args.medium_cell_threshold <= median_size < args.large_cell_threshold and len(sizes) > 5:
            # adjust WSI parameters
            # args.tile_size = 512
            # args.overlap = 200
            # args.iou_depth = 200
            processing_dict[img] += "_medium_cell"
        elif (median_size >= args.large_cell_threshold or bloodcell) and len(sizes) > 5:
            # large cells
            processing_dict[img] += "_large_cell"
            labels = segment_cellular_image(wsi, model=model, normalize=False, device=device)[0]
        else:
            doing_wsi = True
            processing_dict[img] += "_small_cell"
            processing_dict[img] += "_doing_wsi"
            # cells are medium or small -> do WSI
            inp = da.from_array(wsi, chunks=args.tile_size)
            labels = segment_wsi(inp, args.overlap, args.iou_depth, args.iou_threshold, normalize=False, model=model,
                                 device=device, bbox_threshold=args.bbox_threshold).compute()

        # # labels to individual masks
        # # filter out masks smaller than min size
        masks = []
        for mask in np.unique(labels):
            m_array = (labels == mask).astype(np.int32)
            if mask == 0:
                continue
            # is m_array at the edge?
            if m_array.sum() < args.cells_min_size and m_array[args.border_size:-args.border_size, args.border_size:-args.border_size].sum() == 0:
                continue
            masks.append(m_array * mask)
        labels = np.max(masks, axis=0)

        labels = relabel_mask(relabel_sequential(labels)[0])
        model.bbox_threshold = args.bbox_threshold

        print(labels.max())

        ### reshaping based on gt label
        if plt_gt:
            # fixing/reversing padding
            if gt_img.shape[0] < 512:
                # remove padding
                padding = 512 - gt_img.shape[0]
                labels = labels[padding // 2:-padding // 2, :]
            if gt_img.shape[1] < 512:
                padding = 512 - gt_img.shape[1]
                labels = labels[:, padding // 2:-padding // 2]

        # save the results
        iio.imwrite(os.path.join(results_inferences, img.split('.')[0] + '.tiff'), labels)

        if plt_gt and 'val' in path_to_all_imgs:
            # save plots of predicted mask with ground truth
            pass
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(labels)
            ax[0].set_title('Predicted')
            ax[1].imshow(gt_mask)
            ax[1].set_title('Ground Truth')
            plt.savefig(os.path.join(results_inspections, img.split('.')[0] + '.png'))
        elif plt_gt:
            # save plots of predicted mask
            plt.imshow(labels)
            plt.title(img.split('.')[0])
            # save as
            plt.savefig(os.path.join(results_inspections, img.split('.')[0] + '.png'))
        print(f"Processed {img}")

        if args.verbose:
            plt.imshow(labels)
            plt.title(img.split('.')[0] + "_{}".format(labels.max()))
            plt.show()

            gt_label_path = "./evals/tuning/labels/" + img.split('.')[0] + '_label.tiff'
            gt_mask = iio.imread(gt_label_path)
            # compute f1 between gt and predicted
            f1 = f1_score(gt_mask.ravel(), labels.ravel())
            print(f"F1 score: {f1}")

            plt.imshow(gt_mask)
            plt.show()

            plt.show()
            plt.imshow(wsi)
            plt.title(img.split('.')[0])
            plt.show()

        plt.close()

    # wsi_imgs_dict = {'flagged': wsi_imgs_flagged, 'regular': wsi_imgs_reg, 'no_wsi': no_wsi_imgs}
    # Save dictionary as a JSON file
    # json_path = os.path.join(args.results_path, 'wsi_imgs_dict.json')
    # with open(json_path, 'w') as json_file:
    #     json.dump(wsi_imgs_dict, json_file, indent=4)

    # do processing_dict to text file
    with open(os.path.join(args.results_path, 'processing_dict.txt'), 'w') as f:
        for key, value in processing_dict.items():
            f.write(f"{key}:{value}\n")

    print("Done.")

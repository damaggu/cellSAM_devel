import dask.array as da
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.segmentation import relabel_sequential

from cellSAM.model import get_local_model, segment_cellular_image
from cellSAM.utils import relabel_mask
from cellSAM.wsi import segment_wsi


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
    cellsam_pipeline(
        img='./debugdata/images/OpenTest_049.tif',
        model_path=None,
        chunks=256,
        low_contrast_enhancement=False,
        use_wsi=True,
        gauge_cell_size=False,
        visualize=True,
    )
import dask.array as da
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from cellSAM.model import get_model
from skimage.segmentation import relabel_sequential

from cellSAM.model import get_local_model, segment_cellular_image
from cellSAM.utils import relabel_mask, get_median_size, enhance_low_contrast
from cellSAM.wsi import segment_wsi


def use_cellsize_gaging(
        inp,
        model,
        device,
        overlap=200,
        iou_depth=200,
        iou_threshold=0.5,
        bbox_threshold=0.4,
        medium_cell_threshold=0.002,
        tile_size=256,
):
    labels = segment_wsi(inp, overlap, iou_depth, iou_threshold, normalize=False, model=model,
                         device=device, bbox_threshold=bbox_threshold).compute()

    median_size, sizes, sizes_abs = get_median_size(labels)

    print(f"Median size: {median_size:.4f}")

    # only if cells are small we to WSI inference
    if median_size < medium_cell_threshold:
        doing_wsi = True
        # cells are medium or small -> do WSI
        inp = da.from_array(inp, chunks=tile_size)
        labels = segment_wsi(inp, overlap, iou_depth, iou_threshold, normalize=False, model=model,
                             device=device, bbox_threshold=bbox_threshold).compute()
    else:
        labels = segment_cellular_image(inp, model=model, normalize=False, device=device)[0]

    return labels


def plot_output(labels,
                border_size=5,
                cells_min_size=500,
                ):
    # labels to individual masks
    # filter out masks smaller than min size

    uniq_labs = np.unique(labels)
    masks = np.zeros((labels.shape[0], labels.shape[1]), dtype=np.int32)
    for idx in tqdm(range(len(uniq_labs))):
        mask = uniq_labs[idx]
        m_array = (labels == mask).astype(np.int32)
        if mask == 0:
            continue
        # is m_array at the edge?
        if m_array.sum() < cells_min_size and m_array[border_size:-border_size,
                                                   border_size:-border_size].sum() == 0:
            continue
        masks[(labels==mask)] = mask

    if len(masks) == 0:
        print("No cells found")
        return
    
    labels = masks

    # result = labels
    # result = relabel_mask(relabel_sequential(labels)[0]) 
    # relabel_mask already calls relabel_sequential, so above line seems redundant
    result = relabel_mask(labels)

    plt.imshow(result)
    plt.show()

def load_image(img, swap_channels=False):
    img = iio.imread(img)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if swap_channels:
        # switch last 2 channels bc nuclear and whoelcell are switched, #TODO: autmatically detect or have input arg like cellpose
        img = img[..., [0, 2, 1]]
    img = img.astype(np.float32)
    # normalize to 0-1 min max - channelwise
    for i in range(3):
        # To accomodate empty channels
        if (np.max(img[..., i]) - np.min(img[..., i])) != 0:
            img[..., i] = (img[..., i] - np.min(img[..., i])) / (np.max(img[..., i]) - np.min(img[..., i]))
        else:
            img[..., i] = img[..., i]

    return img


def cellsam_pipeline(
        img,  # str or np.array
        chunks=256,
        model_path=None,
        bbox_threshold=0.4,
        low_contrast_enhancement=True,
        swap_channels=False,
        use_wsi=True,
        gauge_cell_size=True,
        visualize=False,
        overlap=200,
        iou_depth=200,
        iou_threshold=0.5,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_path is not None:
        modelpath = model_path
        model = get_local_model(modelpath)
        model.bbox_threshold = bbox_threshold
        model = model.to(device)
    else:
        model = None
        
        # To prevent creating model for each block
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Warning, using standard model. For better performance, use a model trained on your data.")
        model = get_model(None)
        model = model.to(device)
        model.eval()
        

    if isinstance(img, str):
        img = load_image(img, swap_channels=swap_channels)

    if low_contrast_enhancement:
        img = enhance_low_contrast(img)

    inp = da.from_array(img, chunks=chunks)

    if use_wsi:
        if gauge_cell_size:
            labels = use_cellsize_gaging(inp, model, device)
        else:
            labels = segment_wsi(inp, overlap, iou_depth, iou_threshold, normalize=False, model=model,
                                 device=device, bbox_threshold=bbox_threshold).compute()
    else:
        labels = segment_cellular_image(inp, model=model, normalize=False, device=device)[0]

    if visualize:
        plot_output(labels)
        plt.imshow(img)
        plt.show()


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
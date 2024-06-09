import numpy as np

from dask_image.ndmeasure._utils import _label
import dask.array as da
import functools
import operator
from .utils import relabel_mask
from .model import segment_cellular_image

def segment_wsi(X, overlap_depth: int=64, **segmentation_kwargs):
    """ Wholeslide segmentation function. Performs segmentation by tiling image with some overlap,
     computing segmentations, and stitching cells together using connected components.

    Args:
        X: np.ndarray, shape (H, W, C)
        overlap_depth: int, depth of overlap between tiles
        segmentation_kwargs: Any additional keyword arguments passed to segment_cellular_image
    Returns:
        np.ndarray, shape (H, W)
    """

    def _segment_fn(img):
        mask = segment_cellular_image(img, **segmentation_kwargs)[0]
        # Unsqueezing the last index is necessary -- don't remove this! It's a dask requirement.
        return mask[:, :, None].astype(np.int32)

    assert len(X.shape) == 3
    
    img = da.from_array(X, chunks=(512, 512, -1))
    out = img.map_overlap(
        _segment_fn, depth=(overlap_depth, overlap_depth, 0), boundary='none', trim=True, meta=np.array((), dtype=np.int32))[:,:,0]
    
    block_iter = zip(
        np.ndindex(*out.numblocks),
        map(functools.partial(operator.getitem, out),
            da.core.slices_from_chunks(out.chunks))
    )
    
    labeled_blocks = np.empty(out.numblocks, dtype=object)
    
    total = 0
    for index, label_block in block_iter:
        n_cells = np.max(label_block)
        offset = da.where(label_block > 0, total, _label.LABEL_DTYPE.type(0))
        label_block += offset
        labeled_blocks[index] = label_block
        total += n_cells
    
    combined = da.block(labeled_blocks.tolist())
    
    label_groups  = _label.label_adjacency_graph(
        combined, None, total
    )
    new_labeling = _label.connected_components_delayed(label_groups)
    relabeled = _label.relabel_blocks(combined, new_labeling)
    
    out_mask_relabel = relabel_mask(relabeled)
    return out_mask_relabel
import torch

from typing import List, Tuple
from skimage.measure import label as disconnect
from torch.nn.functional import one_hot


def seg_to_masks(seg_map: torch.Tensor) -> Tuple[List[torch.BoolTensor], List[int]]:
    masks = one_hot(seg_map).permute(2, 0, 1).to(torch.bool)
    masks = masks[masks.any(dim=(1, 2))]
    masks = masks.unbind(dim=0)
    labels = [seg_map[mask][0].item() for mask in masks]
    masks_connected = []
    labels_connected = []
    for mask, label in zip(masks, labels):
        mask_connected = torch.tensor(disconnect(mask, connectivity=1), dtype=torch.int64)
        mask_connected = one_hot(mask_connected).permute(2, 0, 1).to(torch.bool).unbind(dim=0)[1:]
        labels_connected += [label] * len(mask_connected)
        masks_connected += mask_connected

    return masks_connected, labels_connected

import torch

from PIL import Image
from torch.nn.functional import one_hot

from .segmentation.utils import segment, encode_seg
from .color_map.code_image import code_image, prompt_from_seg


def preferences_from_map(saliency_map, seg_map, mode="soft"):
    pref = {}
    if mode == "hard":
        saliency_map_binary = torch.where(saliency_map > 0.8, True, False)
        high_preferences = seg_map[saliency_map_binary]
        for i in range(151):
            if i in high_preferences:
                pref[i] = 0.8
            else:
                pref[i] = 0
    elif mode == "soft":
        seg_map_one_hot = one_hot(seg_map.to(torch.int64), 151).to(torch.bool)
        for i in range(151):
            pref[i] = (torch.sigmoid(10 * (saliency_map[seg_map_one_hot[:, :, i]] - 0.5))).mean()
    return pref


def encode_image(image_pil: Image, config, labels, saliency_map=None) -> dict:
    seg_128, control_image = segment(image_pil, config)
    seg_128_pil = Image.fromarray(seg_128.numpy())

    # Compute rate, lossless sm coding
    rate_sm = encode_seg(seg_128_pil)

    preferences = preferences_from_map(saliency_map, seg_128)

    # Encode the color guide from the image
    P, g, rate_cm, selective_color_map, encoded_color_map, mse = code_image(
        image=image_pil.resize((config.seg_dim, config.seg_dim)),
        seg_ts=seg_128.to(torch.int64),
        q=config.q_step,
        preferences=preferences,
        coarsening_strength=config.coarsening_factor,
        threshold=config.threshold)

    # Prompt from the labels
    prompt = prompt_from_seg(seg_128, labels)

    return {"coarsening_matrix": P,
            "guide": g,
            "rate_color_map": rate_cm,
            "rate_segmentation_map": rate_sm,
            "control_image": control_image,
            "prompt": prompt,
            "color_map_image": selective_color_map,
            "encoded_color_map": encoded_color_map,
            "mse_encoding": mse,
            "seg_map": seg_128}

import os
import sys
import yaml
import time
import torch
import logging
import argparse
import warnings

from PIL import Image
from pathlib import Path
from datetime import datetime
from huggingface_hub import login
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image

from config import Config, Preferences, LABELS
from eval.metrics import weighted_mse, segmentation_score
from encoder_decoder.sum_saliency.inference import inference
from encoder_decoder.encode import encode_image
from encoder_decoder.decode import decode_image, load_model_parameters
from encoder_decoder.segmentation.utils import segment
from encoder_decoder.decoder_utils.loader import load_pipeline

logger = logging.getLogger(__name__)
file_logger = logging.getLogger(__name__)

QUALITY_PARAMETER = {1: (0.0005, 1),  # High  ~ 0.085 bpp
                     2: (0.0015, 5),
                     3: (0.002, 6),  # Mid-High ~ 0.050 bpp
                     4: (0.0015, 10),
                     5: (0.003, 14),  # Mid-Low ~ 0.019 bpp
                     6: (0.0045, 22),
                     8: (0.0015, 26),
                     9: (0.0035, 27),  # Low ~ 0.007 bpp
                     10: (0.0045, 29)}  # Ultra-Low ~ 0.003 bpp

GUIDANCE_SCALE = [(20000, 12), (12000, 15), (8000, 16), (5000, 18), (4000, 21), (3500, 22), (3000, 24), (2500, 25),
                  (2000, 27), (1700, 30), (1500, 31), (1200, 32), (930, 35), (800, 37), (723, 43), (600, 65), (540, 75),
                  (390, 90), (350, 110), (300, 120), (270, 140), (228, 150), (150, 160), (120, 170), (100, 190), (80, 250), (10, 300)]


def find_scale(n_g):
    for (n, g) in GUIDANCE_SCALE:
        if n_g >= n:
            return g
    return 10


def format_image(image_size, image_pil):
    if image_pil.width < image_pil.height:
        image_pil = image_pil.resize((image_size, int(image_pil.height * image_size / image_pil.width)))
    else:
        image_pil = image_pil.resize((int(image_pil.width * image_size / image_pil.height), image_size))
    return image_pil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help="Specify config to run", default="./config.yml")
    args = parser.parse_args()

    config = Config(args.config)
    preferences = Preferences(args.preferences).list_val

    loglevel = logging.DEBUG if config.verbose else logging.INFO

    logging.basicConfig(stream=sys.stdout, level=loglevel, format='%(name)-8s %(levelname)-6s %(message)s')
    logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL + 1)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.CRITICAL + 1)
    warnings.filterwarnings("ignore")

    now = datetime.now()
    str_now = now.strftime("%Y-%m-%d_%H:%M")
    output_path = config.save_dir + config.exp_name + "_" + str_now + "/"
    os.mkdir(output_path)

    logger.info("Starting: " + config.exp_name)
    logger.info("Images loaded from: " + config.data_dir)
    logger.info("Saving config as: " + output_path + "config.yml")

    with open(output_path + "config.yml", "w") as file:
        yaml.dump(config.yaml_config, file)

    device = torch.device("cuda")
    login(token="")#insert token

    logger.info("Loading models...")
    pipeline = load_pipeline(device, config.cache_dir).to("cuda")

    lambda_infos = "/home/tobordin/compactdisk/users/tobordin/hierachical_coding/src/lambda_test/"
    model_parameters = load_model_parameters(lambda_infos)

    img_files = list(Path(config.data_dir).glob("*"))
    img_files.sort()

    for i, img_file in enumerate(img_files[:]):

        saliency_map = inference(img_file)
        saliency_map_tensor = torch.tensor(saliency_map)
        saliency_map_tensor_128 = torch.nn.functional.interpolate(saliency_map_tensor.unsqueeze(0).unsqueeze(0),
                                                              (128, 128), mode='bilinear').squeeze(0).squeeze(0)
        saliency_map_tensor_768 = torch.nn.functional.interpolate(saliency_map_tensor.unsqueeze(0).unsqueeze(0),
                                                              (config.image_dim, config.image_dim), mode='bilinear').squeeze(0)
        heatmap = torch.cat((saliency_map_tensor_768, torch.zeros(2, config.image_dim, config.image_dim)))
        heatmap_pil = to_pil_image(heatmap)
        qp = config.qp
        config.qp = qp
        q_step, coarse_str = QUALITY_PARAMETER[qp]
        config.coarsening_factor = coarse_str
        config.q_step = q_step

        image_save_name = output_path + os.path.basename(os.path.splitext(img_file)[0])

        logger.info("Opening image: " + str(img_file))

        image_pil = Image.open(img_file).convert("RGB")
        image_pil = format_image(config.image_dim, image_pil)
        image_pil = image_pil.crop((0, 0, config.image_dim, config.image_dim))
        image_pil.save(image_save_name + "_0_input.png")
        save_image(saliency_map_tensor, image_save_name + "_5_heatmap.png")
        overlay = Image.blend(image_pil, heatmap_pil, 0.5)
        overlay.save(image_save_name + "_6_overlay_heatmap.png")
        # encode the image
        t0 = time.time()
        encoded_image = encode_image(image_pil, config, LABELS, saliency_map_tensor_128)
        t1 = time.time()
        logger.info("encoding time: " + str(t1-t0) + "s")
        seg_map = encoded_image["seg_map"].to(torch.float32)
        seg_map = torch.nn.functional.interpolate(seg_map.unsqueeze(0).unsqueeze(0),
                                                  (config.image_dim, config.image_dim),
                                                  mode='nearest').squeeze(0).squeeze(0).to(torch.int64)

        total_bits = encoded_image["rate_color_map"] + encoded_image["rate_segmentation_map"]

        logger.info("Bits color map: " + str(encoded_image["rate_color_map"]))
        logger.info("Bits segmentation map: " + str(encoded_image["rate_segmentation_map"]))
        logger.info("bpp :" + str(total_bits / config.image_dim / config.image_dim))
        logger.info("MSE - color_map :" + str(encoded_image["mse_encoding"]))

        save_image(encoded_image["color_map_image"], image_save_name + "_2_color_map.png")
        save_image(encoded_image["encoded_color_map"], image_save_name + "_3_encoded_color_map_{}_{}.png".format(qp, total_bits))
        guidance_scale = find_scale(encoded_image["guide"].nelement())
        config.guidance_scale = guidance_scale
        guidance_scale = config.guidance_scale
        # decode the image
        with torch.no_grad():
            t0 = time.time()
            image_out = decode_image(pipeline=pipeline,
                                     prompt=config.prompt + encoded_image["prompt"],
                                     negative_prompt=config.negative_prompt,
                                     control_image=encoded_image["control_image"],
                                     guide=encoded_image["guide"],
                                     coarsening_matrix=encoded_image["coarsening_matrix"],
                                     config=config,
                                     guidance_parameters=model_parameters)
            t1 = time.time()
            logger.info("Decoding time: " + str(t1 - t0) + "s")
            image_out.save(image_save_name + "_1_decoded_{}_{}.png".format(qp, total_bits))
            seg_out, seg_image_out = segment(image_out, config)
            seg_out = torch.nn.functional.interpolate(seg_out.unsqueeze(0).unsqueeze(0),
                                                      (config.image_dim, config.image_dim),
                                                      mode='nearest').squeeze(0).squeeze(0).to(torch.int64)
            seg_image_out.save(image_save_name + "_4_seg_out.png")
            w_mse = weighted_mse(image_pil, image_out, preferences, seg_map)
            seg_score = segmentation_score(seg_map, seg_out)
            logger.info("Weighted mse:" + str(w_mse))
            logger.info("BCE seg:" + str(seg_score))
        with open(image_save_name + "_stats_{}.yml".format(qp), 'w') as file:
            stats = {
                "prompt": encoded_image["prompt"] + config.prompt,
                "seg_map_bpp": encoded_image["rate_segmentation_map"]/config.image_dim/config.image_dim,
                "signal_bpp": encoded_image["rate_color_map"]/config.image_dim/config.image_dim,
                "bpp": total_bits/config.image_dim/config.image_dim,
                "w_mse": str(w_mse),
                "BCE": str(seg_score),
                "guidance": guidance_scale,
                "qp": qp,
                "g": str(encoded_image["guide"].nelement()),
                "coarsening_factor": config.coarsening_factor,
                "q_step": config.q_step
            }
            yaml.dump(stats, file)


if __name__ == "__main__":
    main()

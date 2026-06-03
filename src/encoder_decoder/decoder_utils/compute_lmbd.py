import os
import torch
import pickle
import numpy as np
import torchvision.transforms.functional as tf

from PIL import Image
from loader import load_pipeline
from pathlib import Path
from constants import SEG_PALETTE
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation


def format_image(image_size, image_pil):
    if image_pil.width < image_pil.height:
        image_pil = image_pil.resize((image_size, int(image_pil.height * image_size / image_pil.width)))
    else:
        image_pil = image_pil.resize((int(image_pil.width * image_size / image_pil.height), image_size))
    return image_pil


def encode(x, pipeline, device):
    x = tf.to_tensor(x).to(device)
    z = pipeline.vae.encode(x.unsqueeze(0), return_dict=False)[0].sample()
    return z


def decode(z, pipeline, scale=True):
    z = z / pipeline.vae.config.scaling_factor if scale else z
    x = pipeline.vae.decode(z, return_dict=False)[0]
    return x


def main():
    device = torch.device("cuda")
    pipeline = load_pipeline(device, "/home/tobordin/compactdisk/users/tobordin/hierachical_coding/src/cache")
    pipeline = pipeline.to(device)
    output_path = os.path.join("/home/tobordin/compactdisk/users/tobordin/hierachical_coding/src/", "lambda_test")

    images_path = "/home/tobordin/compactdisk/datasets/imn/dataset/ILSVRC/Data/CLS-LOC/test/"
    images_files = list(Path(images_path).glob("*"))[:50]
    file = open(os.path.join(output_path, "lambda_ts.txt"), "rb")
    lambda_ts = pickle.load(file)
    file.close()
    mean_ts = []
    std_ts = []
    with torch.no_grad():
        for t, alpha_t, lambda_t in lambda_ts:
            error_t = []
            for image_file in images_files:
                image_pil = Image.open(image_file).convert("RGB")
                image_pil = image_pil.resize((512, 512))
                x_0 = tf.to_tensor(image_pil)
                z_0 = encode(image_pil, pipeline, device)
                for k in range(5):
                    eps = torch.randn_like(z_0)
                    z_noised = z_0 + lambda_t * ((1-alpha_t)/alpha_t)**0.5 * eps
                    x_noised = decode(z_noised, pipeline).squeeze(0).to("cpu")
                    error = x_noised - x_0
                    error = error.flatten()
                    error_t.append(error.mean().cpu())
            mean_ts.append(np.array(error_t).mean())
            std_ts.append(np.array(error_t).std())
        print("JOB is done: writing files")
        file = open(os.path.join(output_path, "mean_shift_latent.txt"), "wb")
        pickle.dump([[lambda_ts[i][0], mean_ts[i]] for i in range(len(lambda_ts))], file)
        file.close()
        file = open(os.path.join(output_path, "std_shift_latent.txt"), "wb")
        pickle.dump([[lambda_ts[i][0], std_ts[i]] for i in range(len(lambda_ts))], file)
        file.close()


def main2():
    device = torch.device("cuda")
    pipeline = load_pipeline(device, "/home/tobordin/compactdisk/users/tobordin/hierachical_coding/src/cache")
    pipeline = pipeline.to(device)
    vae_scaling_factor = pipeline.vae.config.scaling_factor
    output_path = os.path.join("/home/tobordin/compactdisk/users/tobordin/hierachical_coding/src/", "lambda_test")
    images_path = "/home/tobordin/compactdisk/datasets/imn/dataset/ILSVRC/Data/CLS-LOC/test/"
    images_files = list(Path(images_path).glob("*"))[:50]
    image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small", cache_dir="../../cache")
    image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small",
                                                                     cache_dir="../../cache")
    ts = []
    alpha_ts = []
    std_ts = []
    pipeline.scheduler.set_timesteps(num_inference_steps=50, device=device)
    timesteps = pipeline.scheduler.timesteps[:]
    with torch.no_grad():
        for t in timesteps:
            ts.append(t.cpu())
            alpha_t = pipeline.scheduler.alphas_cumprod[t]
            alpha_ts.append(alpha_t)
            error_ts = []
            for image_file in images_files:
                image_pil = Image.open(image_file).convert("RGB")
                image_pil = image_pil.resize((512, 512))
                z_0 = encode(image_pil, pipeline, device)
                prompt = ""
                neg_prompt = ""
                pixel_values = image_processor(image_pil, return_tensors="pt").pixel_values
                with torch.no_grad():
                    outputs = image_segmentor(pixel_values)
                seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image_pil.size[::-1]])[0]

                color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
                for label, color in enumerate(SEG_PALETTE):
                    color_seg[seg + 1 == label, :] = color
                color_seg = color_seg.astype(np.uint8)
                control_image = Image.fromarray(color_seg)
                prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
                    prompt=prompt,
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=neg_prompt,
                    prompt_embeds=None,
                    negative_prompt_embeds=None,
                    lora_scale=None,
                )
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
                controlnet = pipeline.controlnet
                image = pipeline.prepare_image(image=control_image,
                                               height=None,
                                               width=None,
                                               batch_size=1,
                                               num_images_per_prompt=1,
                                               device=device,
                                               dtype=controlnet.dtype,
                                               do_classifier_free_guidance=True,
                                               guess_mode=False)

                for k in range(5):
                    eps = torch.randn_like(z_0)
                    alpha_t = pipeline.scheduler.alphas_cumprod[t]
                    z_t = alpha_t ** 0.5 * z_0 * vae_scaling_factor + (1 - alpha_t) ** 0.5 * eps
                    z_t_input = torch.cat([z_t] * 2)
                    down_block_res_samples, mid_block_res_sample = pipeline.controlnet(
                        z_t_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        controlnet_cond=image,
                        conditioning_scale=1.0,
                        guess_mode=False,
                        return_dict=False,
                    )
                    eps_theta = pipeline.unet(
                        z_t_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=None,
                        cross_attention_kwargs=None,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        added_cond_kwargs=None,
                        return_dict=False,
                    )[0]
                    eps_theta_uncond, eps_theta_text = eps_theta.chunk(2)
                    eps_theta = eps_theta_uncond + 6 * (eps_theta_text - eps_theta_uncond)
                    error = eps_theta - eps
                    error = error.flatten()
                    error_ts.append(error.mean().cpu())
            std_ts.append(np.array(error_ts).std())
    file = open(os.path.join(output_path, "lambda_ts.txt"), "wb")
    pickle.dump([(ts[i], alpha_ts[i], std_ts[i]) for i in range(len(ts))], file)
    file.close()


main()

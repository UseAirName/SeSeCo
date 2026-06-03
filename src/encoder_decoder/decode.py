import torch
import pickle
from PIL import Image

from .decoder_utils.guided_diffusion import decode


def load_model_parameters(path):
    file = open(path + "lambda_ts.txt", "rb")
    lambda_vals = pickle.load(file)
    file.close()
    file = open(path + "mean_shift_latent.txt", "rb")
    mean_shift_vals = pickle.load(file)
    file.close()
    file = open(path + "std_shift_latent.txt", "rb")
    std_shift_vals = pickle.load(file)
    file.close()
    return lambda_vals, mean_shift_vals, std_shift_vals


def decode_image(pipeline,
                 guide: torch.Tensor,
                 prompt: str,
                 control_image: Image,
                 config,
                 coarsening_matrix: torch.Tensor,
                 negative_prompt: str,
                 guidance_parameters):
    with torch.no_grad():
        image_out = decode(pipeline=pipeline,
                           prompt=prompt,
                           negative_prompt=negative_prompt,
                           image=control_image,
                           c_map=guide.to("cuda"),
                           num_inference_steps=config.steps,
                           controlnet_conditioning_scale=config.control_scale,
                           guidance_scale=config.guidance_scale,
                           repeat_guidance=config.repeat,
                           conditional_scale=config.conditional_scale,
                           coarsening_matrix=coarsening_matrix.to("cuda"),
                           eps_error=guidance_parameters[0],
                           lambda_mean=guidance_parameters[1],
                           lambda_std=guidance_parameters[2])[0]
    return image_out

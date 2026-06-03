import torch
import torch_dct as dct
import torchvision.transforms
import torchvision.transforms.functional as tf

from PIL import Image
from typing import Literal
from diffusers import DiffusionPipeline


def color_map(x: torch.Tensor | Image.Image,
              size: int,
              method: Literal["gaussian", "identity"] = "identity"

              ) -> torch.Tensor:
    """
    Color map computed with 2D-DCT
    :param x: image tensor or PIL
    :param size: thresholding parameter in the DCT
    :param method: /
    :return: tensor of the color map
    """
    if type(x) == Image.Image:
        x = tf.to_tensor(x)
    # 2d DCT of x
    X = dct.dct_2d(x)
    # Filter
    if method == "gaussian":
        tsh_filter = torch.zeros_like(X)
    elif method == "identity":
        tsh_filter = torch.zeros_like(X)
        for i in range(size):
            tsh_filter[:, i, i] = 1
    else:
        raise RuntimeError("Non implemented method: ", method)
    X = torch.matmul(tsh_filter, X)
    X = torch.matmul(X, tsh_filter)
    # inverse 2d DCT
    x_out = dct.idct_2d(X)
    return x_out


def latent_color_loss(c_x: torch.Tensor,
                      z_0_t: torch.Tensor,
                      pipeline: DiffusionPipeline,
                      mean_error: float,
                      resolution: int

                      ) -> torch.Tensor:
    """
    Loss between conditional color map and estimated color map at timestep t
    :param c_x: conditional color map
    :param z_0_t: estimated real latent at timestep t
    :param pipeline: diffusion pipeline (here for the decoder VAE)
    :param mean_error: estimated shifting of the mean
    :param resolution: resolution of the color map
    :return: guidance loss value
    """
    # Decoded estimated image
    x_0_t = pipeline.vae.decode(z_0_t / pipeline.vae.config.scaling_factor, return_dict=False)[0]
    x_0_t = x_0_t * 0.5 + 0.5
    x_0_t = x_0_t.clamp(0, 1).squeeze(0)

    # Estimate color maps
    c_0_t = color_map(x_0_t - mean_error * torch.ones_like(x_0_t), resolution)

    # Resize color maps
    c_x = tf.resize(c_x, [resolution, resolution],
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=False)
    c_0_t = tf.resize(c_0_t, [resolution, resolution],
                      interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=False)

    # Compute loss between color maps
    loss = torch.nn.functional.mse_loss(c_x, c_0_t, reduction="sum")
    return loss


def latent_coarsening_loss(c_x: torch.Tensor,
                           z_0_t: torch.Tensor,
                           pipeline: DiffusionPipeline,
                           mean_error: float,
                           A: torch.Tensor
                           ) -> torch.Tensor:
    """
    Loss between conditional color map and estimated color map at timestep t
    :param c_x: conditional color map
    :param z_0_t: estimated real latent at timestep t
    :param pipeline: diffusion pipeline (here for the decoder VAE)
    :param mean_error: estimated shifting of the mean
    :param A: matrix to compute the coarsening
    :return: guidance loss value
    """
    # Decoded estimated image
    x_0_t = pipeline.vae.decode(z_0_t / pipeline.vae.config.scaling_factor, return_dict=False)[0]
    x_0_t = x_0_t * 0.5 + 0.5
    x_0_t = tf.resize(x_0_t.clamp(0, 1).squeeze(0), [128, 128])

    # Estimate color maps
    x_flatten = (x_0_t).flatten(1, 2).T
    c_0_t = torch.matmul(A, x_flatten)
    c_r, c_g, c_b = (c_x - c_0_t).T
    AA_T_inv = torch.diag(1 / torch.matmul(A, A.T).diag())
    loss_r = 1 / AA_T_inv.norm() * torch.matmul(torch.matmul(c_r, AA_T_inv), c_r.T)
    loss_g = 1 / AA_T_inv.norm() * torch.matmul(torch.matmul(c_g, AA_T_inv), c_g.T)
    loss_b = 1 / AA_T_inv.norm() * torch.matmul(torch.matmul(c_b, AA_T_inv), c_b.T)
    loss = loss_r + loss_g + loss_b
#    loss = (c_r*c_r + c_g*c_g + c_b*c_b).mean()/60
    return loss

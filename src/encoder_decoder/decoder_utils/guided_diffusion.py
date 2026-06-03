import torch

from PIL import Image
from typing import List, Optional
from diffusers import DiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.pipelines.controlnet import MultiControlNetModel

from .lmbd import mean_shift_latent, std_shift_latent, lambda_t
from .guidance_loss import latent_color_loss, latent_coarsening_loss


def diffusion_step(z_t: torch.Tensor,
                   t: torch.Tensor,
                   i: int,
                   control_image: torch.Tensor,
                   prompt_embeds: torch.Tensor,
                   conditional_scale: float,
                   pipeline: StableDiffusionControlNetPipeline,
                   controlnet_keep: List,
                   controlnet_conditioning_scale,
                   ) -> torch.Tensor:
    """
    Compute a step estimation of the diffusion model with conditional guidance and controlnet guidance
    :param z_t: sample
    :param t: timestep
    :param i: index of timestep
    :param control_image: image for controlnet
    :param prompt_embeds: prompt embeddings
    :param conditional_scale: scaling of the conditional diffusion
    :param controlnet_conditioning_scale: controlnet conditioning scale per timstep
    :param controlnet_keep: control_net to use per timestep
    :param pipeline: diffusion pipeline
    :return: epsilon estimation after unconditional guidance
    """
    z_t = torch.cat([z_t] * 2)
    z_t = pipeline.scheduler.scale_model_input(z_t, t)

    # controlnet(s) inference
    z_t_cn = z_t
    prompt_embeds_cn = prompt_embeds

    if isinstance(controlnet_keep[i], list):
        cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
    else:
        controlnet_cond_scale = controlnet_conditioning_scale
        if isinstance(controlnet_cond_scale, list):
            controlnet_cond_scale = controlnet_cond_scale[0]
        cond_scale = controlnet_cond_scale * controlnet_keep[i]

    down_block_res_samples, mid_block_res_sample = pipeline.controlnet(
        z_t_cn,
        t,
        encoder_hidden_states=prompt_embeds_cn,
        controlnet_cond=control_image,
        conditioning_scale=cond_scale,
        guess_mode=False,
        return_dict=False,
    )

    # predict the noise residual
    eps_theta = pipeline.unet(
        z_t,
        t,
        encoder_hidden_states=prompt_embeds,
        timestep_cond=None,
        cross_attention_kwargs=pipeline.cross_attention_kwargs,
        down_block_additional_residuals=down_block_res_samples,
        mid_block_additional_residual=mid_block_res_sample,
        added_cond_kwargs=None,
        return_dict=False,
    )[0]

    # Split variance and mean models, split unconditional and conditional values
    eps_theta_uncond, eps_theta_text = eps_theta.chunk(2)

    # Apply unconditional guidance with scaling
    eps_theta = eps_theta_uncond + conditional_scale * (eps_theta_text - eps_theta_uncond)

    # Return epsilon estimation of the model
    return eps_theta


def guided_diffusion_step(z_t: torch.Tensor,
                          t: torch.Tensor,
                          i: int,
                          control_image,
                          prompt_embeds: torch.Tensor,
                          c_map: torch.Tensor,
                          coarsening_matrix: torch.Tensor,
                          conditional_scale: float,
                          guidance_scale: float,
                          controlnet_conditioning_scale: List,
                          controlnet_keep: List,
                          pipeline: StableDiffusionControlNetPipeline,
                          eps_error,
                          lambda_mean,
                          lambda_std
                          ) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute a step estimation of the diffusion model with conditional guidance, fine guidance and control_net guidance
    :param z_t: sample
    :param t: timestep
    :param i: index of timestep
    :param control_image: image for controlnet
    :param prompt_embeds: prompt embeddings
    :param c_map: conditional color map
    :param coarsening_matrix: coarsening matrix to compute c_map
    :param conditional_scale: scaling of the conditional diffusion
    :param guidance_scale: guidance scale for color guidance
    :param controlnet_conditioning_scale: controlnet conditioning scale per timstep
    :param controlnet_keep: control_net to use per timestep
    :param pipeline: diffusion pipeline
    :param eps_error: error on epsilon prediction
    :param lambda_mean: shifting of the mean on decoding
    :param lambda_std: std on decoding        lmbda_t = get_lmbda(t.item(), eps_error)
    :return: epsilon estimation after unconditional guidance
    """
    # Compute the gradient of the classifier loss
    # Grad_{z_t} l(z_0_t, sigma)
    with torch.enable_grad():
        # Gradient on z_t
        z_t_var = z_t.detach().requires_grad_(True)

        z_t = torch.cat([z_t] * 2)
        z_t = pipeline.scheduler.scale_model_input(z_t, t)

        # controlnet(s) inference
        z_t_cn = z_t
        prompt_embeds_cn = prompt_embeds

        if isinstance(controlnet_keep[i], list):
            cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
        else:
            controlnet_cond_scale = controlnet_conditioning_scale
            if isinstance(controlnet_cond_scale, list):
                controlnet_cond_scale = controlnet_cond_scale[0]
            cond_scale = controlnet_cond_scale * controlnet_keep[i]

        down_block_res_samples, mid_block_res_sample = pipeline.controlnet(
            z_t_cn,
            t,
            encoder_hidden_states=prompt_embeds_cn,
            controlnet_cond=control_image,
            conditioning_scale=cond_scale,
            guess_mode=False,
            return_dict=False,
        )

        # predict the noise residual
        eps_theta = pipeline.unet(
            z_t,
            t,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=None,
            cross_attention_kwargs=pipeline.cross_attention_kwargs,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            added_cond_kwargs=None,
            return_dict=False,
        )[0]
        # Split variance and mean models, split unconditional and conditional values
        eps_theta_uncond, eps_theta_text = eps_theta.chunk(2)

        # Apply unconditional guidance
        eps_theta = eps_theta_uncond + conditional_scale * (eps_theta_text - eps_theta_uncond)

        # Compute z_0_t from epsilon_theta
        alpha_t = pipeline.scheduler.alphas_cumprod[t]
        lambda_ts = lambda_t(t.item(), eps_error)
        mean_shift = mean_shift_latent(t.item(), lambda_mean)

        z_0_t = (z_t_var - (1 - alpha_t) ** 0.5 * eps_theta) / alpha_t ** 0.5

        # Compute the guidance loss
        loss = latent_coarsening_loss(c_x=c_map, z_0_t=z_0_t, pipeline=pipeline, mean_error=0,
                                      A=coarsening_matrix)

        # Compute the gradient of the loss
        grad = torch.autograd.grad(loss, z_t_var)[0].detach()

    std_error = std_shift_latent(t, lambda_std)

    # Compute fine guidance delta
    delta_epsilon = guidance_scale * grad * alpha_t / (2 * lambda_ts * (1 - alpha_t) ** 0.5)
    return eps_theta, delta_epsilon, loss


def decode(pipeline: StableDiffusionControlNetPipeline,
           prompt: str,
           negative_prompt: str,
           image: Image,
           c_map: torch.Tensor,
           coarsening_matrix: torch.Tensor,
           eps_error=None,
           lambda_mean=None,
           lambda_std=None,
           height: Optional[int] = None,
           width: Optional[int] = None,
           num_inference_steps: int = 20,
           guidance_scale: float = 7.5,
           conditional_scale: float = 7.5,
           repeat_guidance: int = 3,
           controlnet_conditioning_scale: float = 1.0,
           num_images_per_prompt: int = 1,
           clip_skip: Optional = None,
           prompt_embeds: Optional[torch.tensor] = None,
           latents: Optional[torch.FloatTensor] = None,
           guide_diffusion: Optional[bool] = True
           ):
    """
    Decode image from its clip embeddings using a diffusion model pipeline
    :param pipeline: diffusion pipeline
    :param image: controlnet image
    :param height: height of generated image
    :param width: width of generated image
    :param guidance_scale: scale of the fine guidance
    :param controlnet_conditioning_scale:
    :param clip_skip:
    :param num_images_per_prompt: 1 for now
    :param conditional_scale: scale of the conditional guidance
    :param c_map: color_map guide
    :param coarsening_matrix: coarsening matrix of the color map
    :param num_inference_steps: number of diffusion steps
    :param prompt: prompt added to the image embeds
    :param negative_prompt: prompt added to the image embeds
    :param repeat_guidance: number of repeated guidance per step
    :param guide_diffusion: set to "False" to not guide with color
    :param prompt_embeds: Do not provide if prompt is available
    :param latents: optional latents to initialize diffusion
    :param eps_error: error on epsilon prediction
    :param lambda_mean: shifting of the mean on decoding
    :param lambda_std: std on decoding
    :return: decoded image from clip_embeds and sigma
    """
    controlnet = pipeline.controlnet._orig_mod if is_compiled_module(pipeline.controlnet) else pipeline.controlnet

    control_guidance_start = 0.
    control_guidance_end = 1.

    # align format for control guidance
    if not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
        mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
        control_guidance_start, control_guidance_end = (
            mult * [control_guidance_start],
            mult * [control_guidance_end],
        )

    negative_prompt_embeds = None
    pipeline._guidance_scale = conditional_scale
    pipeline._clip_skip = clip_skip
    pipeline._cross_attention_kwargs = None

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = pipeline._execution_device

    if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
        controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

    # 3. Encode input prompt
    text_encoder_lora_scale = (
        pipeline.cross_attention_kwargs.get("scale", None) if pipeline.cross_attention_kwargs is not None else None
    )
    prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        pipeline.do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        lora_scale=text_encoder_lora_scale,
        clip_skip=pipeline.clip_skip,
    )
    # For classifier free guidance, we need to do two forward passes.
    # Here we concatenate the unconditional and text embeddings into a single batch
    # to avoid doing two forward passes
    if pipeline.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    # 4. Prepare image
    if isinstance(controlnet, ControlNetModel):
        image = pipeline.prepare_image(
            image=image,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=controlnet.dtype,
            do_classifier_free_guidance=pipeline.do_classifier_free_guidance,
            guess_mode=False,
        )
        height, width = image.shape[-2:]
    elif isinstance(controlnet, MultiControlNetModel):
        images = []

        # Nested lists as ControlNet condition
        if isinstance(image[0], list):
            # Transpose the nested image list
            image = [list(t) for t in zip(*image)]

        for image_ in image:
            image_ = pipeline.prepare_image(
                image=image_,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=pipeline.do_classifier_free_guidance,
                guess_mode=False,
            )

            images.append(image_)

        image = images
        height, width = image[0].shape[-2:]
    else:
        assert False

    # 5. Prepare timesteps
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipeline.scheduler.timesteps[:]
    t_start = timesteps[0]
    pipeline._num_timesteps = len(timesteps)

    # 6. Prepare latent variables
    num_channels_latents = pipeline.unet.config.in_channels
    z_t = pipeline.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        None,
        latents,
    )

    # 7.2 Create tensor stating which controlnets to keep
    controlnet_keep = []
    for i in range(len(timesteps)):
        keeps = [
            1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
            for s, e in zip(control_guidance_start, control_guidance_end)
        ]
        controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

    # 8. Denoising loop
    progress_bar = pipeline.progress_bar(timesteps)
    for i, t in enumerate(progress_bar):
        z_tm1, eps_theta = None, None

        # Do not guide on last step
        if i == len(progress_bar) - 1:
            guide_diffusion = False

        # Guided diffusion
        if guide_diffusion and i != 0:
            # Repeat guidance on step t
            for k in range(repeat_guidance):
                # Edit eps_theta
                eps_theta, delta_epsilon, loss = guided_diffusion_step(z_t=z_t,
                                                                       t=t,
                                                                       i=i,
                                                                       control_image=image,
                                                                       prompt_embeds=prompt_embeds,
                                                                       c_map=c_map,
                                                                       coarsening_matrix=coarsening_matrix,
                                                                       conditional_scale=conditional_scale,
                                                                       guidance_scale=guidance_scale,
                                                                       controlnet_conditioning_scale=controlnet_conditioning_scale,
                                                                       controlnet_keep=controlnet_keep,
                                                                       pipeline=pipeline,
                                                                       eps_error=eps_error,
                                                                       lambda_mean=lambda_mean,
                                                                       lambda_std=lambda_std)

                # Modify eps_theta
                eps_theta = eps_theta + delta_epsilon

                # Step in the scheduler
                z_tm1 = pipeline.scheduler.step(eps_theta, t, z_t)[0]

                loss_desc = "loss_color :" + "{:.5f}".format(loss.item())
                progress_bar.set_description(desc=loss_desc + "| ")

                # Go back one step back
                if k != repeat_guidance - 1:
                    pipeline.scheduler._step_index -= 1
                    tm1 = timesteps[i + 1]
                    alpha_t, alpha_tm1 = pipeline.scheduler.alphas_cumprod[t], pipeline.scheduler.alphas_cumprod[tm1]
                    eps = torch.randn_like(z_tm1)
                    # Add the corresponding noise for timestep t
                    z_t = (alpha_t / alpha_tm1) ** 0.5 * z_tm1 + (1 - alpha_t / alpha_tm1) ** 0.5 * eps
        else:
            eps_theta = diffusion_step(z_t=z_t,
                                       t=t,
                                       i=i,
                                       control_image=image,
                                       prompt_embeds=prompt_embeds,
                                       conditional_scale=conditional_scale,
                                       controlnet_conditioning_scale=controlnet_conditioning_scale,
                                       controlnet_keep=controlnet_keep,
                                       pipeline=pipeline)
            # Step in the scheduler
            z_tm1 = pipeline.scheduler.step(eps_theta, t, z_t)[0]
        # Going to next timestep
        z_t = z_tm1

    image = pipeline.vae.decode(z_t / pipeline.vae.config.scaling_factor, return_dict=False)[0]

    image = pipeline.image_processor.postprocess(image, output_type="pil", do_denormalize=[True] * image.shape[0])
    # Offload all models
    pipeline.maybe_free_model_hooks()

    return image


def encode_image(pipeline: DiffusionPipeline, image):
    z = pipeline.vae.encode(image, return_dict=False)[0].sample()
    return z

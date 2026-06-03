import torch
import open_clip

from diffusers import DPMSolverMultistepScheduler, StableDiffusionControlNetPipeline, ControlNetModel


def load_pipeline(device: torch.device, cache_dir: str):
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_seg",
                                                 torch_dtype=torch.float32,
                                                 device=device,
                                                 cache_dir=cache_dir)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained("benjamin-paine/stable-diffusion-v1-5",
                                                                 controlnet=controlnet,
                                                                 torch_dtype=torch.float32,
                                                                 cache_dir=cache_dir)
    scheduler_config = {
        'beta_schedule': 'scaled_linear',
        'beta_start': 0.00085,
        'beta_end': 0.012,
        'prediction_type': 'epsilon',
        "num_train_timesteps": 1000,
        "steps_offset": 1,
        "thresholding": False,
        "use_karras_sigmas": True,
        'algorithm_type': 'dpmsolver++'
    }
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(scheduler_config)
    return pipeline


def load_clip_model(device: torch.device, cache_dir: str):
    """
    Load clip model and preprocessing
    """
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-L-14',
                                                                           pretrained='openai',
                                                                           device=device,
                                                                           cache_dir=cache_dir)
    return clip_model, clip_preprocess

import cv2
import numpy as np
import torch
from diffusers import (
    StableDiffusionInpaintPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
)
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    BlipProcessor,
    BlipForConditionalGeneration,
)
import skfmm


def resize(img, new_shape):
    img_h, img_w = img.shape[:2]
    if img_h > new_shape[0] or img_w > new_shape[1]:
        return cv2.resize(
            img.astype(np.uint8), (new_shape[1], new_shape[0]), cv2.INTER_AREA
        )
    else:
        return cv2.resize(
            img.astype(np.uint8), (new_shape[1], new_shape[0]), cv2.INTER_LINEAR
        )


def get_expand_mask(mask, expand_direction, expand_pixels):
    old_h, old_w = mask.shape[:2]
    new_w = old_w + expand_pixels
    new_mask = np.zeros([old_h, new_w])
    if expand_direction.lower() == "right":
        new_mask[:, :old_w] = mask
    elif expand_direction.lower() == "left":
        new_mask[:, expand_pixels:] = mask
    else:
        raise Exception(f"Doesn't support direction {expand_direction}")

    new_mask = new_mask.astype(np.uint8)
    return new_mask


def get_expand_region(image_shape, expand_direction, expand_pixels):
    old_h, old_w = image_shape
    new_w = old_w + expand_pixels
    expand_region = np.zeros([old_h, new_w])

    if expand_direction.lower() == "right":
        expand_region[:, old_w:] = 255
    elif expand_direction.lower() == "left":
        expand_region[:, :expand_pixels] = 255
    else:
        raise Exception(f"Doesn't support direction {expand_direction}")

    return expand_region.astype(np.uint8)


def get_sdf_map(mask):
    phi = np.where(mask == 255, 0.5, -0.5)
    sdf_map = skfmm.distance(phi, dx=1)
    return sdf_map


def get_binary_mask(sdf_map):
    binary_mask = np.where(sdf_map.squeeze() < 0, 0, 255)
    return binary_mask


def get_input(mask, expand_region, input_shape=[256, 256]):
    mask = resize(mask, input_shape)
    expand_region = resize(expand_region, input_shape)
    _, mask = cv2.threshold(mask, 128, 255, 0)
    _, expand_region = cv2.threshold(expand_region, 128, 255, 0)
    sdf_map = get_sdf_map(mask)
    expand_region = expand_region / 127.5 - 1
    data = np.concatenate([[sdf_map], [expand_region]], axis=0).astype(
        np.float32
    )
    data = torch.as_tensor(data).unsqueeze(0)
    return data


def get_average_color(image, mask):
    selected_pixels = image[mask > 0]
    average_color = selected_pixels.mean(axis=0).astype(int)  # (R, G, B)
    return average_color


def fill_img(img, mask, expand_direction, expand_pixels):
    average_color = get_average_color(img, mask)

    old_h, old_w = img.shape[:2]
    new_w = old_w + expand_pixels
    new_img = np.zeros([old_h, new_w, 3])

    if expand_direction.lower() == "left":
        new_img[:, :expand_pixels] = average_color
    elif expand_direction.lower() == "right":
        new_img[:, old_h:] = average_color

    return new_img


def restore_from_mask(
    pipe,
    init_images,
    mask_images,
    prompts=[],
    negative_prompts=[],
    num_inference_steps=30,
    guidance_scale=7.5,
    denoise_strength=0.75,
    sampler="euler",
):
    """
    Restore an image using stable diffusion inpainting with customizable parameters.

    Args:
        cropped_image (PIL.Image): The input image to be restored
        mask_image (PIL.Image): The mask indicating areas to be inpainted
        prompt (str): Text prompt for guided generation
        negative_prompt (str): Text prompt for what to avoid in generation
        num_inference_steps (int): Number of denoising steps
        guidance_scale (float): How strongly the image should conform to prompt (CFG scale)
        seed (int, optional): Random seed for reproducibility
        denoise_strength (float): Strength of denoising, between 0.0 and 1.0
        sampler (str): Sampling method to use ('euler_a', 'euler', 'heun', 'dpm_2',
                       'dpm_2_ancestral', 'lms', 'ddim', 'pndm')

    Returns:
        PIL.Image: The restored image
    """

    # Set device and optimize memory
    torch.cuda.empty_cache()
    pipe.enable_attention_slicing()

    if sampler == "euler_a":
        from diffusers import EulerAncestralDiscreteScheduler

        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config
        )
    elif sampler == "euler":
        from diffusers import EulerDiscreteScheduler

        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config
        )
    elif sampler == "heun":
        from diffusers import HeunDiscreteScheduler

        pipe.scheduler = HeunDiscreteScheduler.from_config(
            pipe.scheduler.config
        )
    elif sampler == "dpm_2":
        from diffusers import KDPM2DiscreteScheduler

        pipe.scheduler = KDPM2DiscreteScheduler.from_config(
            pipe.scheduler.config
        )
    elif sampler == "dpm_2_ancestral":
        from diffusers import KDPM2AncestralDiscreteScheduler

        pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(
            pipe.scheduler.config
        )
    elif sampler == "lms":
        from diffusers import LMSDiscreteScheduler

        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    elif sampler == "ddim":
        from diffusers import DDIMScheduler

        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif sampler == "pndm":
        from diffusers import PNDMScheduler

        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
    elif sampler == "DPM":
        from diffusers import DPMSolverMultistepScheduler

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++",
        )

    with torch.inference_mode():
        outputs = pipe(
            prompt=prompts,
            negative_prompt=negative_prompts,
            image=init_images,
            mask_image=mask_images,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            output_type="np",
        ).images
    torch.cuda.empty_cache()

    images = []
    for image in outputs:
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        images.append(image)

    return images


def generate_image_caption(model, processor, image, device):
    inputs = processor(image, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
    )
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption


def get_sd_pipeline(model_id, seed):
    torch.cuda.empty_cache()
    if seed is not None:
        torch.manual_seed(seed)

    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder"
    )
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    return pipe


def get_blip(model_id):
    model = BlipForConditionalGeneration.from_pretrained(model_id)
    processor = BlipProcessor.from_pretrained(model_id)

    return model, processor

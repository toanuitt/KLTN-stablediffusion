import cv2
import numpy as np
import torch

from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionInpaintPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    ControlNetModel,
)
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    BlipProcessor,
    BlipForConditionalGeneration,
)
import skfmm
from PIL import Image


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
        new_img[:, expand_pixels:] = img
    elif expand_direction.lower() == "right":
        new_img[:, old_w:] = average_color
        new_img[:, :old_w] = img

    return new_img


def get_object_image(image, mask):
    mask_inv = cv2.cvtColor(~mask, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_or(image, mask_inv)


def get_object_focus_image(image, mask):
    object_image = get_object_image(image, mask)
    mask_h, mask_w = mask.shape[:2]

    binary_mask = np.where(mask > 128, True, False)
    start_w, end_w = -1, -1
    for row in range(mask_w):
        if np.any(binary_mask[:, row]) and start_w == -1:
            start_w = row
        if (
            not np.all(binary_mask[:, row])
            and not np.any(binary_mask[:, row])
            and start_w != -1
        ):
            end_w = row
            break

    start_h, end_h = -1, -1
    for col in range(mask_h):
        if np.any(binary_mask[col, :]) and start_h == -1:
            start_h = col
        if (
            not np.all(binary_mask[col, :])
            and not np.any(binary_mask[col, :])
            and start_h != -1
        ):
            end_h = col
            break

    print(f"{start_h=}, {end_h=}, {start_w=}, {end_w=}")
    object_image = object_image[start_h:end_h, start_w:end_w].copy()
    return object_image


def perform_outpaint(
    pipe,
    pipe_ver,
    torch_generator,
    init_images,
    mask_images,
    prompts=[""],
    negative_prompts=[""],
    object_images=[],
    num_inference_steps=30,
    guidance_scale=7.5,
    denoise_strength=0.75,
    sampler="euler",
):
    torch.cuda.empty_cache()

    sampler = sampler.lower()
    if sampler == "euler":
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
    elif sampler == "lms":
        from diffusers import LMSDiscreteScheduler

        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    elif sampler == "ddim":
        from diffusers import DDIMScheduler

        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif sampler == "pndm":
        from diffusers import PNDMScheduler

        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
    elif sampler == "dpm":
        from diffusers import DPMSolverMultistepScheduler

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++",
        )

    init_images = (
        torch.as_tensor(np.array(init_images, dtype=np.float16))
        .permute(0, 3, 1, 2)
        .cuda()
    )
    mask_images = torch.as_tensor(
        np.array(mask_images, dtype=np.float16)
    ).cuda()
    if len(object_images) > 0:
        ip_image = Image.fromarray(object_images[0])
        object_images = [image / 255.0 for image in object_images]
        object_images = (
            torch.as_tensor(np.array(object_images, dtype=np.float16))
            .permute(0, 3, 1, 2)
            .cuda()
        )

    with torch.inference_mode():
        if pipe_ver == 2:
            outputs = pipe(
                image=init_images,
                mask_image=mask_images,
                prompt=prompts,
                negative_prompt=negative_prompts,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                output_type="np",
                strength=denoise_strength,
            ).images
        elif isinstance(pipe, StableDiffusionInpaintPipeline):
            outputs = pipe(
                prompt=prompts,
                negative_prompt=negative_prompts,
                image=init_images,
                generator=torch_generator,
                mask_image=mask_images,
                ip_adapter_image=ip_image,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                output_type="np",
                strength=denoise_strength,
            ).images
        elif isinstance(pipe, StableDiffusionControlNetInpaintPipeline):
            outputs = pipe(
                prompt=prompts,
                negative_prompt=negative_prompts,
                num_inference_steps=num_inference_steps,
                generator=torch_generator,
                eta=1.0,
                image=init_images,
                mask_image=mask_images,
                control_image=init_images,
                ip_adapter_image=ip_image,
                output_type="np",
            ).images

            # outputs = pipe(
            #     prompt=prompts,
            #     negative_prompt=negative_prompts,
            #     image=init_images,
            #     mask_image=mask_images,
            #     control_image=object_images,
            #     ip_adapter_image=ip_image,
            #     controlnet_conditioning_scale=0.9,
            #     control_guidance_end=0.9,
            #     guidance_scale=guidance_scale,
            #     num_inference_steps=num_inference_steps,
            #     output_type="np",
            #     strength=denoise_strength,
            # ).images

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


def get_sd_pipeline(pipeline_opts):
    torch.cuda.empty_cache()

    model_id = pipeline_opts["model_id"]
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float16
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        model_id, subfolder="tokenizer", torch_dtype=torch.float16
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=torch.float16
    )
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=torch.float16
    )

    if pipeline_opts["model_safetensor_path"] is not None:
        pipe = StableDiffusionInpaintPipeline.from_singlefile(
            pipeline_opts["model_safetensor_path"]
        )
        return pipe

    if pipeline_opts["controlnet_id"] is not None:
        controlnet = ControlNetModel.from_pretrained(
            pipeline_opts["controlnet_id"], torch_dtype=torch.float16
        )

        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            model_id,
            # vae=vae,
            # text_encoder=text_encoder,
            # tokenizer=tokenizer,
            # unet=unet,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    else:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            # vae=vae,
            # text_encoder=text_encoder,
            # tokenizer=tokenizer,
            # unet=unet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )

    ip_adapter_id = pipeline_opts["ip_adapter_id"]
    if ip_adapter_id is not None and pipeline_opts["model_ver"] == 1.5:
        pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="models",
            weight_name=[ip_adapter_id],
            image_encoder_folder="models/image_encoder",
            torch_dtype=torch.float16,
        )
        pipe.set_ip_adapter_scale(0.5)

    return pipe


def get_blip(model_id):
    model = BlipForConditionalGeneration.from_pretrained(model_id)
    processor = BlipProcessor.from_pretrained(model_id)

    return model, processor


def get_torch_generator(seed, device="cpu"):
    return torch.Generator(device=device).manual_seed(seed)


def restore_object(
    image,
    mask,
    expand_direction,
    expand_pixels,
    prompt,
    negative_prompt,
    num_inference_steps,
    guidance_scale,
    denoise_strength,
    sampler,
    opts,
    torch_generator,
    pix2pix_model,
    pipeline,
    blip_model,
    blip_proccessor,
):
    unet_input_shape = [512, 512]
    expand_mask = get_expand_mask(mask, expand_direction, expand_pixels)

    final_h, final_w = expand_mask.shape[:2]
    expand_region = get_expand_region(
        image.shape[:2], expand_direction, expand_pixels
    )
    final_h, final_w = expand_mask.shape[:2]
    data = get_input(expand_mask, expand_region, [256, 256])
    expand_sdf_map = pix2pix_model.predict(data)

    complete_mask = get_binary_mask(expand_sdf_map)
    complete_mask = resize(complete_mask, [final_h, final_w])
    _, complete_mask = cv2.threshold(complete_mask, 128, 255, 0)

    expand_mask = np.where(expand_region == 255, complete_mask, 0)
    expand_mask = resize(expand_mask, unet_input_shape)
    _, expand_mask = cv2.threshold(expand_mask, 128, 255, 0)

    object_image = get_object_focus_image(image, mask)
    if prompt == "":
        prompt = generate_image_caption(
            blip_model, blip_proccessor, object_image, opts["device"]
        )

    print(prompt)

    image_filled = fill_img(image, mask, expand_direction, expand_pixels)

    cv2.imwrite("expand_region.png", expand_region)
    cv2.imwrite("expand_mask.png", expand_mask)
    cv2.imwrite("img_filled.png", image_filled)
    cv2.imwrite("object_image.png", object_image.astype(np.uint8))

    object_image = resize(object_image, unet_input_shape)
    if opts["sd"]["ip_adapter_id"] is None:
        object_images = []
    else:
        object_images = [object_image]

    image_filled = resize(image_filled, unet_input_shape)
    image_filled = image_filled.astype(np.float16) / 255.0

    # prompt = ""
    negative_prompt = negative_prompt + opts["blip"]["default_negative_prompt"]
    result_image = perform_outpaint(
        pipe=pipeline,
        pipe_ver=opts["sd"]["model_ver"],
        torch_generator=torch_generator,
        init_images=[image_filled],
        mask_images=[expand_mask],
        prompts=[prompt],
        negative_prompts=[negative_prompt],
        object_images=object_images,
        sampler=sampler,
        num_inference_steps=num_inference_steps,
        denoise_strength=denoise_strength,
        guidance_scale=guidance_scale,
    )[0]

    result_image = resize(result_image, [final_h, final_w])
    cv2.imwrite("result.png", result_image)

    return cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

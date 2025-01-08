import argparse
import os

import yaml
import gradio as gr
import numpy as np
import cv2

from src.model import Pix2PixModel
from src import utils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pix2pix-config", type=str, default="configs/pix2pix.yaml"
    )
    parser.add_argument(
        "--sd-pipeline-config", type=str, default="configs/sd_pipeline.yaml"
    )
    parser.add_argument("--blip-config", type=str, default="configs/blip.yaml")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    return args


def init_models(args):
    assert os.path.exists(
        args.pix2pix_config
    ), f"Cannot find {args.pix2pix_config}"
    assert os.path.exists(
        args.sd_pipeline_config
    ), f"Cannot find {args.sd_pipeline_config}"
    assert os.path.exists(args.blip_config), f"Cannot find {args.blip_config}"

    with open(args.pix2pix_config) as pix2pix_file:
        pix2pix_opts = yaml.safe_load(pix2pix_file)

    with open(args.sd_pipeline_config) as sd_pipeline_file:
        sd_pipeline_opts = yaml.safe_load(sd_pipeline_file)

    with open(args.blip_config) as blip_file:
        blip_opts = yaml.safe_load(blip_file)

    opts = dict()
    opts["pix2pix"] = pix2pix_opts
    opts["sd"] = sd_pipeline_opts
    opts["blip"] = blip_opts
    opts["device"] = args.device

    pix2pix_model = Pix2PixModel(opts["pix2pix"])
    pipeline = utils.get_sd_pipeline(opts["sd"]["model_id"], opts["sd"]["seed"])
    blip_model, blip_proccessor = utils.get_blip(opts["blip"]["model_id"])

    return pix2pix_model, pipeline, blip_model, blip_proccessor


def process_image(
    img_with_mask,
    expand_direction,
    expand_pixels,
    prompt,
    negative_prompt,
    num_inference_steps,
    guidance_scale,
    denoise_strength,
    sampler,
):
    expand_pixels = int(expand_pixels)
    image = img_with_mask["background"]

    mask = cv2.cvtColor(img_with_mask["layers"][0], cv2.COLOR_BGR2GRAY)
    expand_mask = utils.get_expand_mask(mask, expand_direction, expand_pixels)

    final_h, final_w = expand_mask.shape[:2]
    expand_region = utils.get_expand_region(
        image.shape[:2], expand_direction, expand_pixels
    )
    final_h, final_w = expand_mask.shape[:2]
    data = utils.get_input(expand_mask, expand_region, [256, 256])
    expand_sdf_map = pix2pix_model.predict(data)

    complete_mask = utils.get_binary_mask(expand_sdf_map)
    complete_mask = utils.resize(complete_mask, [final_h, final_w])
    _, complete_mask = cv2.threshold(complete_mask, 128, 255, 0)

    expand_mask = np.where(expand_region == 0, complete_mask, 0)

    image_filled = utils.fill_img(image, mask, expand_direction, expand_pixels)
    print(image_filled.shape)
    caption = utils.generate_image_caption(
        blip_model, blip_proccessor, image_filled, device
    )

    neg_prompt = "worst quality, low quality, illustration, 3d, 2d, painting, cartoons, text, sketch, open mouth"

    result_image = utils.restore_from_mask(
        pipe=pipeline,
        init_images=[image_filled],
        mask_images=[expand_mask],
        prompts=[caption],
        sampler=sampler,
        num_inference_steps=num_inference_steps,
        denoise_strength=denoise_strength,
        guidance_scale=guidance_scale,
    )

    result_image = utils.resize(result_image, [final_h, final_w])
    cv2.imwrite("result.png", result_image)


with gr.Blocks() as demo:
    gr.Markdown("# Stable Diffusion Inpainting Demo")
    with gr.Row():
        with gr.Column():
            img_with_mask = gr.ImageEditor(
                label="Image for inpainting with mask",
                sources=["upload"],
                type="numpy",
                image_mode="RGB",
                brush=gr.Brush(colors=["#ffffff"], color_mode="fixed"),
                eraser=gr.Eraser(),
                layers=False,
            )
            expand_direction = gr.Radio(
                label="Direction to expand image", choices=["Left", "Right"]
            )
            expand_pixels = gr.Number(
                label="Number of pixels to expand",
                minimum=0,
                maximum=1024,
                precision=0,
            )
            prompt = gr.Textbox(
                label="Prompt", placeholder="Enter your prompt here"
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                placeholder="Enter your negative prompt here",
            )
            num_inference_steps = gr.Slider(
                minimum=1,
                maximum=100,
                value=30,
                label="Number of Inference Steps",
            )
            guidance_scale = gr.Slider(
                minimum=1.0, maximum=20.0, value=7.5, label="Guidance Scale"
            )
            denoise_strength = gr.Slider(
                minimum=0.0, maximum=1.0, value=1.0, label="Denoise Strength"
            )
            sampler = gr.Dropdown(
                choices=["euler", "plms", "ddim"],
                label="Sampler",
                value="euler",
            )
            submit = gr.Button("Generate")
        with gr.Column():
            output = gr.Image(label="Result")

    submit.click(
        fn=process_image,
        inputs=[
            img_with_mask,
            expand_direction,
            expand_pixels,
            prompt,
            negative_prompt,
            num_inference_steps,
            guidance_scale,
            denoise_strength,
            sampler,
        ],
        outputs=output,
    )

if __name__ == "__main__":
    args = get_args()
    pix2pix_model, pipeline, blip_model, blip_proccessor = init_models(args)
    device = args.device
    if device != "cpu":
        device = f"cuda:{device}"
    demo.launch(share=True)

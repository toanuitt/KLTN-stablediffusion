import argparse
import os

import yaml
import gradio as gr
import numpy as np
import cv2

from src.model import Pix2PixModel
from src import utils
from src.segmentation import *
from src.button import create_control_elements


opts = pix2pix_model = blip_model = blip_proccessor = pipeline = None


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

    if args.device == "cpu":
        opts["device"] = args.device
    else:
        opts["device"] = f"cuda:{args.device}"

    pix2pix_model = Pix2PixModel(opts["pix2pix"])
    pipeline = utils.get_sd_pipeline(opts["sd"])
    blip_model, blip_proccessor = utils.get_blip(opts["blip"]["model_id"])

    pipeline.to(opts["device"])
    blip_model.to(opts["device"])

    return opts, pix2pix_model, pipeline, blip_model, blip_proccessor


def process_image_mask(
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
    global opts, pix2pix_model, pipeline, blip_model, blip_proccessor

    expand_pixels = int(expand_pixels)
    image = img_with_mask["background"]
    unet_input_shape = [512, 512]

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

    expand_mask = np.where(expand_region == 255, complete_mask, 0)
    expand_mask = utils.resize(expand_mask, unet_input_shape)
    _, expand_mask = cv2.threshold(expand_mask, 128, 255, 0)

    object_image = utils.get_object_focus_image(image, mask)
    print(object_image.shape)
    if prompt == "":
        prompt = utils.generate_image_caption(
            blip_model, blip_proccessor, object_image, opts["device"]
        )

    print(prompt)

    image_filled = utils.fill_img(image, mask, expand_direction, expand_pixels)

    cv2.imwrite("expand_region.png", expand_region)
    cv2.imwrite("expand_mask.png", expand_mask)
    cv2.imwrite("img_filled.png", image_filled)
    cv2.imwrite("object_image.png", object_image.astype(np.uint8))

    if opts["sd"]["ip_adapter_id"] is None:
        object_images = []
    else:
        object_images = [object_image]

    image_filled = utils.resize(image_filled, unet_input_shape)
    image_filled = image_filled.astype(np.float16) / 255.0

    negative_prompt = negative_prompt + opts["blip"]["default_negative_prompt"]
    result_image = utils.restore_from_mask(
        pipe=pipeline,
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

    result_image = utils.resize(result_image, [final_h, final_w])
    cv2.imwrite("result.png", result_image)
    return cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)


def process_image_yolo(
    img_upload,
    mask,
    expand_direction,
    expand_pixels,
    prompt,
    negative_prompt,
    num_inference_steps,
    guidance_scale,
    denoise_strength,
    sampler,
):
    mask_output = mask
    cv2.imwrite("result.png", mask_output)
    return img_upload


stored_masks = []

css = """
#image-editor {
    display: block;
    margin-left: auto;
    margin-right: auto;
}
"""
with gr.Blocks() as demo:
    gr.Markdown("# Stable Diffusion Inpainting Demo")
    with gr.Row():
        with gr.Column():
            with gr.Tabs() as tabs:
                with gr.Tab("FlexMask Inpaint"):
                    img_with_mask = gr.ImageEditor(
                        label="Upload image",
                        sources=["upload"],
                        type="numpy",
                        image_mode="RGB",
                        brush=gr.Brush(colors=["#ffffff"], color_mode="fixed"),
                        eraser=gr.Eraser(),
                        layers=False,
                        height=512,
                        width=1024,
                        transforms=[],
                        elem_id="image-editor",
                    )
                    inpaint_controls = create_control_elements()
                    submit_inpaint = gr.Button("Generate (FlexMask)")

                with gr.Tab("YOLODraw Inpaint"):
                    img_upload = gr.Image(
                        label="Upload image",
                        sources=["upload"],
                        type="numpy",
                        image_mode="RGB",
                    )
                    detect_btn = gr.Button("Detect Objects")
                    class_dropdown = gr.Dropdown(
                        label="Select object", choices=[], type="index"
                    )
                    mask_output = gr.Image(
                        label="Masked Image", type="numpy", image_mode="RGB"
                    )
                    masked_region = gr.Image(
                        label="Masked Region", type="numpy", image_mode="RGB"
                    )
                    upload_controls = create_control_elements()
                    submit_upload = gr.Button("Generate (YOLODraw)")
        with gr.Column():
            output = gr.Image(label="Result")
    detect_btn.click(
        fn=detect_objects, inputs=[img_upload], outputs=[class_dropdown]
    )

    class_dropdown.change(
        fn=update_mask,
        inputs=[img_upload, class_dropdown],
        outputs=[mask_output, masked_region],
    )
    submit_inpaint.click(
        fn=process_image_mask,
        inputs=[
            img_with_mask,
            *inpaint_controls,
        ],
        outputs=output,
    )
    submit_upload.click(
        fn=process_image_yolo,
        inputs=[
            img_upload,
            mask_output,
            *upload_controls,
        ],
        outputs=output,
    )

if __name__ == "__main__":
    args = get_args()
    opts, pix2pix_model, pipeline, blip_model, blip_proccessor = init_models(
        args
    )
    demo.launch(share=True)

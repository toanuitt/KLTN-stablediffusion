import argparse
import os

import yaml
import gradio as gr
import cv2

from src.model import Pix2PixModel
from src import utils
from src.segmentation import *
from src.button import create_control_elements


opts = dict()
torch_generator = None
pix2pix_model = None
blip_model = None
blip_proccessor = None
pipeline = None

stored_masks = []

css = """
#image-editor {
    display: block;
    margin-left: auto;
    margin-right: auto;
}
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pix2pix-config", type=str, default="configs/pix2pix.yaml"
    )
    parser.add_argument(
        "--sd-pipeline-config", type=str, default="configs/sd_pipeline.yaml"
    )
    parser.add_argument("--blip-config", type=str, default="configs/blip.yaml")
    parser.add_argument("--yolo-model", type=str, default="configs/yolo11.yaml")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=69)
    args = parser.parse_args()
    return args


def init_models(args):
    global opts, torch_generator, pix2pix_model, pipeline, blip_model, blip_proccessor
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

    with open(args.yolo_model) as yolo_file:
        yolo_opts = yaml.safe_load(yolo_file)

    init_opts = dict()
    init_opts["pix2pix"] = pix2pix_opts
    init_opts["sd"] = sd_pipeline_opts
    init_opts["blip"] = blip_opts
    init_opts["yolo"] = yolo_opts
    init_opts["seed"] = args.seed

    if args.device == "cpu":
        init_opts["device"] = args.device
    else:
        init_opts["device"] = f"cuda:{args.device}"

    initialize_yolo(yolo_opts)
    pix2pix_model = Pix2PixModel(init_opts["pix2pix"])
    pipeline = utils.get_sd_pipeline(init_opts["sd"])
    blip_model, blip_proccessor = utils.get_blip(init_opts["blip"]["model_id"])

    pipeline.to(init_opts["device"])
    blip_model.to(init_opts["device"])

    torch_generator = utils.get_torch_generator(init_opts["seed"])

    opts = init_opts


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
    global opts, torch_generator, pix2pix_model, pipeline, blip_model, blip_proccessor

    expand_pixels = int(expand_pixels)
    image = img_with_mask["background"]
    mask = cv2.cvtColor(img_with_mask["layers"][0], cv2.COLOR_BGR2GRAY)

    result_image = utils.restore_object(
        image=image,
        mask=mask,
        expand_direction=expand_direction,
        expand_pixels=expand_pixels,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        denoise_strength=denoise_strength,
        sampler=sampler,
        opts=opts,
        torch_generator=torch_generator,
        pix2pix_model=pix2pix_model,
        pipeline=pipeline,
        blip_model=blip_model,
        blip_proccessor=blip_proccessor,
    )

    return result_image


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
    global opts, torch_generator, pix2pix_model, pipeline, blip_model, blip_proccessor

    expand_pixels = int(expand_pixels)
    image = img_upload
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    result_image = utils.restore_object(
        image=image,
        mask=mask,
        expand_direction=expand_direction,
        expand_pixels=expand_pixels,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        denoise_strength=denoise_strength,
        sampler=sampler,
        opts=opts,
        torch_generator=torch_generator,
        pix2pix_model=pix2pix_model,
        pipeline=pipeline,
        blip_model=blip_model,
        blip_proccessor=blip_proccessor,
    )

    return result_image


if __name__ == "__main__":
    args = get_args()
    init_models(args)

    with gr.Blocks() as demo:
        gr.Markdown("# Stable Diffusion Inpainting Demo")
        with gr.Row():
            with gr.Column():
                with gr.Tabs() as tabs:
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
                        temp_class_dropdown = gr.Dropdown()
                        mask_output = gr.Image(
                            label="Masked Image", type="numpy", image_mode="RGB"
                        )
                        masked_region = gr.Image(
                            label="Masked Region",
                            type="numpy",
                            image_mode="RGB",
                        )
                        upload_controls = create_control_elements()
                        submit_upload = gr.Button("Generate (YOLODraw)")

                    with gr.Tab("FlexMask Inpaint"):
                        img_with_mask = gr.ImageEditor(
                            label="Upload image",
                            sources=["upload"],
                            type="numpy",
                            image_mode="RGB",
                            brush=gr.Brush(
                                colors=["#ffffff"], color_mode="fixed"
                            ),
                            eraser=gr.Eraser(),
                            layers=False,
                            height=512,
                            width=1024,
                            transforms=[],
                            elem_id="image-editor",
                        )
                        inpaint_controls = create_control_elements()
                        submit_inpaint = gr.Button("Generate (FlexMask)")

            with gr.Column():
                output = gr.Image(label="Result")

        img_upload.clear(
            fn=clear_state, inputs=[class_dropdown], outputs=[class_dropdown]
        )

        detect_btn.click(
            fn=detect_objects,
            inputs=[img_upload],
            outputs=[class_dropdown],
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

    demo.launch(share=True)

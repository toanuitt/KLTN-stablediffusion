import argparse
import os

import yaml
import gradio as gr
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
    parser.add_argument("--yolo-model", type=str, default="configs/yolo11.yaml")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=69)
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

    with open(args.yolo_model) as yolo_file:
        yolo_opts = yaml.safe_load(yolo_file)

    opts = dict()
    opts["pix2pix"] = pix2pix_opts
    opts["sd"] = sd_pipeline_opts
    opts["blip"] = blip_opts
    opts["yolo"] = yolo_opts
    opts["seed"] = args.seed

    if args.device == "cpu":
        opts["device"] = args.device
    else:
        opts["device"] = f"cuda:{args.device}"

    initialize_yolo(yolo_opts)
    pix2pix_model = Pix2PixModel(opts["pix2pix"])
    pipeline = utils.get_sd_pipeline(opts["sd"])
    blip_model, blip_proccessor = utils.get_blip(opts["blip"]["model_id"])

    pipeline.to(opts["device"])
    blip_model.to(opts["device"])

    torch_generator = utils.get_torch_generator(opts["seed"])

    return (
        opts,
        torch_generator,
        pix2pix_model,
        pipeline,
        blip_model,
        blip_proccessor,
    )


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
                        label="Masked Region", type="numpy", image_mode="RGB"
                    )
                    upload_controls = create_control_elements()
                    submit_upload = gr.Button("Generate (YOLODraw)")

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

        with gr.Column():
            output = gr.Image(label="Result")

    img_upload.clear(
        fn=clear_state, inputs=[class_dropdown], outputs=[class_dropdown]
    )

    detect_btn.click(
        fn=detect_objects,
        inputs=[img_upload, opts["device"]],
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

if __name__ == "__main__":
    args = get_args()
    (
        opts,
        torch_generator,
        pix2pix_model,
        pipeline,
        blip_model,
        blip_proccessor,
    ) = init_models(args)
    demo.launch(share=True)

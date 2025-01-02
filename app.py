import gradio as gr
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np

# Initialize the pipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)

if torch.cuda.is_available():
    pipe = pipe.to("cuda")

def inpaint(
    image,
    mask,
    prompt,
    negative_prompt,
    num_inference_steps,
    guidance_scale,
    seed
):
    # Convert mask to black and white
    if mask is not None:
        mask = Image.fromarray(mask).convert("L")
        mask = Image.fromarray(np.array(mask) * 255)
    
    # Convert numpy array to PIL Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Set the random seed for reproducibility
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)
    else:
        generator = None
    
    # Run inference
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]
    
    return output

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Stable Diffusion v2 Inpainting")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", type="numpy")
            mask_image = gr.Image(
                label="Draw Mask",
                type="numpy",
                tool="sketch",
                brush_radius=20,
            )
            prompt = gr.Textbox(label="Prompt")
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                value="low quality, bad anatomy, bad hands, cropped, worst quality"
            )
            
            with gr.Row():
                num_inference_steps = gr.Slider(
                    label="Number of Steps",
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=30
                )
                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=1,
                    maximum=20,
                    step=0.5,
                    value=7.5
                )
                seed = gr.Number(
                    label="Seed (blank for random)",
                    precision=0
                )
            
            run_button = gr.Button("Generate")
        
        with gr.Column():
            output_image = gr.Image(label="Output")
    
    run_button.click(
        fn=inpaint,
        inputs=[
            input_image,
            mask_image,
            prompt,
            negative_prompt,
            num_inference_steps,
            guidance_scale,
            seed
        ],
        outputs=output_image,
    )

if __name__ == "__main__":
    demo.launch()